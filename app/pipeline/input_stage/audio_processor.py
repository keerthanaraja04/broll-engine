"""
Audio processor for the input stage.

Responsibilities (in order)
----------------------------
1. validate_audio()       — extension check + ffprobe duration guard
2. convert_to_wav()       — ffmpeg → 16 kHz mono WAV (idempotent if already wav)
3. detect_silence()       — ffmpeg silencedetect; raise AudioSilentError if needed
4. chunk_at_silence()     — split oversized files at silence points
5. transcribe_chunks()    — call STTAdapter on each chunk; merge results
6. check_transcript()     — raise AudioNoSpeechError for low-confidence results
7. process_audio()        — public façade: runs all steps, returns Transcript

All ffmpeg/ffprobe calls run in a thread pool (asyncio.run_in_executor) so
they never block the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from app.adapters.base import TranscriptResult
from app.adapters.stt_adapter import get_stt_adapter
from app.core.config import get_settings
from app.core.errors import (
    AudioNoSpeechError,
    AudioSilentError,
    AudioTooLongError,
    MediaProcessingError,
    WrongFileTypeError,
)
from app.core.logging import get_logger
from app.pipeline.input_stage.transcript import (
    Sentence,
    Transcript,
    WordTiming,
    sentences_from_words,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    s = get_settings()
    cmd = [s.FFMPEG_BINARY, "-y", *args]
    try:
        return subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            f"ffmpeg not found at '{s.FFMPEG_BINARY}'",
            details={"binary": s.FFMPEG_BINARY},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(
            f"ffmpeg timed out after {timeout}s",
            details={"cmd": " ".join(cmd)},
        ) from exc


def _run_ffprobe(*args: str, timeout: int = 30) -> dict[str, Any]:
    s = get_settings()
    cmd = [s.FFPROBE_BINARY, "-v", "quiet", "-print_format", "json", *args]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            f"ffprobe not found at '{s.FFPROBE_BINARY}'",
            details={"binary": s.FFPROBE_BINARY},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError("ffprobe timed out") from exc

    if result.returncode != 0:
        raise MediaProcessingError(
            "ffprobe failed",
            details={"stderr": result.stderr.decode(errors="replace")[:400]},
        )
    try:
        return json.loads(result.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise MediaProcessingError("ffprobe output is not valid JSON") from exc


def _probe_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe."""
    probe = _run_ffprobe("-show_format", "-show_streams", path)
    # Try format-level duration first (most reliable)
    dur_str = probe.get("format", {}).get("duration")
    if dur_str is None:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") in ("audio", "video"):
                dur_str = stream.get("duration")
                break
    try:
        return float(dur_str or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _probe_file_format(path: str) -> str:
    """Return the format name string from ffprobe, e.g. 'mp3', 'wav'."""
    probe = _run_ffprobe("-show_format", path)
    return probe.get("format", {}).get("format_name", "").lower()


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _parse_silence_intervals(stderr_text: str) -> list[SilenceInterval]:
    """
    Parse ffmpeg silencedetect stderr output into SilenceInterval list.

    Each silence is reported as pairs of lines:
        silence_start: 0.5
        silence_end: 1.2 | silence_duration: 0.7
    """
    starts: list[float] = []
    ends: list[float] = []

    for line in stderr_text.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
        if m_end:
            ends.append(float(m_end.group(1)))

    # Pair up starts and ends; if a silence extends to end-of-file, end = ∞
    intervals: list[SilenceInterval] = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else float("inf")
        intervals.append(SilenceInterval(start=start, end=end))
    return intervals


# ---------------------------------------------------------------------------
# Step 1 — Validation
# ---------------------------------------------------------------------------

def validate_audio(path: str) -> float:
    """
    Validate extension and duration.

    Returns duration in seconds.
    Raises WrongFileTypeError or AudioTooLongError.
    """
    s = get_settings()
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext not in s.AUDIO_ALLOWED_EXTENSIONS:
        raise WrongFileTypeError(
            f"Audio extension '.{ext}' is not accepted. "
            f"Allowed: {', '.join(s.AUDIO_ALLOWED_EXTENSIONS)}",
            details={"extension": ext, "allowed": s.AUDIO_ALLOWED_EXTENSIONS},
        )

    duration = _probe_duration(path)
    if duration > s.AUDIO_MAX_DURATION_SECONDS:
        raise AudioTooLongError(
            f"Audio duration {duration:.1f}s exceeds the "
            f"{s.AUDIO_MAX_DURATION_SECONDS:.0f}s limit",
            details={"duration_seconds": duration,
                     "max_seconds": s.AUDIO_MAX_DURATION_SECONDS},
        )
    return duration


# ---------------------------------------------------------------------------
# Step 2 — Conversion
# ---------------------------------------------------------------------------

def convert_to_wav(path: str, output_dir: str | None = None) -> str:
    """
    Convert *path* to 16 kHz mono PCM WAV.

    If the file is already a 16 kHz mono WAV it is re-encoded anyway to
    guarantee a clean baseline for silence detection and Whisper.

    Returns the path to the output WAV file.
    """
    s = get_settings()
    tmp_dir = output_dir or s.AUDIO_TEMP_DIR or tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(tmp_dir, f"{base}_16k_mono.wav")

    result = _run_ffmpeg(
        "-i", path,
        "-ar", str(s.AUDIO_CONVERT_SAMPLE_RATE),
        "-ac", str(s.AUDIO_CONVERT_CHANNELS),
        "-vn",           # strip video track
        out_path,
    )
    if result.returncode != 0:
        raise MediaProcessingError(
            "Audio conversion to WAV failed",
            details={"stderr": result.stderr.decode(errors="replace")[:500]},
        )
    logger.info("audio_converted", extra={"input": path, "output": out_path})
    return out_path


# ---------------------------------------------------------------------------
# Step 3 — Silence detection
# ---------------------------------------------------------------------------

def detect_silence(
    wav_path: str,
    duration: float,
) -> list[SilenceInterval]:
    """
    Run ffmpeg silencedetect and return silence intervals.

    Raises AudioSilentError when the file is effectively all-silence.
    """
    s = get_settings()

    result = _run_ffmpeg(
        "-i", wav_path,
        "-af", (
            f"silencedetect=noise={s.SILENCE_DETECT_NOISE_DB}dB"
            f":d={s.SILENCE_DETECT_MIN_DURATION}"
        ),
        "-f", "null", "-",
    )

    # ffmpeg writes silencedetect output to stderr
    stderr = result.stderr.decode(errors="replace")
    intervals = _parse_silence_intervals(stderr)

    # Check for all-silence condition
    total_silent = sum(
        min(iv.end, duration) - iv.start
        for iv in intervals
        if iv.start < duration
    )
    if duration > 0 and total_silent / duration >= s.SILENCE_DETECT_SILENT_RATIO:
        raise AudioSilentError(
            f"Audio is {total_silent/duration*100:.0f}% silent — no audible content found",
            details={
                "duration_seconds": duration,
                "silent_seconds": round(total_silent, 2),
                "silent_ratio": round(total_silent / duration, 3),
            },
        )

    logger.info(
        "silence_detected",
        extra={"intervals": len(intervals), "total_silent_s": round(total_silent, 2)},
    )
    return intervals


# ---------------------------------------------------------------------------
# Step 4 — Chunking
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    path: str
    start_offset: float   # seconds from original file start
    duration: float


def chunk_at_silence(
    wav_path: str,
    file_size_bytes: int,
    silence_intervals: list[SilenceInterval],
    total_duration: float,
    output_dir: str | None = None,
) -> list[AudioChunk]:
    """
    Split *wav_path* into chunks that each fit within AUDIO_GROQ_MAX_BYTES.

    Split points are chosen from the midpoint of the longest available
    silence intervals.  Chunks are trimmed from the WAV using ffmpeg.

    Returns a list of AudioChunk (path + start_offset + duration).
    If the file is already small enough, returns a single-element list
    containing the original path.
    """
    s = get_settings()

    if file_size_bytes <= s.AUDIO_GROQ_MAX_BYTES:
        return [AudioChunk(path=wav_path, start_offset=0.0, duration=total_duration)]

    tmp_dir = output_dir or s.AUDIO_TEMP_DIR or tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(wav_path))[0]

    # Estimate bytes-per-second from file size and duration
    if total_duration > 0:
        bps = file_size_bytes / total_duration
    else:
        bps = 32_000.0  # 16 kHz mono 16-bit PCM fallback

    max_chunk_duration = min(
        s.AUDIO_CHUNK_MAX_DURATION_SECONDS,
        s.AUDIO_GROQ_MAX_BYTES / bps,
    )

    # Build split points: midpoint of silences that are long enough
    split_candidates = sorted(
        [iv for iv in silence_intervals if iv.duration >= s.AUDIO_CHUNK_MIN_SILENCE_SECONDS],
        key=lambda iv: iv.start,
    )

    # Walk through the file, collecting split points
    split_times: list[float] = [0.0]
    last_split = 0.0

    for iv in split_candidates:
        split_point = (iv.start + iv.end) / 2.0
        if split_point - last_split >= max_chunk_duration * 0.8:
            split_times.append(split_point)
            last_split = split_point

    split_times.append(total_duration)

    # Deduplicate and sort
    split_times = sorted(set(split_times))

    chunks: list[AudioChunk] = []
    for i in range(len(split_times) - 1):
        chunk_start = split_times[i]
        chunk_end = split_times[i + 1]
        chunk_dur = chunk_end - chunk_start
        out_path = os.path.join(tmp_dir, f"{base}_chunk{i:03d}.wav")

        result = _run_ffmpeg(
            "-i", wav_path,
            "-ss", f"{chunk_start:.6f}",
            "-t", f"{chunk_dur:.6f}",
            "-c", "copy",
            out_path,
        )
        if result.returncode != 0:
            raise MediaProcessingError(
                f"Audio chunking failed at chunk {i}",
                details={"stderr": result.stderr.decode(errors="replace")[:300]},
            )
        chunks.append(
            AudioChunk(path=out_path, start_offset=chunk_start, duration=chunk_dur)
        )
        logger.debug(
            "audio_chunk_created",
            extra={"chunk": i, "start": chunk_start, "dur": chunk_dur, "path": out_path},
        )

    logger.info("audio_chunked", extra={"n_chunks": len(chunks)})
    return chunks


# ---------------------------------------------------------------------------
# Step 5 — Transcription + merge
# ---------------------------------------------------------------------------

async def transcribe_chunks(
    chunks: list[AudioChunk],
    language: str | None = None,
) -> TranscriptResult:
    """
    Transcribe each chunk sequentially, then merge into a single
    TranscriptResult by offsetting all timestamps by chunk.start_offset.
    """
    from app.adapters.base import TranscriptSegment, WordTiming as AdapterWord

    stt = get_stt_adapter()
    merged_words: list[AdapterWord] = []
    merged_segments: list[TranscriptSegment] = []
    detected_language = language or "en"
    total_duration = 0.0
    full_text_parts: list[str] = []
    seg_id_offset = 0

    for chunk in chunks:
        logger.info(
            "transcribing_chunk",
            extra={"start": chunk.start_offset, "dur": chunk.duration,
                   "path": chunk.path},
        )
        result = await stt.transcribe(chunk.path, language=language)
        if result.language and result.language != "en":
            detected_language = result.language
        total_duration = max(total_duration, chunk.start_offset + result.duration)
        full_text_parts.append(result.text)

        offset = chunk.start_offset
        for w in result.words:
            merged_words.append(
                AdapterWord(word=w.word, start=w.start + offset, end=w.end + offset)
            )
        for seg in result.segments:
            merged_segments.append(
                TranscriptSegment(
                    id=seg_id_offset + seg.id,
                    text=seg.text,
                    start=seg.start + offset,
                    end=seg.end + offset,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    words=[
                        AdapterWord(word=w.word, start=w.start + offset, end=w.end + offset)
                        for w in seg.words
                    ],
                )
            )
        seg_id_offset += len(result.segments)

    from app.adapters.base import TranscriptResult as AdapterResult
    return AdapterResult(
        text=" ".join(full_text_parts),
        language=detected_language,
        duration=total_duration,
        words=merged_words,
        segments=merged_segments,
    )


# ---------------------------------------------------------------------------
# Step 6 — Quality check
# ---------------------------------------------------------------------------

def check_transcript_quality(result: TranscriptResult) -> list[str]:
    """
    Inspect STT result for no-speech / low-confidence conditions.

    Returns a list of warning strings (empty = all good).
    Raises AudioNoSpeechError when confidence is catastrophically low.
    """
    s = get_settings()
    warnings: list[str] = []

    # Empty transcript
    if not result.text.strip():
        raise AudioNoSpeechError(
            "Transcription returned empty text — no speech detected",
            details={"language": result.language},
        )

    # Low average log-probability across all segments
    if result.segments:
        avg_logprob = sum(seg.avg_logprob for seg in result.segments) / len(result.segments)
        if avg_logprob < s.AUDIO_MIN_AVG_LOGPROB:
            raise AudioNoSpeechError(
                f"Transcript confidence too low (avg_logprob={avg_logprob:.3f}, "
                f"threshold={s.AUDIO_MIN_AVG_LOGPROB})",
                details={"avg_logprob": avg_logprob, "threshold": s.AUDIO_MIN_AVG_LOGPROB},
            )

        # Fraction of segments with high no-speech probability
        high_no_speech = sum(
            1 for seg in result.segments if seg.no_speech_prob > 0.8
        )
        no_speech_fraction = high_no_speech / len(result.segments)
        if no_speech_fraction >= s.AUDIO_MAX_NO_SPEECH_FRACTION:
            raise AudioNoSpeechError(
                f"{no_speech_fraction*100:.0f}% of transcript segments have high "
                f"no_speech probability — audio likely contains no recognisable speech",
                details={"no_speech_fraction": round(no_speech_fraction, 3)},
            )
        if no_speech_fraction > 0.3:
            warnings.append(
                f"{no_speech_fraction*100:.0f}% of segments have elevated "
                f"no_speech_prob — transcript may contain errors"
            )

    # Very few words
    if len(result.words) < 3:
        warnings.append(
            f"Only {len(result.words)} word(s) detected — transcript may be incomplete"
        )

    return warnings


# ---------------------------------------------------------------------------
# Step 7 — Public façade
# ---------------------------------------------------------------------------

def _adapter_words_to_pipeline(words: list) -> list[WordTiming]:
    return [WordTiming(word=w.word, start=w.start, end=w.end) for w in words]


async def process_audio(
    audio_path: str,
    language: str | None = None,
    output_dir: str | None = None,
) -> Transcript:
    """
    Full audio ingestion pipeline.

    1. Validate extension + duration.
    2. Convert to 16 kHz mono WAV.
    3. Silence detection (raises AudioSilentError if all-silence).
    4. Chunk at silence if file > AUDIO_GROQ_MAX_BYTES.
    5. Transcribe each chunk and merge timestamps.
    6. Quality-check transcript (raises AudioNoSpeechError if needed).
    7. Build and return Transcript.

    Args:
        audio_path:   Absolute path to the input audio file.
        language:     Optional ISO 639-1 language hint.
        output_dir:   Directory for temp WAV / chunk files.
                      Defaults to AUDIO_TEMP_DIR or system tmp.

    Returns:
        Transcript — used by all downstream stages.
    """
    loop = asyncio.get_running_loop()

    # Steps 1–4 are CPU/subprocess-bound → run in thread pool
    def _sync_pipeline() -> tuple[str, list[AudioChunk], float]:
        duration = validate_audio(audio_path)
        wav_path = convert_to_wav(audio_path, output_dir)
        file_size = os.path.getsize(wav_path)
        silences = detect_silence(wav_path, duration)
        chunks = chunk_at_silence(
            wav_path, file_size, silences, duration, output_dir
        )
        return wav_path, chunks, duration

    wav_path, chunks, duration = await loop.run_in_executor(None, _sync_pipeline)

    # Step 5 — async STT
    adapter_result = await transcribe_chunks(chunks, language=language)

    # Step 6 — quality check (sync, fast)
    warnings = check_transcript_quality(adapter_result)

    # Step 7 — build Transcript
    pipeline_words = _adapter_words_to_pipeline(adapter_result.words)
    sentences = sentences_from_words(pipeline_words)

    # Map adapter segments → Sentences where we don't have sentence-level splits
    if not sentences and adapter_result.segments:
        sentences = [
            Sentence(
                id=i,
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=[
                    WordTiming(word=w.word, start=w.start, end=w.end)
                    for w in seg.words
                ],
            )
            for i, seg in enumerate(adapter_result.segments)
        ]

    transcript = Transcript(
        language=adapter_result.language,
        duration=adapter_result.duration,
        words=pipeline_words,
        sentences=sentences,
        source="audio",
        audio_path=wav_path,
        warnings=warnings,
    )

    logger.info(
        "audio_processed",
        extra={
            "language": transcript.language,
            "duration": transcript.duration,
            "words": transcript.word_count,
            "sentences": len(transcript.sentences),
            "warnings": len(warnings),
        },
    )
    return transcript
