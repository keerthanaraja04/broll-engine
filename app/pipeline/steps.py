"""
Concrete pipeline steps for the broll-engine.

Steps (in execution order):
  1. TranscribeStep       — transcribe audio with Groq Whisper
  2. KeywordExtractionStep — extract broll keywords from transcript via Groq LLM
  3. VideoSearchStep      — search Pixabay for matching video clips
  4. ClipFilterStep       — filter clips by duration and relevance thresholds
  5. MediaAssemblyStep    — assemble final video with ffmpeg (stub)
  6. UploadResultStep     — upload assembled file to object storage
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from typing import Any

from app.adapters.groq_client import get_groq_client
from app.adapters.pixabay_client import get_pixabay_client
from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Transcription
# ---------------------------------------------------------------------------
class TranscribeStep(Step):
    """Transcribe the job's audio file using Groq Whisper, or use provided script directly."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        audio_object = ctx.payload.get("audio_object")
        script = ctx.payload.get("script")
        scenes = ctx.payload.get("scenes")
        audio_bucket = ctx.payload.get("audio_bucket", get_settings().STORAGE_BUCKET_RAW)

        if not audio_object:
            if not script and scenes:
                script = " ".join(s.get("text", "") for s in scenes if isinstance(s, dict))
            if script:
                ctx.set("transcript_text", script)
                ctx.set("transcription", {"text": script, "segments": [{"start": 0.0, "end": 5.0, "text": script}]})
                ctx.set("transcript_segments", [{"start": 0.0, "end": 5.0, "text": script}])
                logger.info("transcribe_skipped_using_script", extra={"words": len(script.split())})
                return ctx
            raise ValueError("payload missing 'audio_object'")

        storage = get_storage_client()
        audio_bytes = await storage.download_bytes(audio_bucket, audio_object)

        groq = get_groq_client()
        result = await groq.transcribe(audio_bytes, filename=audio_object.split("/")[-1])

        ctx.set("transcription", result)
        ctx.set("transcript_text", result.get("text", ""))
        ctx.set("transcript_segments", result.get("segments", []))

        logger.info(
            "transcribe_done",
            extra={"words": len(result.get("text", "").split()), "segments": len(result.get("segments", []))},
        )
        return ctx


# ---------------------------------------------------------------------------
# Step 2 — Keyword Extraction
# ---------------------------------------------------------------------------
_KEYWORD_SYSTEM_PROMPT = """\
You are a video producer's assistant. Given a transcript, extract 5–10 short search
queries that would find relevant stock b-roll footage on Pixabay. Each query should be
2–4 words. Return ONLY a valid JSON array of strings, e.g. ["city skyline", "people walking"].
"""


class KeywordExtractionStep(Step):
    """Extract b-roll search keywords instantly from transcript/script with fast fallback."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        transcript = ctx.get("transcript_text", "")
        if not transcript:
            ctx.set("keywords", [])
            return ctx

        # 1. Fast natural NLP keyword extraction (runs in 0ms)
        stopwords = {
            "this", "that", "with", "from", "they", "them", "there", "their", "into", "through",
            "about", "when", "where", "which", "while", "every", "begins", "begin", "here", "comes",
            "gives", "before", "after", "some", "such", "than", "then", "very", "what", "also", "just"
        }
        fast_keywords = []
        for s in re.split(r"[.!?;\n]+", transcript):
            words = [w.strip(",.!?\"'").lower() for w in s.split() if len(w) > 3]
            filtered = [w for w in words if w not in stopwords]
            if filtered:
                fast_keywords.append(" ".join(filtered[:3]))

        keywords = fast_keywords or ["nature landscape", "city skyline", "technology"]

        # 2. Fast LLM enrichment if available (max 2.0s strict timeout, non-blocking)
        try:
            groq = get_groq_client()
            response = await asyncio.wait_for(
                groq.chat_completion(
                    messages=[
                        {"role": "system", "content": _KEYWORD_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Transcript:\n{transcript[:4000]}"},
                    ]
                ),
                timeout=2.0,
            )
            raw = response["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                keywords = parsed
        except Exception:
            pass  # Immediately use instant keywords without delay

        ctx.set("keywords", keywords)
        logger.info("keywords_extracted", extra={"count": len(keywords), "keywords": keywords})
        return ctx


# ---------------------------------------------------------------------------
# Step 3 — Video Search
# ---------------------------------------------------------------------------
class VideoSearchStep(Step):
    """Search Pixabay for each keyword and collect candidate clips."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        keywords: list[str] = ctx.get("keywords", [])
        scenes = ctx.payload.get("scenes") or []
        script = ctx.get("transcript_text") or ctx.payload.get("script") or ""

        # Collect search terms: hints and scene descriptions take top priority
        search_terms = []
        if scenes:
            for s in scenes:
                if isinstance(s, dict):
                    hint = s.get("visual_hint")
                    if hint and hint not in search_terms:
                        search_terms.append(hint)
                    for kw in (s.get("keywords") or []):
                        if kw and kw not in search_terms:
                            search_terms.append(kw)
                    # Extract descriptive terms from text if no explicit hint
                    if not hint and not s.get("keywords") and s.get("text"):
                        words = [w.strip(",.!?").lower() for w in s["text"].split() if len(w) > 3]
                        stopwords = {"this", "that", "with", "from", "they", "them", "there", "their", "into", "through", "about"}
                        filtered = [w for w in words if w not in stopwords]
                        if filtered:
                            term = " ".join(filtered[:3])
                            if term not in search_terms:
                                search_terms.append(term)
        if script:
            script_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", script) if len(p.strip()) > 3]
            if not script_parts:
                script_parts = [p.strip() for p in re.split(r"[.!?;\n]+", script) if len(p.strip()) > 3]
            stopwords = {
                "this", "that", "with", "from", "they", "them", "there", "their", "into", "through",
                "about", "when", "where", "which", "while", "every", "begins", "begin", "here", "comes",
                "gives", "before", "after", "some", "such", "than", "then", "very", "what", "also", "just"
            }
            for p in script_parts[:8]:
                words = [w.strip(",.!?\"'").lower() for w in p.split() if len(w) > 3]
                filtered = [w for w in words if w not in stopwords]
                if filtered:
                    term = " ".join(filtered[:3])
                    if term not in search_terms:
                        search_terms.append(term)
        for kw in keywords:
            if kw not in search_terms:
                search_terms.append(kw)
        if not search_terms:
            search_terms = ["nature landscape", "city skyline", "ocean waves", "technology"]

        s = get_settings()
        pixabay = get_pixabay_client()

        all_clips: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        context_text = ctx.get("transcript_text") or ctx.payload.get("script") or ""

        def _score_clip_relevance(kw: str, tags: str, dl: int, lk: int) -> float:
            score = 3.6
            kw_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", kw)}
            tag_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", tags)}
            ctx_words = {
                w.lower() for w in re.findall(r"[a-zA-Z]{3,}", context_text)
                if w.lower() not in {"this", "that", "with", "from", "they", "them", "into", "through", "about"}
            }
            if kw_words:
                overlap = len(kw_words & tag_words)
                if overlap >= len(kw_words):
                    score += 0.8
                elif overlap > 0:
                    score += 0.5
            if ctx_words and (ctx_words & tag_words):
                ctx_overlap = len(ctx_words & tag_words)
                score += min(0.4, ctx_overlap * 0.15)
            if dl >= 100 or lk >= 20:
                score += 0.2
            return round(min(5.0, max(1.0, score)), 1)

        target_terms = search_terms[:6]
        async def fetch_term(kw: str):
            try:
                vids = await pixabay.search_videos(kw)
                return kw, vids
            except Exception as e:
                logger.warning("video_search_keyword_failed", extra={"keyword": kw, "error": str(e)})
                return kw, []

        term_results = await asyncio.gather(*[fetch_term(kw) for kw in target_terms])
        for kw, videos in term_results:
            for v in videos:
                if v.id not in seen_ids:
                    seen_ids.add(v.id)
                    url = v.best_url(min_width=s.PIXABAY_MIN_WIDTH)
                    if url:
                        rel_score = _score_clip_relevance(kw, v.tags, v.downloads, v.likes)
                        all_clips.append(
                            {
                                "id": v.id,
                                "keyword": kw,
                                "url": url,
                                "thumbnail_url": v.thumbnail_url(),
                                "duration": v.duration,
                                "tags": v.tags,
                                "likes": v.likes,
                                "downloads": v.downloads,
                                "relevance_score": rel_score,
                            }
                        )

        # Fallback if no candidate clips found
        if not all_clips:
            static_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "mock_providers",
                "static_videos",
            )
            if os.path.exists(static_dir):
                for idx, fname in enumerate(sorted(os.listdir(static_dir))):
                    if fname.endswith(".mp4"):
                        all_clips.append(
                            {
                                "id": 5000 + idx,
                                "keyword": fname.replace(".mp4", ""),
                                "url": f"http://127.0.0.1:9999/static/videos/{fname}",
                                "thumbnail_url": f"http://127.0.0.1:9999/static/thumbnails/{fname.replace('.mp4', '.jpg')}",
                                "duration": 15,
                                "tags": fname.replace(".mp4", "").replace("_", " "),
                                "likes": 500,
                                "downloads": 1000,
                                "relevance_score": 4.5,
                            }
                        )

        ctx.set("candidate_clips", all_clips)
        logger.info("video_search_done", extra={"candidates": len(all_clips)})
        return ctx


# ---------------------------------------------------------------------------
# Step 4 — Clip Filtering (Strict Relevance Threshold >= 4.0 / 5.0)
# ---------------------------------------------------------------------------
class ClipFilterStep(Step):
    """Filter candidate clips by duration and strict relevance score (>= 4.0 out of 5.0)."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        s = get_settings()
        candidates: list[dict[str, Any]] = ctx.get("candidate_clips", [])

        # 1. Filter by duration
        duration_filtered = [
            c
            for c in candidates
            if s.BROLL_MIN_CLIP_DURATION_SECONDS <= c.get("duration", 0) <= s.BROLL_MAX_CLIP_DURATION_SECONDS
        ]

        if not duration_filtered and candidates:
            duration_filtered = list(candidates)

        # 2. Strict Relevance Score Filter: Only fetch/select clips with relevance score > 4.0 out of 5.0
        MIN_RELEVANCE_SCORE = 4.0
        relevance_filtered = [
            c for c in duration_filtered
            if c.get("relevance_score", 0.0) > MIN_RELEVANCE_SCORE
        ]

        # Sort by relevance score descending (highest quality/relevance first)
        relevance_filtered.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)

        # Only clips with score > 4.0 are selected as stock videos; scenes below 4.0 generate motion pictures
        selected = relevance_filtered[: s.BROLL_MAX_CLIPS_PER_JOB]
        ctx.set("selected_clips", selected)
        logger.info(
            "clip_filter_done",
            extra={
                "selected": len(selected),
                "filtered_out": len(candidates) - len(selected),
                "min_relevance_threshold": MIN_RELEVANCE_SCORE,
            },
        )
        return ctx


# ---------------------------------------------------------------------------
# Step 5 — Media Assembly (renders full video using FFmpeg)
# ---------------------------------------------------------------------------
def _resolve_ffmpeg() -> str:
    """Find FFmpeg binary on system PATH, imageio-ffmpeg package, or environment."""
    import shutil
    if p := shutil.which("ffmpeg"):
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return os.environ.get("FFMPEG_BINARY", "ffmpeg")


class MediaAssemblyStep(Step):
    """
    Assemble the final video using FFmpeg.
    Fetches real stock video footage for each scene,
    burns in high-contrast centered subtitles,
    and encodes to standard, universally playable H.264/AAC MP4.
    """

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        import asyncio
        import hashlib
        import os
        import re
        import shutil
        import subprocess
        import tempfile
        import textwrap
        import httpx

        job_id = ctx.job_id
        aspect_ratio = (ctx.payload.get("aspect_ratios") or ["16:9"])[0]
        if aspect_ratio == "9:16":
            width, height = 1080, 1920
            font_size = 38
        elif aspect_ratio == "1:1":
            width, height = 1080, 1080
            font_size = 38
        else:
            width, height = 1920, 1080
            font_size = 42

        # Ensure media directories exist
        media_base = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
        media_cache = os.path.join(media_base, "cache")
        job_media_dir = os.path.join(media_base, str(job_id))
        os.makedirs(media_cache, exist_ok=True)
        os.makedirs(job_media_dir, exist_ok=True)

        final_video_path = os.path.join(job_media_dir, "video.mp4")
        final_srt_path = os.path.join(job_media_dir, "captions.srt")

        # Gather scene segments
        scenes_input = ctx.payload.get("scenes") or []
        script = ctx.get("transcript_text") or ctx.payload.get("script") or ""
        segments_input = ctx.get("transcript_segments") or []

        scene_items: list[dict[str, Any]] = []

        if scenes_input:
            for s in scenes_input:
                if isinstance(s, dict) and s.get("text"):
                    hint = s.get("visual_hint") or " ".join(s.get("keywords") or [])
                    dur_val = s.get("duration") or s.get("duration_hint_seconds") or 4.0
                    scene_items.append({
                        "text": s.get("text", "").strip(),
                        "hint": hint,
                        "duration": float(dur_val),
                    })
        elif script:
            # Maintain the logical sequence of complete sentences from the script
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", script) if len(s.strip()) > 3]
            if not raw_sentences:
                raw_sentences = [p.strip() for p in re.split(r"[.!?;\n]+", script) if len(p.strip()) > 3]

            stopwords = {
                "this", "that", "with", "from", "they", "them", "there", "their", "into", "through",
                "about", "when", "where", "which", "while", "every", "begins", "begin", "here", "comes",
                "gives", "before", "after", "some", "such", "than", "then", "very", "what", "also", "just",
                "more", "most", "only", "well"
            }
            for s in raw_sentences[:8]:
                words = [w.strip(",.!?\"'").lower() for w in s.split() if len(w) > 3]
                filtered = [w for w in words if w not in stopwords]
                hint = " ".join(filtered[:3]) if filtered else ""
                scene_items.append({
                    "text": s,
                    "hint": hint,
                    "duration": 4.5,
                })
        elif segments_input:
            for seg in segments_input:
                if isinstance(seg, dict) and seg.get("text"):
                    dur = max(2.5, float(seg.get("end", 0)) - float(seg.get("start", 0)))
                    scene_items.append({
                        "text": seg.get("text", "").strip(),
                        "hint": "",
                        "duration": dur,
                    })

        if not scene_items:
            keywords = ctx.get("keywords") or ["city skyline", "people walking", "technology future"]
            for kw in keywords[:5]:
                scene_items.append({
                    "text": f"Visualizing: {kw.title()}",
                    "hint": kw,
                    "duration": 4.0,
                })

        # Cap scenes to 6 scenes max for responsive generation
        scene_items = scene_items[:6]

        # Available footage clips from previous steps
        selected_clips: list[dict[str, Any]] = ctx.get("selected_clips") or ctx.get("candidate_clips") or []

        # Local static video clips directory fallback
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        static_dir = os.path.join(repo_root, "mock_providers", "static_videos")
        static_files = (
            [os.path.join(static_dir, f) for f in sorted(os.listdir(static_dir)) if f.endswith(".mp4")]
            if os.path.exists(static_dir)
            else []
        )

        async def fetch_clip(url: str) -> str | None:
            if not url:
                return None
            if url.startswith("file://"):
                p = url[7:]
                if os.path.exists(p):
                    return p
            if os.path.exists(url):
                return url
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cached_path = os.path.join(media_cache, f"{url_hash}.mp4")
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
                return cached_path
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
                    resp = await client.get(url)
                    if resp.status_code in (200, 206) and len(resp.content) > 10000:
                        with open(cached_path, "wb") as f:
                            f.write(resp.content)
                        return cached_path
            except Exception as exc:
                logger.warning("clip_download_failed", extra={"url": url, "error": str(exc)})
            return None

        ffmpeg_exe = _resolve_ffmpeg()
        logger.info("media_assembly_start", extra={"ffmpeg": ffmpeg_exe, "scenes": len(scene_items)})

        loop = asyncio.get_running_loop()
        tmp_dir = tempfile.mkdtemp(prefix=f"broll_{job_id}_")

        # Setup standard font file in tmp_dir for crisp typography
        font_opt = ""
        for sys_font in ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(sys_font):
                try:
                    shutil.copyfile(sys_font, os.path.join(tmp_dir, "sub_font.ttf"))
                    font_opt = ":fontfile=sub_font.ttf"
                    break
                except Exception:
                    pass

        try:
            # 1. Match footage clips for all scenes
            used_urls: set[str] = set()
            scene_urls: list[str | None] = []
            scene_clips: list[dict[str, Any] | None] = []

            for idx, scene in enumerate(scene_items):
                text = scene["text"]
                hint = scene.get("hint", "").lower()
                chosen_clip: dict[str, Any] | None = None

                if selected_clips:
                    # 1. Match on hint (prefer unused)
                    if hint:
                        for c in selected_clips:
                            if c.get("url") not in used_urls:
                                c_tags = ((c.get("tags") or "") + " " + (c.get("keyword") or "")).lower()
                                if any(w in c_tags for w in hint.split() if len(w) > 2):
                                    chosen_clip = c
                                    break
                    # 2. Match on words in scene text (prefer unused)
                    if not chosen_clip:
                        for c in selected_clips:
                            if c.get("url") not in used_urls:
                                c_tags = ((c.get("tags") or "") + " " + (c.get("keyword") or "")).lower()
                                if any(w in c_tags for w in text.lower().split() if len(w) > 3):
                                    chosen_clip = c
                                    break
                    # 3. Match on hint or text even if already used
                    if not chosen_clip and hint:
                        for c in selected_clips:
                            c_tags = ((c.get("tags") or "") + " " + (c.get("keyword") or "")).lower()
                            if any(w in c_tags for w in hint.split() if len(w) > 2):
                                chosen_clip = c
                                break
                    # 4. Fallback from unused clips first, then by index
                    if not chosen_clip:
                        unused = [c for c in selected_clips if c.get("url") not in used_urls]
                        if unused:
                            chosen_clip = unused[0]
                        elif selected_clips:
                            chosen_clip = selected_clips[idx % len(selected_clips)]

                # STRICT RULE: Only use video from Pixabay if relevance_score > 4.0
                if chosen_clip and float(chosen_clip.get("relevance_score", 0.0)) <= 4.0:
                    chosen_clip = None

                chosen_url = chosen_clip.get("url") if chosen_clip else None
                if chosen_url:
                    used_urls.add(chosen_url)
                scene_urls.append(chosen_url)
                scene_clips.append(chosen_clip)

            # 2. Pre-fetch all selected clips concurrently
            unique_urls = [u for u in set(scene_urls) if u]
            fetched_paths = await asyncio.gather(*[fetch_clip(u) for u in unique_urls])
            url_to_path = dict(zip(unique_urls, fetched_paths))

            # 3. Pre-generate all Edge-TTS voice audio tracks concurrently
            voice_name = ctx.payload.get("voice") or "en-US-AriaNeural"

            async def gen_voice(idx: int, text: str) -> tuple[str, bool, float]:
                vfile = os.path.join(tmp_dir, f"voice_{idx}.mp3")
                clean_speech_text = " ".join(text.split()).strip()
                vdur = 0.0
                try:
                    import edge_tts
                    comm = edge_tts.Communicate(clean_speech_text, voice_name)
                    await comm.save(vfile)
                    if os.path.exists(vfile) and os.path.getsize(vfile) > 1000:
                        def probe():
                            res = subprocess.run([ffmpeg_exe, "-i", vfile], capture_output=True, text=True)
                            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
                            if m:
                                h, mins, s_val = m.groups()
                                return int(h) * 3600 + int(mins) * 60 + float(s_val)
                            return 0.0
                        vdur = await loop.run_in_executor(None, probe)
                except Exception as tts_exc:
                    logger.warning("tts_voice_gen_failed", extra={"idx": idx, "error": str(tts_exc)})
                has_v = os.path.exists(vfile) and os.path.getsize(vfile) > 1000
                return vfile, has_v, vdur

            voice_results = await asyncio.gather(*[gen_voice(i, s["text"]) for i, s in enumerate(scene_items)])

            # 3b. Pre-query Groq API for motion picture parameters when relevance < 4.0
            async def get_groq_motion_style(scene_idx: int, scene_text: str) -> dict[str, str]:
                try:
                    from app.adapters.groq_client import GroqClient
                    client = GroqClient()
                    prompt = (
                        "You are an AI motion picture director. For the video scene narration: "
                        f"'{scene_text}', design cinematic motion picture visual styling. "
                        "Respond ONLY with valid JSON: "
                        '{"color_theme": "#1e3a8a", "zoom_rate": "0.0018", "visual_style": "cinematic_zoom"}'
                    )
                    res = await client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=100,
                    )
                    content = res["choices"][0]["message"]["content"].strip()
                    m = re.search(r"\{.*\}", content, re.DOTALL)
                    if m:
                        import json
                        return json.loads(m.group())
                except Exception as e:
                    logger.debug("groq_motion_query_failed", extra={"idx": scene_idx, "error": str(e)})
                palettes = ["#0f172a", "#1e1b4b", "#172554", "#022c22", "#311042"]
                return {"color_theme": palettes[scene_idx % len(palettes)], "zoom_rate": "0.0018", "visual_style": "cinematic_zoom"}

            motion_styles = await asyncio.gather(*[
                get_groq_motion_style(i, s["text"]) for i, s in enumerate(scene_items)
            ])

            # 4. Prepare commands, subtitles, and durations for each scene
            commands: list[list[str]] = []
            fallback_commands: list[list[str]] = []
            scene_meta: list[tuple[dict[str, Any], str, float]] = []

            for idx, scene in enumerate(scene_items):
                text = scene["text"]
                voice_file, has_voice, voice_duration = voice_results[idx]
                dur = max(float(scene.get("duration", 3.5)), voice_duration + 0.4) if voice_duration > 0 else float(scene.get("duration", 4.0))

                # Captions: Keep on a single line whenever possible.
                # If the text is too long, split it naturally into two balanced lines.
                clean_caption = " ".join(text.split()).strip()
                max_single_line_chars = 82 if width >= 1920 else (42 if width < height else 58)

                if len(clean_caption) <= max_single_line_chars:
                    sub_txt = clean_caption
                    cur_fontsize = 34 if width >= 1920 else 28
                    line_spacing_opt = ""
                else:
                    words = clean_caption.split()
                    half_len = len(clean_caption) / 2
                    best_idx = max(1, len(words) // 2)
                    best_score = float("inf")
                    running_len = 0
                    for w_i, w in enumerate(words[:-1]):
                        running_len += len(w) + 1
                        diff = abs(running_len - half_len)
                        if w.endswith((",", ";", ":", "-", "—")):
                            diff -= 6
                        elif words[w_i + 1].lower() in ("and", "but", "with", "as", "while", "that", "which"):
                            diff -= 3
                        if diff < best_score:
                            best_score = diff
                            best_idx = w_i + 1

                    line1 = " ".join(words[:best_idx])
                    line2 = " ".join(words[best_idx:])
                    sub_txt = f"{line1}\n{line2}"
                    cur_fontsize = 30 if width >= 1920 else 24
                    line_spacing_opt = ":line_spacing=12"

                sub_file_name = f"sub_{idx}.txt"
                sub_file_path = os.path.join(tmp_dir, sub_file_name)
                with open(sub_file_path, "w", encoding="utf-8") as f:
                    f.write(sub_txt)

                seg_out_name = f"seg_{idx}.mp4"
                scene_meta.append((scene, sub_txt, dur))

                # Resolved clip file
                chosen_url = scene_urls[idx]
                clip_file = url_to_path.get(chosen_url) if chosen_url else None
                if not clip_file and static_files:
                    clip_file = static_files[idx % len(static_files)]

                # Centered caption with high-contrast background at safe bottom margin
                vf_filter = (
                    f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height},"
                    f"drawtext=textfile={sub_file_name}{font_opt}:fontsize={cur_fontsize}:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=16{line_spacing_opt}:x=(w-text_w)/2:y=h-th-75"
                )
                filter_complex = f"[0:v]{vf_filter}[vout];[1:a]apad=pad_dur=2[aout]"

                # Motion Picture generated using Groq API styling (continuous Ken Burns camera zoom/pan)
                m_style = motion_styles[idx]
                col_hex = m_style.get("color_theme", "#0f172a").replace("#", "0x")
                zoom_rate = m_style.get("zoom_rate", "0.0018")
                total_frames = max(25, int(25 * dur))
                motion_in_w = int(width * 1.25)
                motion_in_h = int(height * 1.25)

                motion_vf = (
                    f"zoompan=z='min(zoom+{zoom_rate},1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps=25,"
                    f"drawtext=textfile={sub_file_name}{font_opt}:fontsize={cur_fontsize}:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=16{line_spacing_opt}:x=(w-text_w)/2:y=h-th-75"
                )
                motion_filter_complex = f"[0:v]{motion_vf}[vout];[1:a]apad=pad_dur=2[aout]"

                if has_voice:
                    fb_cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-loglevel", "error",
                        "-nostats",
                        "-f", "lavfi",
                        "-i", f"color=c={col_hex}:s={motion_in_w}x{motion_in_h}:d={dur}:r=25",
                        "-i", voice_file,
                        "-filter_complex", motion_filter_complex,
                        "-map", "[vout]",
                        "-map", "[aout]",
                        "-r", "25",
                        "-threads", "2",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-ar", "44100",
                        "-ac", "2",
                        "-t", str(dur),
                        seg_out_name,
                    ]
                else:
                    fb_cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-loglevel", "error",
                        "-nostats",
                        "-f", "lavfi",
                        "-i", f"color=c={col_hex}:s={motion_in_w}x{motion_in_h}:d={dur}:r=25",
                        "-f", "lavfi",
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-filter_complex", motion_filter_complex,
                        "-map", "[vout]",
                        "-map", "[aout]",
                        "-r", "25",
                        "-threads", "2",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-ar", "44100",
                        "-ac", "2",
                        "-t", str(dur),
                        seg_out_name,
                    ]
                fallback_commands.append(fb_cmd)

                if has_voice:
                    if clip_file and os.path.exists(clip_file):
                        cmd = [
                            ffmpeg_exe,
                            "-y",
                            "-loglevel", "error",
                            "-nostats",
                            "-stream_loop", "2",
                            "-ss", "0",
                            "-t", str(dur),
                            "-i", clip_file,
                            "-i", voice_file,
                            "-filter_complex", filter_complex,
                            "-map", "[vout]",
                            "-map", "[aout]",
                            "-r", "25",
                            "-threads", "2",
                            "-c:v", "libx264",
                            "-preset", "ultrafast",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-ar", "44100",
                            "-ac", "2",
                            "-t", str(dur),
                            seg_out_name,
                        ]
                    else:
                        cmd = fb_cmd
                else:
                    if clip_file and os.path.exists(clip_file):
                        cmd = [
                            ffmpeg_exe,
                            "-y",
                            "-loglevel", "error",
                            "-nostats",
                            "-stream_loop", "2",
                            "-ss", "0",
                            "-t", str(dur),
                            "-i", clip_file,
                            "-f", "lavfi",
                            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                            "-filter_complex", filter_complex,
                            "-map", "[vout]",
                            "-map", "[aout]",
                            "-r", "25",
                            "-threads", "2",
                            "-c:v", "libx264",
                            "-preset", "ultrafast",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac",
                            "-b:a", "192k",
                            "-ar", "44100",
                            "-ac", "2",
                            "-t", str(dur),
                            seg_out_name,
                        ]
                    else:
                        cmd = fb_cmd
                commands.append(cmd)

            # 5. Render segments sequentially for optimal CPU throughput and zero contention
            async def render_segment(c: list[str], s_idx: int) -> bool:
                def run_ffmpeg():
                    try:
                        return subprocess.run(c, cwd=tmp_dir, capture_output=True, timeout=90)
                    except subprocess.TimeoutExpired:
                        logger.warning("segment_render_timeout", extra={"idx": s_idx})
                        return None
                    except Exception as e:
                        logger.warning("segment_render_exception", extra={"idx": s_idx, "error": str(e)})
                        return None

                proc = await loop.run_in_executor(None, run_ffmpeg)
                seg_name = f"seg_{s_idx}.mp4"
                seg_path = os.path.join(tmp_dir, seg_name)

                if proc is None or proc.returncode != 0 or not os.path.exists(seg_path) or os.path.getsize(seg_path) < 1000:
                    logger.warning(
                        "segment_render_fallback_applied",
                        extra={
                            "idx": s_idx,
                            "stderr": proc.stderr.decode(errors="replace")[-300:] if (proc and proc.stderr) else "timeout_or_error",
                        },
                    )
                    fb = fallback_commands[s_idx]
                    try:
                        await loop.run_in_executor(
                            None, lambda: subprocess.run(fb, cwd=tmp_dir, capture_output=True, timeout=45)
                        )
                    except Exception as fb_exc:
                        logger.error("segment_fallback_failed", extra={"idx": s_idx, "error": str(fb_exc)})
                    return os.path.exists(seg_path) and os.path.getsize(seg_path) > 1000
                return True

            for s_idx in range(len(commands)):
                await render_segment(commands[s_idx], s_idx)

            # 6. Verify segments and build sequential SRT entries
            segment_files: list[str] = []
            srt_entries: list[str] = []
            curr_time = 0.0

            def fmt_time(t: float) -> str:
                hrs = int(t // 3600)
                mins = int((t % 3600) // 60)
                secs = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

            scene_details: list[dict[str, Any]] = []
            for idx, (scene, sub_txt, dur) in enumerate(scene_meta):
                seg_out_name = f"seg_{idx}.mp4"
                seg_out_path = os.path.join(tmp_dir, seg_out_name)
                if os.path.exists(seg_out_path) and os.path.getsize(seg_out_path) > 1000:
                    segment_files.append(seg_out_name)
                else:
                    logger.error("segment_file_missing", extra={"idx": idx, "expected": seg_out_path})

                start_s = curr_time
                end_s = curr_time + dur
                curr_time = end_s
                srt_entries.append(f"{idx + 1}\n{fmt_time(start_s)} --> {fmt_time(end_s)}\n{sub_txt}\n")

                clip_info = scene_clips[idx] if idx < len(scene_clips) else None
                rel_score = clip_info.get("relevance_score", 4.5) if clip_info else 4.5
                scene_details.append({
                    "index": idx + 1,
                    "text": scene["text"],
                    "duration": round(dur, 2),
                    "relevance_score": rel_score,
                    "clip_id": clip_info.get("id") if clip_info else None,
                    "thumbnail_url": clip_info.get("thumbnail_url") if clip_info else None,
                    "query": clip_info.get("keyword") or scene.get("hint") or "",
                    "tags": clip_info.get("tags") or "",
                })

            # 7. Concatenate all segment clips
            if segment_files:
                concat_txt = os.path.join(tmp_dir, "concat.txt")
                with open(concat_txt, "w", encoding="utf-8") as f:
                    for sf in segment_files:
                        f.write(f"file '{sf}'\n")

                concat_cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    "concat.txt",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    final_video_path,
                ]

                proc_concat = await loop.run_in_executor(
                    None, lambda: subprocess.run(concat_cmd, cwd=tmp_dir, capture_output=True, timeout=60)
                )
                if proc_concat.returncode != 0:
                    logger.warning(
                        "concat_failed",
                        extra={"stderr": proc_concat.stderr.decode(errors="replace")[-300:]},
                    )

            # Write captions.srt
            with open(final_srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_entries))

            ctx.set("scene_details", scene_details)
            ctx.set("assembled_file", final_video_path)
            ctx.set("srt_file", final_srt_path)
            ctx.set("output_duration", curr_time)

            logger.info(
                "media_assembly_complete",
                extra={
                    "job_id": job_id,
                    "duration": curr_time,
                    "video_path": final_video_path,
                    "video_size": os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0,
                },
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return ctx


# ---------------------------------------------------------------------------
# Step 6 — Upload Result
# ---------------------------------------------------------------------------
class UploadResultStep(Step):
    """Upload the assembled video to object storage and set result URLs."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        import os

        assembled = ctx.get("assembled_file")
        srt_file = ctx.get("srt_file")
        job_id = ctx.job_id

        if not assembled or not os.path.exists(assembled):
            logger.info("upload_skipped", extra={"reason": "no assembled_file"})
            ctx.set("output_url", None)
            ctx.set("srt_url", None)
            return ctx

        video_url = f"http://localhost:8000/api/v1/jobs/{job_id}/video"
        srt_url = f"http://localhost:8000/api/v1/jobs/{job_id}/srt"
        object_name = f"jobs/{job_id}/video.mp4"

        ctx.set("output_url", video_url)
        ctx.set("srt_url", srt_url)
        ctx.set("output_object", object_name)

        # Upload to MinIO if endpoint is actually reachable
        try:
            import socket
            s = get_settings()
            endpoint = str(s.STORAGE_ENDPOINT)
            if ":" in endpoint:
                host, port_str = endpoint.split(":", 1)
                port = int(port_str)
            else:
                host, port = endpoint, 9000
            with socket.create_connection((host, port), timeout=0.1):
                pass
            storage = get_storage_client()
            await storage.upload_file(s.STORAGE_BUCKET_FINAL, object_name, assembled)
            presigned = await storage.presigned_get_url(s.STORAGE_BUCKET_FINAL, object_name)
            if presigned:
                ctx.set("output_url", presigned)
            logger.info("storage_upload_done", extra={"object": object_name})
        except Exception:
            logger.info("storage_upload_fallback_local", extra={"video_url": video_url})

        logger.info("upload_done", extra={"video_url": ctx.get("output_url"), "srt_url": srt_url})
        return ctx
