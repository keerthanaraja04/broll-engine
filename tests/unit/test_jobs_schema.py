"""
Unit tests for job request/response schemas and input validation.

Covers:
- Empty script rejection
- Wrong file type
- Bad aspect ratio
- Audio duration guard
- Valid payloads parse correctly
- canonical_hash() determinism
- validate_audio_filename()
- validate_aspect_ratios()
"""

from __future__ import annotations

import os
import pytest

# Bootstrap env before any app import
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# JobCreateRequest — valid cases
# ---------------------------------------------------------------------------
class TestJobCreateRequestValid:
    def test_audio_url_only(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req = JobCreateRequest(audio_url="https://example.com/audio.mp3")
        assert str(req.audio_url) == "https://example.com/audio.mp3"
        assert req.aspect_ratios == ["16:9"]
        assert req.captions.burn_in is False
        assert req.captions.srt is False

    def test_script_source(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req = JobCreateRequest(
            script="This is a valid script",
            aspect_ratios=["16:9", "9:16"],
            voice="en-US-standard",
        )
        assert req.script == "This is a valid script"
        assert req.voice == "en-US-standard"
        assert "9:16" in req.aspect_ratios

    def test_scenes_source(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest, SceneInput

        req = JobCreateRequest(
            scenes=[SceneInput(text="Scene one", keywords=["city"])],
            style="cinematic",
            avoid=["violence", "weapons"],
        )
        assert req.scenes is not None
        assert len(req.scenes) == 1
        assert req.avoid == ["violence", "weapons"]

    def test_all_valid_aspect_ratios(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req = JobCreateRequest(
            audio_url="https://example.com/a.mp3",
            aspect_ratios=["16:9", "9:16", "1:1"],
        )
        assert len(req.aspect_ratios) == 3

    def test_callback_url_accepted(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req = JobCreateRequest(
            script="Hello world",
            callback_url="https://hooks.example.com/broll",
        )
        assert "hooks.example.com" in str(req.callback_url)

    def test_captions_options(self) -> None:
        from app.api.schemas.jobs import CaptionOptions, JobCreateRequest

        req = JobCreateRequest(
            script="Captions test",
            captions=CaptionOptions(burn_in=True, srt=True),
        )
        assert req.captions.burn_in is True
        assert req.captions.srt is True


# ---------------------------------------------------------------------------
# JobCreateRequest — validation failures
# ---------------------------------------------------------------------------
class TestJobCreateRequestInvalid:
    def test_no_source_raises(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises(ValidationError) as exc_info:
            JobCreateRequest()
        assert "exactly one" in str(exc_info.value).lower() or "provide" in str(exc_info.value).lower()

    def test_multiple_sources_raises(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises(ValidationError):
            JobCreateRequest(
                audio_url="https://example.com/a.mp3",
                script="Also a script",
            )

    def test_empty_script_raises(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises((ValidationError, Exception)):
            JobCreateRequest(script="   ")

    def test_blank_script_whitespace_only(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises((ValidationError, Exception)):
            JobCreateRequest(script="\n\t  \n")

    def test_empty_scenes_list_raises(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises(ValidationError):
            # min_length=1 on scenes field
            JobCreateRequest(scenes=[])

    def test_invalid_aspect_ratio_in_literal(self) -> None:
        from pydantic import ValidationError
        from app.api.schemas.jobs import JobCreateRequest

        with pytest.raises(ValidationError):
            # "4:3" is not in the Literal set
            JobCreateRequest(
                audio_url="https://example.com/a.mp3",
                aspect_ratios=["4:3"],  # type: ignore[list-item]
            )


# ---------------------------------------------------------------------------
# canonical_hash()
# ---------------------------------------------------------------------------
class TestCanonicalHash:
    def test_same_body_same_hash(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req1 = JobCreateRequest(script="Hello world")
        req2 = JobCreateRequest(script="Hello world")
        assert req1.canonical_hash() == req2.canonical_hash()

    def test_different_body_different_hash(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        req1 = JobCreateRequest(script="Hello world")
        req2 = JobCreateRequest(script="Different text")
        assert req1.canonical_hash() != req2.canonical_hash()

    def test_hash_is_hex_string(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        h = JobCreateRequest(script="test").canonical_hash()
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)

    def test_extra_fields_change_hash(self) -> None:
        from app.api.schemas.jobs import JobCreateRequest

        r1 = JobCreateRequest(script="test")
        r2 = JobCreateRequest(script="test", voice="en-US")
        assert r1.canonical_hash() != r2.canonical_hash()


# ---------------------------------------------------------------------------
# validate_audio_filename()
# ---------------------------------------------------------------------------
class TestValidateAudioFilename:
    def test_accepted_extensions(self) -> None:
        from app.api.schemas.jobs import validate_audio_filename

        for ext in ("mp3", "mp4", "wav", "m4a", "ogg", "webm", "flac"):
            result = validate_audio_filename(f"audio.{ext}")
            assert result == ext

    def test_case_insensitive(self) -> None:
        from app.api.schemas.jobs import validate_audio_filename

        assert validate_audio_filename("AUDIO.MP3") == "mp3"

    def test_rejected_extension_raises(self) -> None:
        from app.core.errors import WrongFileTypeError
        from app.api.schemas.jobs import validate_audio_filename

        with pytest.raises(WrongFileTypeError) as exc_info:
            validate_audio_filename("document.pdf")
        assert exc_info.value.code == "WRONG_FILE_TYPE"

    def test_exe_raises(self) -> None:
        from app.core.errors import WrongFileTypeError
        from app.api.schemas.jobs import validate_audio_filename

        with pytest.raises(WrongFileTypeError):
            validate_audio_filename("malware.exe")

    def test_no_extension_raises(self) -> None:
        from app.core.errors import WrongFileTypeError
        from app.api.schemas.jobs import validate_audio_filename

        with pytest.raises(WrongFileTypeError):
            validate_audio_filename("audiofile")

    def test_wrong_file_type_error_has_details(self) -> None:
        from app.core.errors import WrongFileTypeError
        from app.api.schemas.jobs import validate_audio_filename

        with pytest.raises(WrongFileTypeError) as exc_info:
            validate_audio_filename("clip.avi")
        assert exc_info.value.details is not None
        assert "extension" in exc_info.value.details


# ---------------------------------------------------------------------------
# validate_aspect_ratios()
# ---------------------------------------------------------------------------
class TestValidateAspectRatios:
    def test_valid_ratios_no_error(self) -> None:
        from app.api.schemas.jobs import validate_aspect_ratios

        validate_aspect_ratios(["16:9"])
        validate_aspect_ratios(["9:16"])
        validate_aspect_ratios(["1:1"])
        validate_aspect_ratios(["16:9", "9:16", "1:1"])

    def test_invalid_ratio_raises(self) -> None:
        from app.core.errors import BadAspectRatioError
        from app.api.schemas.jobs import validate_aspect_ratios

        with pytest.raises(BadAspectRatioError) as exc_info:
            validate_aspect_ratios(["4:3"])
        assert exc_info.value.code == "BAD_ASPECT_RATIO"

    def test_mixed_valid_invalid_raises(self) -> None:
        from app.core.errors import BadAspectRatioError
        from app.api.schemas.jobs import validate_aspect_ratios

        with pytest.raises(BadAspectRatioError) as exc_info:
            validate_aspect_ratios(["16:9", "21:9"])
        assert "21:9" in str(exc_info.value.details)

    def test_error_lists_bad_values_in_details(self) -> None:
        from app.core.errors import BadAspectRatioError
        from app.api.schemas.jobs import validate_aspect_ratios

        with pytest.raises(BadAspectRatioError) as exc_info:
            validate_aspect_ratios(["4:5", "3:2"])
        details = exc_info.value.details or {}
        assert "invalid" in details
        assert set(details["invalid"]) == {"4:5", "3:2"}


# ---------------------------------------------------------------------------
# Error class attributes
# ---------------------------------------------------------------------------
class TestJobErrorCodes:
    def test_empty_script_error_code(self) -> None:
        from app.core.errors import EmptyScriptError

        err = EmptyScriptError("blank script")
        assert err.code == "EMPTY_SCRIPT"
        assert err.http_status == 422
        assert err.retryable is False

    def test_wrong_file_type_error_code(self) -> None:
        from app.core.errors import WrongFileTypeError

        err = WrongFileTypeError("bad file")
        assert err.code == "WRONG_FILE_TYPE"
        assert err.http_status == 415

    def test_bad_aspect_ratio_error_code(self) -> None:
        from app.core.errors import BadAspectRatioError

        err = BadAspectRatioError("4:3 not allowed")
        assert err.code == "BAD_ASPECT_RATIO"
        assert err.http_status == 422

    def test_audio_too_long_error_code(self) -> None:
        from app.core.errors import AudioTooLongError

        err = AudioTooLongError("audio > 10 min")
        assert err.code == "AUDIO_TOO_LONG"
        assert err.http_status == 422
        assert err.retryable is False

    def test_idempotency_conflict_error_code(self) -> None:
        from app.core.errors import IdempotencyKeyConflictError

        err = IdempotencyKeyConflictError("key conflict")
        assert err.code == "IDEMPOTENCY_KEY_CONFLICT"
        assert err.http_status == 409
        assert err.retryable is False
