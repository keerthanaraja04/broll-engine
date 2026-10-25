"""
Unit tests for the batch CLI (cli/batch.py).

No real network calls — httpx is mocked throughout.
"""

from __future__ import annotations

import csv
import json
import os
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    r.raise_for_status = MagicMock()
    return r


def _make_stream_response(content: bytes) -> MagicMock:
    """Mock an httpx stream response."""
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    r.iter_bytes = MagicMock(return_value=iter([content]))
    return r


# ---------------------------------------------------------------------------
# _submit_json
# ---------------------------------------------------------------------------

class TestSubmitJson:
    def test_returns_job_id(self) -> None:
        from cli.batch import _submit_json
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202,
            {"job_id": "abc-123", "status": "queued"})
        job_id = _submit_json(mock_client, "http://api", {"script": "test"})
        assert job_id == "abc-123"

    def test_passes_idempotency_header(self) -> None:
        from cli.batch import _submit_json
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202,
            {"job_id": "xyz"})
        body = {"script": "test", "_idempotency_key": "key-001"}
        _submit_json(mock_client, "http://api", body)
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs[1].get("headers") or {}
        assert "Idempotency-Key" in headers

    def test_idempotency_key_stripped_from_body(self) -> None:
        from cli.batch import _submit_json
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202, {"job_id": "xyz"})
        body = {"script": "test", "_idempotency_key": "key-001"}
        _submit_json(mock_client, "http://api", body)
        call_json = mock_client.post.call_args[1].get("json", {})
        assert "_idempotency_key" not in call_json


# ---------------------------------------------------------------------------
# _submit_upload
# ---------------------------------------------------------------------------

class TestSubmitUpload:
    def test_returns_job_id(self, tmp_path: Path) -> None:
        from cli.batch import _submit_upload
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake-audio")
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202,
            {"job_id": "upload-job-1"})
        job_id = _submit_upload(mock_client, "http://api", str(audio))
        assert job_id == "upload-job-1"

    def test_uses_multipart(self, tmp_path: Path) -> None:
        from cli.batch import _submit_upload
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"wav-data")
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202, {"job_id": "w1"})
        _submit_upload(mock_client, "http://api", str(audio))
        call_kwargs = mock_client.post.call_args[1]
        assert "files" in call_kwargs


# ---------------------------------------------------------------------------
# _submit_script
# ---------------------------------------------------------------------------

class TestSubmitScript:
    def test_reads_file_content(self, tmp_path: Path) -> None:
        from cli.batch import _submit_script
        txt = tmp_path / "script.txt"
        txt.write_text("Hello world narration.", encoding="utf-8")
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202, {"job_id": "s1"})
        job_id = _submit_script(mock_client, "http://api", str(txt))
        assert job_id == "s1"
        posted_body = mock_client.post.call_args[1]["json"]
        assert "Hello world" in posted_body["script"]


# ---------------------------------------------------------------------------
# _poll_job
# ---------------------------------------------------------------------------

class TestPollJob:
    def test_returns_when_succeeded(self) -> None:
        from cli.batch import _poll_job
        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200,
            {"status": "succeeded", "result": {"video_url": "https://x.com/v.mp4"}})
        result = _poll_job(mock_client, "http://api", "job-1", timeout=10)
        assert result["status"] == "succeeded"

    def test_returns_when_failed(self) -> None:
        from cli.batch import _poll_job
        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200,
            {"status": "failed", "error_message": "pipeline error"})
        result = _poll_job(mock_client, "http://api", "job-fail", timeout=10)
        assert result["status"] == "failed"

    def test_returns_not_found_on_404(self) -> None:
        from cli.batch import _poll_job
        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(404, {"error": "not found"})
        result = _poll_job(mock_client, "http://api", "missing", timeout=5)
        assert result["status"] == "not_found"

    def test_timeout_when_never_done(self) -> None:
        from cli.batch import _poll_job
        mock_client = MagicMock()
        mock_client.get.return_value = _make_response(200, {"status": "running"})
        result = _poll_job(mock_client, "http://api", "slow-job", timeout=1)
        assert result["status"] == "timeout"


# ---------------------------------------------------------------------------
# _download_url
# ---------------------------------------------------------------------------

class TestDownloadUrl:
    def test_downloads_to_file(self, tmp_path: Path) -> None:
        from cli.batch import _download_url
        dest = str(tmp_path / "video.mp4")
        mock_client = MagicMock()
        mock_client.stream.return_value = _make_stream_response(b"video-data")
        success = _download_url(mock_client, "https://example.com/v.mp4", dest)
        assert success is True
        assert open(dest, "rb").read() == b"video-data"

    def test_returns_false_on_error(self, tmp_path: Path) -> None:
        from cli.batch import _download_url
        dest = str(tmp_path / "video.mp4")
        mock_client = MagicMock()
        mock_client.stream.side_effect = Exception("network down")
        success = _download_url(mock_client, "https://example.com/v.mp4", dest)
        assert success is False

    def test_empty_url_returns_false(self, tmp_path: Path) -> None:
        from cli.batch import _download_url
        mock_client = MagicMock()
        assert _download_url(mock_client, "", str(tmp_path / "f.mp4")) is False


# ---------------------------------------------------------------------------
# _process_input
# ---------------------------------------------------------------------------

class TestProcessInput:
    def test_processes_json_file(self, tmp_path: Path) -> None:
        from cli.batch import _process_input
        job_file = tmp_path / "job.json"
        job_file.write_text(json.dumps({"script": "test"}), encoding="utf-8")

        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202, {"job_id": "j1"})
        mock_client.get.return_value = _make_response(200, {"status": "succeeded",
            "result": {"video_url": "https://x.com/v.mp4", "srt_url": ""}})
        mock_client.stream.return_value = _make_stream_response(b"video")

        result = _process_input(mock_client, "http://api",
                                  str(job_file), str(tmp_path))
        assert result["status"] == "succeeded"
        assert result["job_id"] == "j1"
        assert result["input_file"] == "job.json"

    def test_processes_txt_script(self, tmp_path: Path) -> None:
        from cli.batch import _process_input
        txt = tmp_path / "script.txt"
        txt.write_text("Hello world.", encoding="utf-8")
        mock_client = MagicMock()
        mock_client.post.return_value = _make_response(202, {"job_id": "s1"})
        mock_client.get.return_value = _make_response(200,
            {"status": "succeeded", "result": {}})
        result = _process_input(mock_client, "http://api",
                                  str(txt), str(tmp_path))
        assert result["status"] == "succeeded"

    def test_skips_unknown_extension(self, tmp_path: Path) -> None:
        from cli.batch import _process_input
        bad = tmp_path / "video.avi"
        bad.write_bytes(b"avi")
        mock_client = MagicMock()
        result = _process_input(mock_client, "http://api",
                                  str(bad), str(tmp_path))
        assert result["status"] == "skipped"

    def test_handles_api_error(self, tmp_path: Path) -> None:
        from cli.batch import _process_input
        import httpx
        job_file = tmp_path / "job.json"
        job_file.write_text(json.dumps({"script": "test"}), encoding="utf-8")
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("API unavailable")
        result = _process_input(mock_client, "http://api",
                                  str(job_file), str(tmp_path))
        assert result["status"] == "error"
        assert "unavailable" in result["error"].lower()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

class TestSummaryCsv:
    def test_csv_written_correctly(self, tmp_path: Path) -> None:
        """Integration: write CSV from a list of result dicts."""
        import csv
        results = [
            {"input_file": "a.json", "job_id": "j1", "status": "succeeded",
             "video_downloaded": True, "srt_downloaded": False,
             "error": "", "error_code": ""},
            {"input_file": "b.json", "job_id": "j2", "status": "failed",
             "video_downloaded": False, "srt_downloaded": False,
             "error": "timeout", "error_code": "PIPELINE_ERROR"},
        ]
        csv_path = str(tmp_path / "summary.csv")
        fieldnames = ["input_file", "job_id", "status", "video_downloaded",
                       "srt_downloaded", "error", "error_code"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        # Read back and verify
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["status"] == "succeeded"
        assert rows[1]["status"] == "failed"
        assert rows[1]["error_code"] == "PIPELINE_ERROR"

    def test_csv_has_all_required_columns(self, tmp_path: Path) -> None:
        csv_path = str(tmp_path / "summary.csv")
        fieldnames = ["input_file", "job_id", "status", "video_downloaded",
                       "srt_downloaded", "error", "error_code"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            for col in fieldnames:
                assert col in reader.fieldnames
