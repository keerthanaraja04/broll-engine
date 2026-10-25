"""Test fixtures package — provides paths to CC0 test files."""

from __future__ import annotations

import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_WAV = os.path.join(FIXTURE_DIR, "sample_audio.wav")
CLIP_CITY = os.path.join(FIXTURE_DIR, "clip_city.jpg")
CLIP_NATURE = os.path.join(FIXTURE_DIR, "clip_nature.jpg")
CLIP_ABSTRACT = os.path.join(FIXTURE_DIR, "clip_abstract.jpg")


def audio_bytes() -> bytes:
    with open(AUDIO_WAV, "rb") as f:
        return f.read()


def image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
