"""
mock-providers: lightweight Flask app that fakes Groq and Pixabay REST APIs.

Endpoints exposed:

  Groq (OpenAI-compatible):
    POST /groq/v1/chat/completions
    POST /groq/v1/audio/transcriptions

  Pixabay:
    GET  /pixabay/          (image search)
    GET  /pixabay/videos/   (video search)

  Utility:
    GET  /health
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from flask import Flask, Response, jsonify, request

app = Flask(__name__)


import os
from flask import send_from_directory

STATIC_VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_videos")

AVAILABLE_CLIPS = [
    {
        "file": "city.mp4",
        "tags": "city, urban, skyline, street, traffic, car, architecture",
        "keywords": ["city", "urban", "skyline", "street", "traffic", "car", "building", "transport"],
    },
    {
        "file": "classroom.mp4",
        "tags": "classroom, school, study, presentation, learning, education, office, meeting",
        "keywords": ["classroom", "school", "study", "learn", "education", "student", "presentation", "meeting", "office"],
    },
    {
        "file": "people.mp4",
        "tags": "people walking, pedestrians, crowd, lifestyle, mall, society, community",
        "keywords": ["people", "walking", "pedestrian", "crowd", "lifestyle", "human", "friends", "community"],
    },
    {
        "file": "technology.mp4",
        "tags": "technology, ai, computer, camera, digital, futuristic, screen, facial recognition",
        "keywords": ["technology", "tech", "ai", "computer", "digital", "data", "software", "screen", "code", "future"],
    },
    {
        "file": "industry.mp4",
        "tags": "industry, worker, factory, construction, safety, engineering, manufacture",
        "keywords": ["industry", "worker", "factory", "construction", "work", "engineering", "machine", "business"],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_tags() -> str:
    tag_pool = [
        "city", "skyline", "people walking", "nature", "ocean", "forest",
        "technology", "office", "traffic", "mountains", "sunset", "coffee",
        "laptop", "teamwork", "business meeting", "construction", "rain",
    ]
    return ", ".join(random.sample(tag_pool, k=min(5, len(tag_pool))))


def _make_video_hit(i: int, query: str = "") -> dict[str, Any]:
    import zlib
    q_hash = (zlib.crc32(query.encode("utf-8")) % 800000) if query else 0
    vid_id = 100000 + q_hash + i
    q_lower = query.lower()

    matched = None
    if q_lower:
        for clip in AVAILABLE_CLIPS:
            if any(k in q_lower for k in clip["keywords"]):
                matched = clip
                break
    if not matched:
        matched = AVAILABLE_CLIPS[i % len(AVAILABLE_CLIPS)]

    video_filename = matched["file"]
    video_url = f"http://127.0.0.1:9999/static/videos/{video_filename}"
    tags = f"{query}, {matched['tags']}" if query else matched["tags"]

    return {
        "id": vid_id,
        "pageURL": f"https://pixabay.com/videos/mock-{vid_id}/",
        "type": "film",
        "tags": tags,
        "duration": 15,
        "picture_id": f"mock-{vid_id}",
        "user_id": 9900 + i,
        "user": f"mock_creator_{i}",
        "views": random.randint(1000, 50000),
        "downloads": random.randint(500, 10000),
        "likes": random.randint(100, 5000),
        "comments": random.randint(5, 200),
        "videos": {
            "large": {
                "url": video_url,
                "width": 1920,
                "height": 1080,
                "size": 10000000,
                "thumbnail": f"http://127.0.0.1:9999/static/videos/{video_filename}.jpg",
            },
            "medium": {
                "url": video_url,
                "width": 1280,
                "height": 720,
                "size": 6000000,
                "thumbnail": f"http://127.0.0.1:9999/static/videos/{video_filename}.jpg",
            },
            "small": {
                "url": video_url,
                "width": 960,
                "height": 540,
                "size": 3000000,
                "thumbnail": f"http://127.0.0.1:9999/static/videos/{video_filename}.jpg",
            },
            "tiny": {
                "url": video_url,
                "width": 640,
                "height": 360,
                "size": 1500000,
                "thumbnail": f"http://127.0.0.1:9999/static/videos/{video_filename}.jpg",
            },
        },
    }


def _make_image_hit(i: int) -> dict[str, Any]:
    img_id = 2000 + i
    return {
        "id": img_id,
        "pageURL": f"https://pixabay.com/photos/mock-{img_id}/",
        "type": "photo",
        "tags": _random_tags(),
        "previewURL": f"https://cdn.pixabay.com/photo/mock-preview-{img_id}.jpg",
        "webformatURL": f"https://cdn.pixabay.com/photo/mock-webformat-{img_id}.jpg",
        "imageURL": f"https://cdn.pixabay.com/photo/mock-full-{img_id}.jpg",
        "imageWidth": 1920,
        "imageHeight": 1280,
        "imageSize": random.randint(500000, 5000000),
        "views": random.randint(100, 50000),
        "downloads": random.randint(50, 10000),
        "likes": random.randint(10, 5000),
        "user": f"mock_user_{i}",
        "user_id": 8800 + i,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok", "service": "mock-providers"})


# ---------------------------------------------------------------------------
# Groq — chat completions
# ---------------------------------------------------------------------------
@app.post("/groq/v1/chat/completions")
def groq_chat_completions() -> Response:
    body: dict[str, Any] = request.get_json(silent=True) or {}
    model = body.get("model", "llama3-8b-8192")

    # Return a fake keyword extraction JSON array
    keywords = [
        "city skyline", "people walking", "office meeting",
        "nature landscape", "technology screen",
    ]
    content = json.dumps(keywords)

    return jsonify({
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        },
    })


# ---------------------------------------------------------------------------
# Groq — audio transcriptions (Whisper)
# ---------------------------------------------------------------------------
@app.post("/groq/v1/audio/transcriptions")
def groq_transcriptions() -> Response:
    model = request.form.get("model", "whisper-large-v3")
    response_format = request.form.get("response_format", "verbose_json")

    mock_text = (
        "This is a mock transcription of the uploaded audio file. "
        "The speaker discusses technology trends, urban life, and the future of work."
    )

    if response_format == "verbose_json":
        return jsonify({
            "task": "transcribe",
            "language": "english",
            "duration": 12.5,
            "text": mock_text,
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "This is a mock transcription of the uploaded audio file.",
                    "tokens": [1, 2, 3],
                    "temperature": 0.0,
                    "avg_logprob": -0.2,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                },
                {
                    "id": 1,
                    "seek": 400,
                    "start": 4.0,
                    "end": 12.5,
                    "text": "The speaker discusses technology trends, urban life, and the future of work.",
                    "tokens": [4, 5, 6],
                    "temperature": 0.0,
                    "avg_logprob": -0.18,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.02,
                },
            ],
        })

    return jsonify({"text": mock_text})


@app.get("/static/videos/<path:filename>")
def serve_static_video(filename: str) -> Response:
    return send_from_directory(STATIC_VIDEOS_DIR, filename)


# ---------------------------------------------------------------------------
# Pixabay — video search
# ---------------------------------------------------------------------------
@app.get("/pixabay/videos/")
def pixabay_videos() -> Response:
    query = request.args.get("q", "")
    per_page = int(request.args.get("per_page", 10))
    hits = [_make_video_hit(i, query=query) for i in range(min(per_page, 20))]
    return jsonify({
        "total": 500,
        "totalHits": len(hits) * 5,
        "hits": hits,
    })


# ---------------------------------------------------------------------------
# Pixabay — image search
# ---------------------------------------------------------------------------
@app.get("/pixabay/")
def pixabay_images() -> Response:
    per_page = int(request.args.get("per_page", 10))
    hits = [_make_image_hit(i) for i in range(min(per_page, 20))]
    return jsonify({
        "total": 1000,
        "totalHits": 200,
        "hits": hits,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=False)
