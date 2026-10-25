"""
Scene segment cache — stores normalised MP4 segments on disk.

Purpose:
  - Normalising a clip (trim/scale/reframe/brightness) is expensive.
  - If the user wants to swap a clip for a different scene, we can reuse
    the already-normalised segments for all other scenes.
  - Cache keys are deterministic: ``<job_id>:<scene_id>:<clip_id>:<aspect>``

Cache layout:
  <RENDER_SCENE_CACHE_DIR or system tmp>/broll_cache/<job>/<scene>_<clip>_<aspect>.mp4
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _cache_root() -> str:
    s = get_settings()
    base = s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    root = os.path.join(base, "broll_cache")
    os.makedirs(root, exist_ok=True)
    return root


def cache_key(job_id: str, scene_id: int, clip_id: str, aspect: str) -> str:
    """Build a deterministic cache key for a normalised segment."""
    raw = f"{job_id}:{scene_id}:{clip_id}:{aspect}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return os.path.join(_cache_root(), f"{job_id}_{scene_id}_{h}.mp4")


def get_cached(job_id: str, scene_id: int, clip_id: str, aspect: str) -> str | None:
    """
    Return the path to a cached normalised segment, or None if not cached.

    Checks file existence and non-zero size.
    """
    if not get_settings().RENDER_SCENE_CACHE_ENABLED:
        return None
    path = cache_key(job_id, scene_id, clip_id, aspect)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        logger.debug("scene_cache_hit", extra={"path": path})
        return path
    return None


def put_cached(path: str, job_id: str, scene_id: int, clip_id: str, aspect: str) -> str:
    """
    Register *path* as the cached normalised segment.

    If the file at *path* is not already at the cache location, it is
    copied (or renamed) there.

    Returns the cache path.
    """
    if not get_settings().RENDER_SCENE_CACHE_ENABLED:
        return path  # cache disabled — return original

    cache_path = cache_key(job_id, scene_id, clip_id, aspect)
    if os.path.abspath(path) == os.path.abspath(cache_path):
        return path  # already at cache location

    try:
        # Copy to cache location
        import shutil
        shutil.copy2(path, cache_path)
        logger.debug("scene_cache_stored", extra={"path": cache_path})
        return cache_path
    except Exception as exc:
        logger.warning("scene_cache_store_failed", extra={"error": str(exc)})
        return path


def invalidate(job_id: str, scene_id: int, clip_id: str, aspect: str) -> None:
    """Remove a cached segment."""
    path = cache_key(job_id, scene_id, clip_id, aspect)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def invalidate_job(job_id: str) -> None:
    """Remove all cached segments for a job."""
    root = _cache_root()
    for fname in os.listdir(root):
        if fname.startswith(f"{job_id}_"):
            try:
                os.unlink(os.path.join(root, fname))
            except OSError:
                pass
