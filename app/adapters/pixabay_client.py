"""
Pixabay API adapter.

Provides:
  - search_videos()  — search for stock video clips by keyword
  - search_images()  — search for stock images by keyword

Both methods handle pagination, minimum resolution filtering, and retries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.errors import PixabayServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class PixabayVideo:
    id: int
    page_url: str
    tags: str
    duration: int
    views: int
    downloads: int
    likes: int
    # video URLs keyed by quality label (e.g. "large", "medium", "small", "tiny")
    videos: dict[str, dict[str, Any]] = field(default_factory=dict)
    user: str = ""
    user_id: int = 0
    picture_id: str = ""

    @classmethod
    def from_api(cls, hit: dict[str, Any]) -> "PixabayVideo":
        return cls(
            id=hit["id"],
            page_url=hit.get("pageURL", ""),
            tags=hit.get("tags", ""),
            duration=hit.get("duration", 0),
            views=hit.get("views", 0),
            downloads=hit.get("downloads", 0),
            likes=hit.get("likes", 0),
            videos=hit.get("videos", {}),
            user=hit.get("user", ""),
            user_id=hit.get("user_id", 0),
            picture_id=hit.get("picture_id", ""),
        )

    def best_url(self, min_width: int = 1280) -> str | None:
        """Return the URL of the best quality variant that meets min_width."""
        priority = ["small", "medium", "large", "tiny"]
        for quality in priority:
            v = self.videos.get(quality, {})
            if v.get("width", 0) >= min_width:
                return v.get("url")
        # Fallback: any URL
        for quality in priority:
            v = self.videos.get(quality, {})
            if url := v.get("url"):
                return url
        return None

    def thumbnail_url(self) -> str | None:
        """Return the best thumbnail image URL from video variants or picture_id."""
        for quality in ["small", "medium", "large", "tiny"]:
            v = self.videos.get(quality, {})
            if thumb := v.get("thumbnail"):
                return thumb
        if self.picture_id:
            return f"https://i.vimeocdn.com/video/{self.picture_id}_295x166.jpg"
        return None


@dataclass
class PixabayImage:
    id: int
    page_url: str
    tags: str
    image_url: str
    preview_url: str
    web_format_url: str
    width: int
    height: int
    views: int
    downloads: int
    likes: int
    user: str = ""

    @classmethod
    def from_api(cls, hit: dict[str, Any]) -> "PixabayImage":
        return cls(
            id=hit["id"],
            page_url=hit.get("pageURL", ""),
            tags=hit.get("tags", ""),
            image_url=hit.get("imageURL", ""),
            preview_url=hit.get("previewURL", ""),
            web_format_url=hit.get("webformatURL", ""),
            width=hit.get("imageWidth", 0),
            height=hit.get("imageHeight", 0),
            views=hit.get("views", 0),
            downloads=hit.get("downloads", 0),
            likes=hit.get("likes", 0),
            user=hit.get("user", ""),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PixabayClient:
    """Async client for the Pixabay REST API."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._video_base = str(self._settings.PIXABAY_VIDEO_BASE_URL).rstrip("/")
        self._image_base = str(self._settings.PIXABAY_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(10.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search_videos(
        self,
        query: str,
        per_page: int | None = None,
        page: int = 1,
        video_type: str | None = None,
        min_width: int | None = None,
        min_height: int | None = None,
    ) -> list[PixabayVideo]:
        """
        Search Pixabay for videos matching *query*.

        Args:
            query:      Search keyword(s).
            per_page:   Results per page (1–200). Defaults to configured value.
            page:       Page number (1-based).
            video_type: "film" | "animation" | "all". Defaults to configured value.
            min_width:  Minimum video width. Defaults to configured value.
            min_height: Minimum video height. Defaults to configured value.

        Returns:
            List of PixabayVideo dataclass instances.
        """
        s = self._settings
        params: dict[str, Any] = {
            "key": s.PIXABAY_API_KEY,
            "q": query,
            "per_page": per_page or s.PIXABAY_RESULTS_PER_PAGE,
            "page": page,
            "video_type": video_type or s.PIXABAY_VIDEO_TYPE,
            "min_width": min_width or s.PIXABAY_MIN_WIDTH,
            "min_height": min_height or s.PIXABAY_MIN_HEIGHT,
        }

        data = await self._get(self._video_base + "/", params)
        hits = data.get("hits", [])
        logger.info(
            "pixabay_video_search",
            extra={"query": query, "total_hits": data.get("totalHits"), "returned": len(hits)},
        )
        return [PixabayVideo.from_api(h) for h in hits]

    async def search_images(
        self,
        query: str,
        per_page: int | None = None,
        page: int = 1,
        image_type: str = "photo",
    ) -> list[PixabayImage]:
        s = self._settings
        params: dict[str, Any] = {
            "key": s.PIXABAY_API_KEY,
            "q": query,
            "per_page": per_page or s.PIXABAY_RESULTS_PER_PAGE,
            "page": page,
            "image_type": image_type,
            "min_width": s.PIXABAY_MIN_WIDTH,
            "min_height": s.PIXABAY_MIN_HEIGHT,
        }

        data = await self._get(self._image_base + "/", params)
        hits = data.get("hits", [])
        return [PixabayImage.from_api(h) for h in hits]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True,
    )
    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, params=params)
            except httpx.TimeoutException as exc:
                raise PixabayServiceError(f"Pixabay request timed out: {exc}") from exc
            except httpx.ConnectError as exc:
                raise PixabayServiceError(f"Could not connect to Pixabay: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Pixabay rate limit exceeded")
        if not response.is_success:
            raise PixabayServiceError(
                f"Pixabay returned HTTP {response.status_code}",
                details={"body": response.text[:500]},
            )
        return response.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_client: PixabayClient | None = None


def get_pixabay_client() -> PixabayClient:
    global _client
    if _client is None:
        _client = PixabayClient()
    return _client
