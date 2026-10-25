"""
Shared data-types for the footage pipeline.

These are the canonical types used by PixabayFootageAdapter, ClipStore,
FootageRouter and any future footage source.  Nothing in this module
imports from other app modules — it is a pure data layer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FootageKind(str, enum.Enum):
    VIDEO = "video"
    IMAGE = "image"      # static image → will be animated with Ken Burns
    CARD  = "card"       # FFmpeg-generated title/data card (tier-4 fallback)


class FallbackTier(int, enum.Enum):
    PIXABAY_VIDEO  = 1
    KEN_BURNS      = 2
    CACHED_CLIP    = 3
    TITLE_CARD     = 4


# ---------------------------------------------------------------------------
# Rendition (a single quality level offered by the source)
# ---------------------------------------------------------------------------

@dataclass
class RenditionInfo:
    """One quality level for a video or image asset."""
    url: str
    width: int
    height: int
    size_bytes: int = 0        # 0 = unknown
    bitrate_kbps: int = 0      # 0 = unknown
    label: str = ""            # e.g. "large", "medium", "small"
    content_type: str = ""     # e.g. "video/mp4", "image/jpeg"

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def meets_minimum(self, min_width: int, min_height: int) -> bool:
        return self.width >= min_width and self.height >= min_height


# ---------------------------------------------------------------------------
# Candidate — normalised footage item returned by any source adapter
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    A normalised footage candidate, source-agnostic.

    All source-specific fields (Pixabay hit ID, etc.) are flattened here
    so the rest of the pipeline never needs to know which adapter produced it.
    """
    # Identity
    id: str                             # "<source>:<provider_id>" e.g. "pixabay:1234567"
    source: str                         # "pixabay" | "local_cache" | "ffmpeg_card"
    kind: FootageKind

    # Best rendition chosen for the target resolution
    url: str
    thumb_url: str
    width: int
    height: int
    duration: float                     # seconds; 0 for static images

    # Licence
    license: str = "Pixabay Content License"
    commercial_use: bool = True
    attribution: str = ""               # "Photo by <user> on Pixabay"
    page_url: str = ""

    # Discovery metadata
    tags: list[str] = field(default_factory=list)
    is_ai_generated: bool = False

    # All available renditions (filled by the adapter, used by best_rendition())
    renditions: list[RenditionInfo] = field(default_factory=list)

    # Set after the asset has been ingested into MinIO
    storage_object: str | None = None   # MinIO object key
    storage_bucket: str | None = None
    checksum_sha256: str | None = None
    file_size_bytes: int = 0

    # Internal
    fallback_tier: FallbackTier = FallbackTier.PIXABAY_VIDEO
    extra: dict[str, Any] = field(default_factory=dict)

    def best_rendition(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        fallback_width: int = 1280,
        fallback_height: int = 720,
    ) -> RenditionInfo | None:
        """
        Return the highest-quality rendition that meets *target* resolution.
        Falls back to the best that meets *fallback* resolution.
        Falls back to the single best available rendition.
        Returns ``None`` if there are no renditions at all.
        """
        if not self.renditions:
            return None

        # Sort highest-pixel-count first
        by_quality = sorted(self.renditions, key=lambda r: r.pixels, reverse=True)

        for r in by_quality:
            if r.meets_minimum(target_width, target_height):
                return r
        for r in by_quality:
            if r.meets_minimum(fallback_width, fallback_height):
                return r
        return by_quality[0]  # best we have

    def tag_set(self) -> frozenset[str]:
        """Return lower-cased tags as a frozenset for quick intersection."""
        return frozenset(t.strip().lower() for t in self.tags if t.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for Redis storage)."""
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind.value,
            "url": self.url,
            "thumb_url": self.thumb_url,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "license": self.license,
            "commercial_use": self.commercial_use,
            "attribution": self.attribution,
            "page_url": self.page_url,
            "tags": self.tags,
            "is_ai_generated": self.is_ai_generated,
            "renditions": [
                {
                    "url": r.url, "width": r.width, "height": r.height,
                    "size_bytes": r.size_bytes, "bitrate_kbps": r.bitrate_kbps,
                    "label": r.label, "content_type": r.content_type,
                }
                for r in self.renditions
            ],
            "storage_object": self.storage_object,
            "storage_bucket": self.storage_bucket,
            "checksum_sha256": self.checksum_sha256,
            "file_size_bytes": self.file_size_bytes,
            "fallback_tier": self.fallback_tier.value,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        renditions = [
            RenditionInfo(
                url=r["url"], width=r["width"], height=r["height"],
                size_bytes=r.get("size_bytes", 0),
                bitrate_kbps=r.get("bitrate_kbps", 0),
                label=r.get("label", ""),
                content_type=r.get("content_type", ""),
            )
            for r in d.get("renditions", [])
        ]
        return cls(
            id=d["id"],
            source=d["source"],
            kind=FootageKind(d["kind"]),
            url=d["url"],
            thumb_url=d.get("thumb_url", ""),
            width=d["width"],
            height=d["height"],
            duration=d.get("duration", 0.0),
            license=d.get("license", "Pixabay Content License"),
            commercial_use=d.get("commercial_use", True),
            attribution=d.get("attribution", ""),
            page_url=d.get("page_url", ""),
            tags=d.get("tags", []),
            is_ai_generated=d.get("is_ai_generated", False),
            renditions=renditions,
            storage_object=d.get("storage_object"),
            storage_bucket=d.get("storage_bucket"),
            checksum_sha256=d.get("checksum_sha256"),
            file_size_bytes=d.get("file_size_bytes", 0),
            fallback_tier=FallbackTier(d.get("fallback_tier", 1)),
            extra=d.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# DownloadedClip — a Candidate that has been ingested into MinIO
# ---------------------------------------------------------------------------

@dataclass
class DownloadedClip:
    """
    A Candidate whose media has been downloaded, validated, and stored in MinIO.
    Returned by ``download_to_minio()``.
    """
    candidate: Candidate
    object_key: str        # MinIO object key
    bucket: str
    checksum_sha256: str
    file_size_bytes: int
    validated_width: int
    validated_height: int
    validated_duration: float


# ---------------------------------------------------------------------------
# FallbackWarning — produced when the router falls back to a lower tier
# ---------------------------------------------------------------------------

@dataclass
class FallbackWarning:
    """Structured warning added to the job's warning list when a tier is skipped."""
    tier: FallbackTier
    reason: str
    keyword: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_str(self) -> str:
        return (
            f"[Tier {self.tier.value}] Fallback for '{self.keyword}': "
            f"{self.reason}"
        )
