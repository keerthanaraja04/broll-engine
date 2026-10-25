"""
Unit tests for ClipStore — register, find_by_tags, get, remove, eviction.
Uses FakeRedis (no real Redis needed).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")

from tests.fixtures.conftest_footage import (
    FakeRedis, make_video_candidate, make_image_candidate,
)


@pytest.fixture()
def store_with_redis(fake_redis: FakeRedis) -> Any:
    from app.adapters.clip_store import ClipStore
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.adapters.clip_store._get_redis", return_value=fake_redis
    ):
        yield ClipStore()


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_stores_clip(
        self, store_with_redis: Any, fake_redis: FakeRedis
    ) -> None:
        c = make_video_candidate(cid="pixabay:1", tags=["city", "skyline"])
        await store_with_redis.register(c)
        stored = fake_redis._store.get("clipstore:pixabay:1")
        assert stored is not None
        import json
        d = json.loads(stored)
        assert d["id"] == "pixabay:1"

    @pytest.mark.asyncio
    async def test_register_creates_tag_index(
        self, store_with_redis: Any, fake_redis: FakeRedis
    ) -> None:
        c = make_video_candidate(cid="pixabay:2", tags=["city", "skyline"])
        await store_with_redis.register(c)
        assert "pixabay:2" in fake_redis._sets.get(
            "clipstore:tagidx:city", set())
        assert "pixabay:2" in fake_redis._sets.get(
            "clipstore:tagidx:skyline", set())

    @pytest.mark.asyncio
    async def test_register_adds_to_index(
        self, store_with_redis: Any, fake_redis: FakeRedis
    ) -> None:
        c = make_video_candidate(cid="pixabay:3")
        await store_with_redis.register(c)
        assert "pixabay:3" in fake_redis._sets.get(
            "clipstore:index", set())


class TestFindByTags:
    @pytest.mark.asyncio
    async def test_find_returns_matching_clip(
        self, store_with_redis: Any
    ) -> None:
        c = make_video_candidate(cid="pixabay:10", tags=["city", "skyline"])
        await store_with_redis.register(c)
        results = await store_with_redis.find_by_tags(["city"])
        assert len(results) == 1
        assert results[0].id == "pixabay:10"

    @pytest.mark.asyncio
    async def test_find_no_match_returns_empty(
        self, store_with_redis: Any
    ) -> None:
        c = make_video_candidate(cid="pixabay:11", tags=["forest"])
        await store_with_redis.register(c)
        assert await store_with_redis.find_by_tags(["ocean"]) == []

    @pytest.mark.asyncio
    async def test_find_empty_tags_returns_empty(
        self, store_with_redis: Any
    ) -> None:
        assert await store_with_redis.find_by_tags([]) == []

    @pytest.mark.asyncio
    async def test_find_ranks_by_match_count(
        self, store_with_redis: Any
    ) -> None:
        c1 = make_video_candidate(cid="pixabay:20", tags=["city", "skyline", "urban"])
        c2 = make_video_candidate(cid="pixabay:21", tags=["city", "forest"])
        await store_with_redis.register(c1)
        await store_with_redis.register(c2)
        results = await store_with_redis.find_by_tags(
            ["city", "skyline", "urban"], limit=2)
        assert results[0].id == "pixabay:20"

    @pytest.mark.asyncio
    async def test_find_filters_by_kind(
        self, store_with_redis: Any
    ) -> None:
        from app.adapters.footage_types import FootageKind
        v = make_video_candidate(cid="pixabay:30", tags=["nature"])
        i = make_image_candidate(cid="pixabay_img:31", tags=["nature"])
        await store_with_redis.register(v)
        await store_with_redis.register(i)
        results = await store_with_redis.find_by_tags(
            ["nature"], kind=FootageKind.VIDEO)
        assert len(results) == 1
        assert results[0].kind == FootageKind.VIDEO

    @pytest.mark.asyncio
    async def test_find_respects_limit(
        self, store_with_redis: Any
    ) -> None:
        for i in range(5):
            c = make_video_candidate(
                cid=f"pixabay:{40+i}", tags=["shared"])
            await store_with_redis.register(c)
        results = await store_with_redis.find_by_tags(["shared"], limit=2)
        assert len(results) == 2


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_clip(self, store_with_redis: Any) -> None:
        c = make_video_candidate(cid="pixabay:50")
        await store_with_redis.register(c)
        result = await store_with_redis.get("pixabay:50")
        assert result is not None
        assert result.id == "pixabay:50"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(
        self, store_with_redis: Any
    ) -> None:
        assert await store_with_redis.get("nonexistent") is None


class TestRemove:
    @pytest.mark.asyncio
    async def test_remove_deletes_clip(
        self, store_with_redis: Any, fake_redis: FakeRedis
    ) -> None:
        c = make_video_candidate(cid="pixabay:60", tags=["city"])
        await store_with_redis.register(c)
        await store_with_redis.remove("pixabay:60")
        assert "clipstore:pixabay:60" not in fake_redis._store
        assert "pixabay:60" not in fake_redis._sets.get(
            "clipstore:index", set())

    @pytest.mark.asyncio
    async def test_remove_cleans_tag_index(
        self, store_with_redis: Any, fake_redis: FakeRedis
    ) -> None:
        c = make_video_candidate(cid="pixabay:61", tags=["city", "skyline"])
        await store_with_redis.register(c)
        await store_with_redis.remove("pixabay:61")
        assert "pixabay:61" not in fake_redis._sets.get(
            "clipstore:tagidx:city", set())

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_safe(
        self, store_with_redis: Any
    ) -> None:
        await store_with_redis.remove("nonexistent")


class TestCount:
    @pytest.mark.asyncio
    async def test_count_zero_initially(
        self, store_with_redis: Any
    ) -> None:
        assert await store_with_redis.count() == 0

    @pytest.mark.asyncio
    async def test_count_after_register(
        self, store_with_redis: Any
    ) -> None:
        await store_with_redis.register(
            make_video_candidate(cid="pixabay:70"))
        await store_with_redis.register(
            make_video_candidate(cid="pixabay:71"))
        assert await store_with_redis.count() == 2
