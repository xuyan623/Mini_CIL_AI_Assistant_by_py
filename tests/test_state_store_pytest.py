from __future__ import annotations

from pathlib import Path

from ai_assistant.state import JsonStateStore


def test_state_store_read_update_flush(tmp_path: Path) -> None:
    store = JsonStateStore()
    target = tmp_path / "state.json"

    def default_factory() -> dict[str, object]:
        return {"version": 1, "items": []}

    loaded = store.read_json(target, default_factory=default_factory)
    assert loaded["version"] == 1
    assert loaded["items"] == []

    updated = store.update_json(
        target,
        updater=lambda payload: {"version": payload["version"], "items": ["a"]},
        default_factory=default_factory,
    )
    assert updated["items"] == ["a"]
    writes = store.flush()
    assert writes == 1
    assert target.exists()

    reloaded = store.read_json(target, default_factory=default_factory)
    assert reloaded["items"] == ["a"]


def test_state_store_cache_and_reset(tmp_path: Path) -> None:
    store = JsonStateStore()
    target = tmp_path / "state.json"
    target.write_text('{"version":1,"items":["x"]}', encoding="utf-8")

    first = store.read_json(target, default_factory=lambda: {"version": 1, "items": []})
    second = store.read_json(target, default_factory=lambda: {"version": 1, "items": []})
    assert first == second
    stats_before = store.get_io_stats()
    assert "cache_entries" in stats_before

    store.reset_transaction()
    stats_after = store.get_io_stats()
    assert stats_after["write_count"] == 0

