from __future__ import annotations

from pathlib import Path

from ai_assistant.state import JsonStateStore
from ai_assistant.storage import safe_load_json


def test_set_json_writes_payload_without_read_modify_cycle(tmp_path: Path) -> None:
    store = JsonStateStore()
    target = tmp_path / "state.json"

    written = store.set_json(target, {"version": 1, "items": [1, 2, 3]})
    assert written["items"] == [1, 2, 3]
    assert store.flush() == 1
    loaded = safe_load_json(target, {})
    assert loaded == {"version": 1, "items": [1, 2, 3]}


def test_set_json_respects_normalizer_and_cache(tmp_path: Path) -> None:
    store = JsonStateStore()
    target = tmp_path / "state.json"

    normalized = store.set_json(
        target,
        {"items": "bad"},
        normalizer=lambda payload: {"items": payload["items"] if isinstance(payload.get("items"), list) else []},
    )
    assert normalized == {"items": []}
    cached = store.read_json(target, default_factory=lambda: {"items": ["fallback"]})
    assert cached == {"items": []}
    store.flush()
