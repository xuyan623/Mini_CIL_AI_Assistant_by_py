from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.history_service import HistoryService


def test_history_load_does_not_force_write(monkeypatch, tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(manager)
    write_calls = {"count": 0}

    def _count_write(path, payload):  # noqa: ARG001
        write_calls["count"] += 1

    monkeypatch.setattr("ai_assistant.state.json_state_store.atomic_write_json", _count_write)
    service.load_payload()
    service.load_payload()
    assert write_calls["count"] == 0


def test_append_event_write_budget(monkeypatch, tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(manager)
    write_calls = {"count": 0}

    def _count_write(path, payload):  # noqa: ARG001
        write_calls["count"] += 1

    monkeypatch.setattr("ai_assistant.state.json_state_store.atomic_write_json", _count_write)
    service.append_event("command", "in", "out", True, 0)
    assert write_calls["count"] <= 1

