from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.history_service import HistoryService


def test_history_basic_operations_and_filters(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(manager)

    service.clear()
    service.append_message("user", "hello")
    service.append_exchange("u1", "a1")
    service.append_event("command", "ai file ls .", "ok", True, 0, metadata={"module": "file"})
    service.append_event("shell_step", "echo hi", "done", True, 0, metadata={"module": "shell"})
    service.append_planner_trace(trace_id="t1", stage="initial", request="r", response="s", ok=True)
    entity = service.append_entity(entity_type="file", value="/tmp/a.c", normalized_value="/tmp/a.c")
    assert entity["entity_type"] == "file"

    related = service.format_related_events("echo")
    assert "相关的历史事件" in related
    recent = service.format_recent_events()
    assert "最近命令输入与输出" in recent

    found = service.find_entities(entity_type="file", keyword="a.c")
    assert len(found) >= 1
    assert found[0]["entity_type"] == "file"

    messages = service.build_messages_for_request("question", extra_system_messages=["extra"])
    merged = "\n".join(item.get("content", "") for item in messages)
    assert "extra" in merged
    assert "question" in merged


def test_history_trim_and_normalize_paths(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(manager)
    for index in range(15):
        service.append_exchange(f"u{index}", f"a{index}")

    service.trim_and_summarize(lambda messages: "summary-text")
    payload = service.load_payload()
    system_messages = [item for item in payload["messages"] if item.get("role") == "system"]
    assert any("历史总结" in item.get("content", "") for item in system_messages)

    # force malformed file then load
    history_path = manager.history_path
    history_path.write_text('{"version":6,"messages":"bad","events":"bad"}', encoding="utf-8")
    fixed = service.load_payload()
    assert isinstance(fixed["messages"], list)
    assert isinstance(fixed["events"], list)


def test_history_resolution_trace_and_recent_non_system(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(manager)
    service.append_exchange("u", "a")
    service.append_resolution_trace(trace_id="t2", request="这个文件", response="/tmp/a.c", ok=True, metadata={"source": "local"})
    events = service.list_events()
    assert any(item.get("event_type") == "resolution" for item in events)

    recent = service.get_recent_non_system_messages(limit=2)
    assert len(recent) <= 2

