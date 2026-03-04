from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.history_service import HistoryService


def test_history_batch_append_and_commit(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(path_manager=manager)
    batch = service.begin_batch()

    event_id = service.append_event_in_batch(
        batch=batch,
        event_type="shell_step",
        input_text="echo hello",
        output_text="ok",
        ok=True,
        exit_code=0,
        metadata={"trace_id": "trace-1", "module": "shell", "phase": "execute"},
    )
    service.append_planner_trace_in_batch(
        batch=batch,
        trace_id="trace-1",
        stage="initial",
        request="req",
        response="resp",
        ok=True,
    )
    service.append_entity_in_batch(
        batch=batch,
        entity_type="file",
        value="./Sam.c",
        normalized_value="/home/mycode/Sam.c",
        source_event_id=event_id,
        trace_id="trace-1",
    )
    assert batch.dirty is True
    service.commit_batch(batch)
    assert batch.dirty is False

    events = service.list_events()
    traces = service.list_planner_traces()
    entities = service.list_entities()
    assert len(events) == 1
    assert len(traces) == 1
    assert len(entities) == 1
    assert events[0]["event_id"] == event_id


def test_history_compat_wrappers_still_persist(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = HistoryService(path_manager=manager)

    service.append_event("command", "ai chat hi", "ok", True, 0, {"trace_id": "trace-x"})
    service.append_planner_trace(
        trace_id="trace-x",
        stage="replan",
        request="q",
        response="a",
        ok=False,
        error_code="request_failed",
    )
    service.append_entity(entity_type="file", value="./a.c", normalized_value="/tmp/a.c", trace_id="trace-x")

    assert len(service.list_events()) == 1
    assert len(service.list_planner_traces()) == 1
    assert len(service.list_entities()) == 1
