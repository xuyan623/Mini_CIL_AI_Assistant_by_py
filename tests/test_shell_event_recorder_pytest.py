from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellService


class _DummyAIClient:
    def chat(self, messages, **kwargs):  # noqa: ANN001,ARG002
        return True, '{"summary":"x","steps":[{"command":"echo hello","purpose":"x"}]}'


def _service(tmp_path: Path) -> ShellService:
    manager = PathManager(project_root=tmp_path)
    history = HistoryService(manager)
    context = ContextService(manager)
    return ShellService(ai_client=_DummyAIClient(), history_service=history, context_service=context)  # type: ignore[arg-type]


def test_record_event_and_interrupt_with_batch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    trace_id = "trace-event"
    batch = service.history_service.begin_batch()
    service._trace_history_batches[trace_id] = batch

    service.event_recorder.record_event(
        event_type="shell_control",
        input_text="prompt",
        output_text="y",
        ok=True,
        exit_code=0,
        metadata={"trace_id": trace_id, "module": "shell", "phase": "control"},
        batch=batch,
    )
    service.event_recorder.record_interrupt(
        trace_id=trace_id,
        stage="runtime",
        reason="ctrl_c",
        step=2,
        command="echo hi",
        batch=batch,
    )
    service.history_service.commit_batch(batch)

    events = service.history_service.list_events()
    assert len(events) == 2
    assert {item["event_type"] for item in events} == {"shell_control", "interrupt"}


def test_extract_entities_from_step_output_with_batch(tmp_path: Path) -> None:
    service = _service(tmp_path)
    trace_id = "trace-entity"
    batch = service.history_service.begin_batch()
    service._trace_history_batches[trace_id] = batch

    service.event_recorder.extract_entities_from_step_output(
        command="test -f ./mycode/Sam.c",
        stdout="",
        stderr="",
        source_event_id="ev-1",
        trace_id=trace_id,
        batch=batch,
    )
    service.event_recorder.extract_entities_from_step_output(
        command="ai file find Sam.c",
        stdout="./mycode/Sam.c\n./mycode/other/Sam.c\n",
        stderr="",
        source_event_id="ev-2",
        trace_id=trace_id,
        batch=batch,
    )
    service.history_service.commit_batch(batch)

    entities = service.history_service.find_entities(entity_type="file")
    assert len(entities) >= 2
    assert any("Sam.c" in item["value"] for item in entities)
