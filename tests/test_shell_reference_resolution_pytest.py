from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant.paths import PathManager
from ai_assistant.planner.types import AIResponseEnvelope, EntityRecord, ReferenceResolutionResult, TaskSpec
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


def _entity(tmp_path: Path) -> EntityRecord:
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("int main(void){return 0;}\n", encoding="utf-8")
    return EntityRecord(
        entity_id="e1",
        entity_type="file",
        value=str(target),
        normalized_value=str(target),
        source_event_id="ev1",
        trace_id="tr1",
        created_at="2026-03-04T00:00:00+00:00",
        metadata={},
    )


def test_vote_reference_repair_and_confidence_clamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    candidate = _entity(tmp_path)
    monkeypatch.setattr(service, "_request_ai", lambda **kwargs: AIResponseEnvelope(ok=True, content="not-json"))
    monkeypatch.setattr(
        service,
        "_repair_planner_output",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"selected_entity_id":"e1","confidence":9,"reason":"ok"}'),
    )
    selected_id, confidence, reason = service.reference_resolution.vote_reference_with_ai(
        description="这个文件",
        candidates=[candidate],
        trace_id="trace-vote",
    )
    assert selected_id == "e1"
    assert confidence == 1.0
    assert reason == "ok"


def test_resolve_detects_vote_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    candidate = _entity(tmp_path)
    task = TaskSpec(raw_description="d", normalized_description="这个文件", capability_id=None, parameters={})

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(  # noqa: ARG005
            status="resolved",
            selected_entity=candidate,
            candidates=[candidate],
            reason="ok",
        ),
    )
    monkeypatch.setattr(service, "_vote_reference_with_ai", lambda **kwargs: ("another-id", 0.9, "conflict"))

    ok, _, message = service.reference_resolution.resolve(task, "trace-conflict")
    assert ok is False
    assert "冲突" in message


def test_resolve_missing_accepts_high_confidence_model_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    candidate = _entity(tmp_path)
    task = TaskSpec(raw_description="d", normalized_description="这个文件", capability_id=None, parameters={})

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(  # noqa: ARG005
            status="missing",
            selected_entity=None,
            candidates=[candidate],
            reason="missing",
        ),
    )
    monkeypatch.setattr(service, "_vote_reference_with_ai", lambda **kwargs: ("e1", 0.9, "ok"))

    ok, resolved_task, message = service.reference_resolution.resolve(task, "trace-missing")
    assert ok is True
    assert message == ""
    assert resolved_task.parameters.get("file", "").endswith("Sam.c")
