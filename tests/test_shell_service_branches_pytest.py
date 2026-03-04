from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_validate_and_parse_helpers(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service._load_json_object('{"a":1}') == {"a": 1}
    assert service._load_json_object("x {\"b\":2} y") == {"b": 2}
    assert service._load_json_object("no-json") is None

    parsed = service._parse_planner_steps_json('{"steps":["echo a", {"command":"echo b"}]}')
    assert parsed == ["echo a", "echo b"]
    assert service._parse_initial_steps('{"steps":["echo a"]}') == ["echo a"]

    assert service._is_natural_language_line("首先，用户描述是...")
    assert not service._is_natural_language_line("echo hello")
    assert service._contains_placeholder_token("<FILE_PATH>")
    assert service._contains_placeholder_token("<abc>")
    assert not service._contains_placeholder_token("echo ok")

    ok_empty, msg_empty = service._validate_shell_command("")
    assert ok_empty is False
    assert "命令为空" in msg_empty
    ok_placeholder, _ = service._validate_shell_command("<FILE_PATH>")
    assert ok_placeholder is False
    ok_text, _ = service._validate_shell_command("首先描述")
    assert ok_text is False
    ok_cmd, _ = service._validate_shell_command("echo hello")
    assert ok_cmd is True


def test_vote_reference_with_ai_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    candidate = EntityRecord(
        entity_id="e1",
        entity_type="file",
        value="/tmp/a.c",
        normalized_value="/tmp/a.c",
        source_event_id="ev1",
        trace_id="tr1",
        created_at="2026-03-04T00:00:00+00:00",
    )

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(
            ok=True,
            content='{"selected_entity_id":"e1","confidence":1.5,"reason":"ok"}',
            attempts=[],
            used_profile="p",
        ),
    )
    selected, confidence, reason = service._vote_reference_with_ai(
        description="这个文件",
        candidates=[candidate],
        trace_id="t1",
    )
    assert selected == "e1"
    assert confidence == 1.0
    assert reason == "ok"

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=False, content="boom", error_code="x"),
    )
    selected2, confidence2, reason2 = service._vote_reference_with_ai(
        description="这个文件",
        candidates=[candidate],
        trace_id="t2",
    )
    assert selected2 == ""
    assert confidence2 == 0.0
    assert reason2 == "boom"


def test_resolve_references_ambiguous_conflict_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = TaskSpec(raw_description="d", normalized_description="这个文件", capability_id=None, parameters={})
    entity = EntityRecord(
        entity_id="e1",
        entity_type="file",
        value=str(tmp_path / "Sam.c"),
        normalized_value=str((tmp_path / "Sam.c").resolve()),
        source_event_id="ev1",
        trace_id="tr1",
        created_at="2026-03-04T00:00:00+00:00",
    )
    (tmp_path / "Sam.c").write_text("int main(){return 0;}\n", encoding="utf-8")

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(status="ambiguous", candidates=[entity], reason="m"),  # noqa: ARG005
    )
    ok_amb, _, msg_amb = service._resolve_references(task, "trace-amb")
    assert ok_amb is False
    assert "多个候选" in msg_amb

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(status="resolved", selected_entity=entity, candidates=[entity], reason="ok"),  # noqa: ARG005
    )
    monkeypatch.setattr(service, "_vote_reference_with_ai", lambda **kwargs: ("another", 0.9, "conflict"))
    ok_conflict, _, msg_conflict = service._resolve_references(
        TaskSpec(raw_description="d", normalized_description="这个文件", capability_id=None, parameters={}),
        "trace-conflict",
    )
    assert ok_conflict is False
    assert "冲突" in msg_conflict

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(status="missing", selected_entity=None, candidates=[entity], reason="missing"),  # noqa: ARG005
    )
    monkeypatch.setattr(service, "_vote_reference_with_ai", lambda **kwargs: ("e1", 0.9, "pick"))
    ok_missing, resolved_task, msg_missing = service._resolve_references(
        TaskSpec(raw_description="d", normalized_description="备份这个文件", capability_id=None, parameters={}),
        "trace-missing",
    )
    assert ok_missing is True
    assert resolved_task.parameters.get("file")
    assert msg_missing == ""


def test_plan_next_with_ai_invalid_json_and_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    transcript = [{"command": "echo hello", "stdout": "hello", "stderr": "", "exit_code": 0}]

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content="not-json"),
    )
    monkeypatch.setattr(
        service,
        "_repair_planner_output",
        lambda **kwargs: AIResponseEnvelope(ok=False, content="repair-failed"),
    )
    ok1, decision1 = service._plan_next_with_ai("desc", transcript, [], "trace-1")
    assert ok1 is False
    assert decision1.action == "abort"

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"action":"next","command":"echo hi","message":"go"}'),
    )
    ok2, decision2 = service._plan_next_with_ai("desc", transcript, [], "trace-2")
    assert ok2 is True
    assert decision2.action == "next"
    assert decision2.command == "echo hi"

