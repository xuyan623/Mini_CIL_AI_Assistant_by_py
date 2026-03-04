from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant.paths import PathManager
from ai_assistant.planner.types import TaskSpec
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


def test_orchestrator_non_interactive_returns_plan_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = TaskSpec(raw_description="x", normalized_description="x", capability_id=None, parameters={})
    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, ["echo hi"], "note"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ok, message = service.orchestrator.run_workflow("x")
    assert ok is True
    assert "仅生成步骤草案" in message


def test_orchestrator_interactive_cancel_at_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = TaskSpec(raw_description="x", normalized_description="x", capability_id=None, parameters={})
    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, ["echo hi"], "note"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(service, "_confirm_with_prompt", lambda prompt: (False, "n"))
    monkeypatch.setattr(service, "_emit_runtime_output", lambda text: None)

    ok, message = service.orchestrator.run_workflow("x")
    assert ok is True
    assert "已取消执行" in message


def test_orchestrator_records_interrupt_on_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = TaskSpec(raw_description="x", normalized_description="x", capability_id=None, parameters={})
    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, ["echo hi"], "note"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(service, "_emit_runtime_output", lambda text: None)

    def _raise_interrupt(prompt: str) -> tuple[bool, str]:
        raise KeyboardInterrupt()

    monkeypatch.setattr(service, "_confirm_with_prompt", _raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        service.orchestrator.run_workflow("x")

    events = service.history_service.list_events()
    assert any(item.get("event_type") == "interrupt" for item in events)
