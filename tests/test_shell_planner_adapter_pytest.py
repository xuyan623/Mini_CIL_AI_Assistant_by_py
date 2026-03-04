from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.planner.types import AIResponseEnvelope, TaskSpec
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


def test_request_ai_caches_extra_system_messages(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    trace_id = "trace-cache"
    call_count = {"build": 0}

    def _build_messages(task_description: str) -> list[str]:  # noqa: ARG001
        call_count["build"] += 1
        return ["system-extra"]

    monkeypatch.setattr(service, "_build_extra_system_messages", _build_messages)
    monkeypatch.setattr(
        service.history_service,
        "build_messages_for_request",
        lambda **kwargs: [{"role": "user", "content": kwargs["user_prompt"]}],
    )
    monkeypatch.setattr(
        service.ai_gateway,
        "chat",
        lambda *args, **kwargs: AIResponseEnvelope(
            ok=True,
            content='{"ok":true}',
            attempts=[{"profile_id": "p2", "ok": True}],
            used_profile="p2",
        ),
    )

    service.planner_adapter.request_ai(
        prompt="p1",
        trace_id=trace_id,
        stage="initial",
        task_description="检查",
        max_tokens=64,
        temperature=0.2,
        timeout=30,
    )
    service.planner_adapter.request_ai(
        prompt="p2",
        trace_id=trace_id,
        stage="replan",
        task_description="检查",
        max_tokens=64,
        temperature=0.2,
        timeout=30,
    )
    assert call_count["build"] == 1
    assert service._trace_profile_order[trace_id][0] == "p2"


def test_plan_from_description_repairs_non_json_steps(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    task = TaskSpec(raw_description="x", normalized_description="x", capability_id=None, parameters={})
    monkeypatch.setattr(service.task_interpreter, "interpret", lambda description, events: task)  # noqa: ARG005
    monkeypatch.setattr(service.reference_resolution, "resolve", lambda task, trace_id: (True, task, ""))  # noqa: ARG005
    monkeypatch.setattr(service, "_interpret_task_with_ai", lambda base_task, trace_id: None)  # noqa: ARG005
    monkeypatch.setattr(service, "_request_ai", lambda **kwargs: AIResponseEnvelope(ok=True, content="not-json"))
    monkeypatch.setattr(
        service,
        "_repair_planner_output",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"steps":[{"command":"echo hi","purpose":"x"}]}'),
    )

    ok, _, commands, _ = service.planner_adapter.plan_from_description("x", "trace-repair")
    assert ok is True
    assert commands == ["echo hi"]


def test_plan_next_returns_abort_when_action_invalid(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"action":"invalid","command":"echo hi","message":"x"}'),
    )
    ok, decision = service.planner_adapter.plan_next("d", [], [], "trace-next")
    assert ok is False
    assert decision.action == "abort"
    assert "JSON" in decision.message
