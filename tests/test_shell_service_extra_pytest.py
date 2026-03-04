from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_assistant.paths import PathManager
from ai_assistant.planner.types import AIResponseEnvelope, EntityRecord, PlanDecision, ReferenceResolutionResult, TaskSpec
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellSafetyReport, ShellService


class _DummyAIClient:
    def chat(self, messages, **kwargs):  # noqa: ANN001,ARG002
        return True, '{"summary":"x","steps":[{"command":"echo hello","purpose":"x"}]}'


def _service(tmp_path: Path) -> ShellService:
    manager = PathManager(project_root=tmp_path)
    history = HistoryService(manager)
    context = ContextService(manager)
    return ShellService(ai_client=_DummyAIClient(), history_service=history, context_service=context)  # type: ignore[arg-type]


def _task(description: str = "检查", capability_id: str | None = None, parameters: dict[str, str] | None = None) -> TaskSpec:
    return TaskSpec(
        raw_description=description,
        normalized_description=description,
        capability_id=capability_id,
        parameters=parameters or {},
    )


def test_confirm_and_repair_prompt_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    service = _service(tmp_path)
    captured_prompts: list[str] = []
    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: captured_prompts.append(kwargs["prompt"]) or AIResponseEnvelope(ok=True, content="{}"),
    )
    service._repair_planner_output(
        trace_id="t1",
        stage="vote",
        task_description="d",
        raw_content="raw",
        expect="reference_vote",
    )
    service._repair_planner_output(
        trace_id="t2",
        stage="decision",
        task_description="d",
        raw_content="raw",
        expect="decision",
    )
    assert "selected_entity_id" in captured_prompts[0]
    assert '"action":"next|done|need_input|abort"' in captured_prompts[1]

    service._active_trace_context = {"trace_id": "trace-1", "step": 2}
    records: list[dict[str, object]] = []
    monkeypatch.setattr(service, "_record_event", lambda **kwargs: records.append(kwargs) or "event-id")
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    accepted, token = service._confirm_with_prompt("继续?(y/n): ")
    assert accepted is True
    assert token == "y"
    assert records and records[0]["event_type"] == "shell_control"
    assert "请输入 y 或 n" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    eof_ok, eof_token = service._confirm_with_prompt("继续?(y/n): ")
    assert eof_ok is False
    assert eof_token == "__eof__"


def test_vote_reference_and_resolution_missing_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    target = tmp_path / "Sam.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")
    entity = EntityRecord(
        entity_id="e1",
        entity_type="file",
        value=str(target),
        normalized_value=str(target),
        source_event_id="ev1",
        trace_id="tr1",
        created_at="2026-03-04T00:00:00+00:00",
    )

    selected_empty, confidence_empty, reason_empty = service._vote_reference_with_ai(
        description="这个文件",
        candidates=[],
        trace_id="t0",
    )
    assert selected_empty == ""
    assert confidence_empty == 0.0
    assert "无候选" in reason_empty

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content="not-json"),
    )
    monkeypatch.setattr(
        service,
        "_repair_planner_output",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"selected_entity_id":"e1","confidence":"bad","reason":"r"}'),
    )
    selected, confidence, reason = service._vote_reference_with_ai(
        description="这个文件",
        candidates=[entity],
        trace_id="t1",
    )
    assert selected == "e1"
    assert confidence == 0.0
    assert reason == "r"

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"selected_entity_id":"e1","confidence":9,"reason":"r2"}'),
    )
    _, confidence2, _ = service._vote_reference_with_ai(description="这个文件", candidates=[entity], trace_id="t2")
    assert confidence2 == 1.0

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(status="missing", selected_entity=None, candidates=[], reason="none"),  # noqa: ARG005
    )
    ok_missing, _, msg_missing = service._resolve_references(_task("这个文件"), "trace-missing")
    assert ok_missing is False
    assert "无法解析" in msg_missing

    monkeypatch.setattr(
        service.reference_resolver,
        "resolve_file_reference",
        lambda description, entities: ReferenceResolutionResult(status="noop", selected_entity=None, candidates=[], reason="noop"),  # noqa: ARG005
    )
    ok_noop, _, msg_noop = service._resolve_references(_task("这个文件"), "trace-noop")
    assert ok_noop is True
    assert msg_noop == ""

    assert service._normalize_file_value("") == ""
    assert service._normalize_file_value(str(target.resolve())) == str(target.resolve())


def test_plan_from_description_and_generate_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)

    invalid_task = TaskSpec(
        raw_description="x",
        normalized_description="x",
        capability_id="__invalid__",
        note="❌ invalid",
        parameters={},
    )
    monkeypatch.setattr(service.task_interpreter, "interpret", lambda description, events: invalid_task)  # noqa: ARG005
    ok_invalid, _, _, msg_invalid = service._plan_from_description("x", "trace-1")
    assert ok_invalid is False
    assert "invalid" in msg_invalid

    valid_task = _task("x", capability_id="code.check", parameters={"file": "Sam.c"})
    monkeypatch.setattr(service.task_interpreter, "interpret", lambda description, events: valid_task)  # noqa: ARG005
    monkeypatch.setattr(service.reference_resolution, "resolve", lambda task, trace_id: (False, task, "resolve failed"))
    ok_resolve, _, _, msg_resolve = service._plan_from_description("x", "trace-2")
    assert ok_resolve is False
    assert "resolve failed" in msg_resolve

    monkeypatch.setattr(service.reference_resolution, "resolve", lambda task, trace_id: (True, task, ""))  # noqa: ARG005
    monkeypatch.setattr(service, "_interpret_task_with_ai", lambda base_task, trace_id: None)  # noqa: ARG005
    monkeypatch.setattr(service.plan_engine, "build_initial_steps", lambda task: (True, [], ""))  # noqa: ARG005
    ok_empty_steps, _, _, msg_empty_steps = service._plan_from_description("x", "trace-3")
    assert ok_empty_steps is False
    assert "未生成有效命令" in msg_empty_steps

    ai_task = _task("x", capability_id=None, parameters={})
    monkeypatch.setattr(service.task_interpreter, "interpret", lambda description, events: ai_task)  # noqa: ARG005
    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=False, content="failed", error_code="request_failed"),
    )
    ok_ai_fail, _, _, msg_ai_fail = service._plan_from_description("x", "trace-4")
    assert ok_ai_fail is False
    assert msg_ai_fail == "failed"

    monkeypatch.setattr(
        service,
        "_plan_from_description",
        lambda description, trace_id: (False, _task(description), [], "bad"),  # noqa: ARG005
    )
    ok_initial, commands, note = service.generate_initial_steps("x")
    assert ok_initial is False
    assert commands == []
    assert note == "bad"

    monkeypatch.setattr(
        service,
        "_plan_from_description",
        lambda description, trace_id: (True, _task(description, parameters={}), ["echo hi"], "note"),  # noqa: ARG005
    )
    with_note_ok, with_note_text = service.generate_command("x")
    assert with_note_ok is True
    assert "📌 note" in with_note_text

    retry_task = _task("x", parameters={})
    retry_task.retry_note = "retry this"
    monkeypatch.setattr(
        service,
        "_plan_from_description",
        lambda description, trace_id: (True, retry_task, ["echo hi"], "note"),  # noqa: ARG005
    )
    with_retry_ok, with_retry_text = service.generate_command("x")
    assert with_retry_ok is True
    assert "📌 retry this" in with_retry_text


def test_plan_next_and_extract_entity_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    transcript = [{"command": "echo a", "stdout": "a", "stderr": "", "exit_code": 0}]

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content="not-json"),
    )
    monkeypatch.setattr(
        service,
        "_repair_planner_output",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"action":"done","message":"ok"}'),
    )
    ok_done, decision_done = service._plan_next_with_ai("desc", transcript, [], "trace-a")
    assert ok_done is True
    assert decision_done.action == "done"

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"action":"next","command":"首先这是解释","message":"m"}'),
    )
    bad_next_ok, bad_next = service._plan_next_with_ai("desc", transcript, [], "trace-b")
    assert bad_next_ok is False
    assert bad_next.action == "abort"
    assert "不是可执行命令" in bad_next.message

    monkeypatch.setattr(
        service,
        "_request_ai",
        lambda **kwargs: AIResponseEnvelope(ok=True, content='{"action":"unknown","command":"echo hi","message":"m"}'),
    )
    unknown_ok, unknown_decision = service._plan_next_with_ai("desc", transcript, [], "trace-c")
    assert unknown_ok is False
    assert unknown_decision.message == "未能解析下一步决策 JSON"

    paths = service._extract_paths_from_text("- /tmp/a\n└─ ./b/c.txt\n\nplain text\n")
    assert "/tmp/a" in paths
    assert "./b/c.txt" in paths

    captured_entities: list[dict[str, object]] = []
    monkeypatch.setattr(service, "_append_file_entity", lambda **kwargs: captured_entities.append(kwargs))
    service._extract_entities_from_step_output(
        command="test -f '/tmp/a.c'",
        stdout="",
        stderr="",
        source_event_id="ev1",
        trace_id="tr1",
    )
    service._extract_entities_from_step_output(
        command="ai backup create /tmp/b.c --keep 5",
        stdout="",
        stderr="",
        source_event_id="ev2",
        trace_id="tr2",
    )
    service._extract_entities_from_step_output(
        command="ai file find Sam.c",
        stdout="└─ ./x/Sam.c\n- ./y/Sam.c\n",
        stderr="",
        source_event_id="ev3",
        trace_id="tr3",
    )
    assert len(captured_entities) >= 4
    assert any(item.get("metadata", {}).get("source") == "test -f" for item in captured_entities)
    assert any(item.get("metadata", {}).get("source") == "ai backup create" for item in captured_entities)
    assert any(item.get("metadata", {}).get("source") == "ai file find" for item in captured_entities)

    service2 = _service(tmp_path)
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(service2.history_service, "append_entity", lambda **kwargs: recorded.append(kwargs))
    service2._append_file_entity(value="", source_event_id="ev4", trace_id="tr4", confidence=0.9)
    assert recorded == []


def test_run_workflow_core_branch_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = _task("做点事")

    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, [], "note"))
    ok_no_steps, message_no_steps = service._run_workflow("x")
    assert ok_no_steps is False
    assert "未生成有效步骤" in message_no_steps

    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, ["echo one"], "note"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(service, "_emit_runtime_output", lambda text: None)

    interrupts: list[dict[str, object]] = []
    monkeypatch.setattr(service, "_record_interrupt", lambda **kwargs: interrupts.append(kwargs))
    monkeypatch.setattr(service, "_confirm_with_prompt", lambda prompt: (False, "__eof__"))
    ok_start_eof, message_start_eof = service._run_workflow("x")
    assert ok_start_eof is True
    assert "已取消执行" in message_start_eof
    assert interrupts and interrupts[-1]["stage"] == "start"

    monkeypatch.setattr(service, "_confirm_with_prompt", lambda prompt: (True, "y"))
    monkeypatch.setattr(service.plan_engine, "resolve_placeholders", lambda command, transcript, task: (False, "", "resolve failed"))
    ok_resolve_fail, msg_resolve_fail = service._run_workflow("x")
    assert ok_resolve_fail is False
    assert "resolve failed" in msg_resolve_fail

    monkeypatch.setattr(service.plan_engine, "resolve_placeholders", lambda command, transcript, task: (True, "echo one", ""))
    monkeypatch.setattr(service, "_validate_shell_command", lambda command: (False, "bad command"))
    ok_cmd_fail, msg_cmd_fail = service._run_workflow("x")
    assert ok_cmd_fail is False
    assert msg_cmd_fail == "bad command"

    monkeypatch.setattr(service, "_validate_shell_command", lambda command: (True, ""))
    monkeypatch.setattr(service.plan_engine, "validate_ai_code_command", lambda command: (False, "bad ai code"))
    ok_ai_code_fail, msg_ai_code_fail = service._run_workflow("x")
    assert ok_ai_code_fail is False
    assert msg_ai_code_fail == "bad ai code"


def test_run_workflow_decision_and_interrupt_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    task = _task("做点事")
    monkeypatch.setattr(service.planner_adapter, "plan_from_description", lambda description, trace_id: (True, task, ["echo one"], "note"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    outputs: list[str] = []
    monkeypatch.setattr(service, "_emit_runtime_output", lambda text: outputs.append(text))
    monkeypatch.setattr(service.plan_engine, "resolve_placeholders", lambda command, transcript, task: (True, "sudo ls", ""))
    monkeypatch.setattr(service, "_validate_shell_command", lambda command: (True, ""))
    monkeypatch.setattr(service.plan_engine, "validate_ai_code_command", lambda command: (True, ""))
    monkeypatch.setattr(service, "safety_check", lambda command: ShellSafetyReport(safe=False, warnings=["w1"]))
    answers = iter([(True, "y"), (False, "__eof__")])
    monkeypatch.setattr(service, "_confirm_with_prompt", lambda prompt: next(answers))
    captured_interrupts: list[dict[str, object]] = []
    monkeypatch.setattr(service, "_record_interrupt", lambda **kwargs: captured_interrupts.append(kwargs))
    ok_warn, msg_warn = service._run_workflow("x")
    assert ok_warn is True
    assert "第 1 步已取消" in msg_warn
    assert any("[WARN] 安全警告：" in item for item in outputs)
    assert captured_interrupts and captured_interrupts[-1]["stage"] == "step_confirm"

    monkeypatch.setattr(service, "_confirm_with_prompt", lambda prompt: (True, "y"))
    monkeypatch.setattr(
        service.step_executor,
        "execute",
        lambda command: SimpleNamespace(ok=True, exit_code=0, stdout="ok\n", stderr=""),
    )
    monkeypatch.setattr(service.plan_engine, "derive_retry_decision", lambda transcript: None)
    monkeypatch.setattr(service.plan_engine, "derive_workflow_decision", lambda task, transcript: PlanDecision(action="need_input", message="need more"))
    need_input_ok, need_input_msg = service._run_workflow("x")
    assert need_input_ok is False
    assert "need more" in need_input_msg

    monkeypatch.setattr(service.plan_engine, "derive_workflow_decision", lambda task, transcript: PlanDecision(action="abort", message="abort now"))
    abort_ok, abort_msg = service._run_workflow("x")
    assert abort_ok is False
    assert "abort now" in abort_msg

    monkeypatch.setattr(service.plan_engine, "derive_workflow_decision", lambda task, transcript: None)
    monkeypatch.setattr(service.planner_adapter, "plan_next", lambda description, transcript, suggested_steps, trace_id: (False, PlanDecision(action="abort", message="next failed")))
    monkeypatch.setattr(service.plan_engine, "fallback_next_from_suggestions", lambda suggested_steps, transcript, task: None)
    monkeypatch.setattr(
        service.step_executor,
        "execute",
        lambda command: SimpleNamespace(ok=False, exit_code=1, stdout="", stderr="bad"),
    )
    fail_ok, fail_msg = service._run_workflow("x")
    assert fail_ok is False
    assert "上一步失败且无法生成下一步" in fail_msg

    monkeypatch.setattr(
        service.step_executor,
        "execute",
        lambda command: SimpleNamespace(ok=True, exit_code=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        service.planner_adapter,
        "plan_next",
        lambda description, transcript, suggested_steps, trace_id: (True, PlanDecision(action="next", command="", message="continue")),
    )
    monkeypatch.setattr(service.plan_engine, "fallback_next_from_suggestions", lambda suggested_steps, transcript, task: None)
    done_ok, done_msg = service._run_workflow("x")
    assert done_ok is True
    assert "执行完成" in done_msg

    monkeypatch.setattr(
        service.planner_adapter,
        "plan_next",
        lambda description, transcript, suggested_steps, trace_id: (True, PlanDecision(action="next", command="", message="continue")),
    )
    monkeypatch.setattr(
        service.plan_engine,
        "fallback_next_from_suggestions",
        lambda suggested_steps, transcript, task: PlanDecision(action="need_input", message="bad fallback"),
    )
    fallback_ok, fallback_msg = service._run_workflow("x")
    assert fallback_ok is False
    assert "bad fallback" in fallback_msg

    service.max_steps = 1
    monkeypatch.setattr(
        service.planner_adapter,
        "plan_next",
        lambda description, transcript, suggested_steps, trace_id: (True, PlanDecision(action="next", command="echo two", message="next")),
    )
    monkeypatch.setattr(
        service.plan_engine,
        "fallback_next_from_suggestions",
        lambda suggested_steps, transcript, task: PlanDecision(action="next", command="echo fallback", message=""),
    )
    max_ok, max_msg = service._run_workflow("x")
    assert max_ok is False
    assert "已达到最大步骤数" in max_msg

    def _raise_keyboard_interrupt(prompt: str) -> tuple[bool, str]:
        raise KeyboardInterrupt()

    monkeypatch.setattr(service, "_confirm_with_prompt", _raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        service._run_workflow("x")
