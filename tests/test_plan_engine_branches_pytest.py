from __future__ import annotations

from pathlib import Path

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import TaskSpec


def _task(capability_id: str | None, file_value: str = "") -> TaskSpec:
    params = {"file": file_value} if file_value else {}
    return TaskSpec(
        raw_description="d",
        normalized_description="d",
        capability_id=capability_id,
        parameters=params,
    )


def test_build_initial_steps_for_code_and_backup(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = tmp_path / "Sam.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")

    ok_fix, steps_fix, _ = engine.build_initial_steps(_task("workflow.code_fix", str(target)))
    assert ok_fix is True
    assert any("ai code check" in step.command for step in steps_fix)
    assert any("ai code optimize" in step.command for step in steps_fix)

    ok_backup, steps_backup, _ = engine.build_initial_steps(_task("backup.create", str(target)))
    assert ok_backup is True
    assert len(steps_backup) == 1
    assert "ai backup create" in steps_backup[0].command


def test_build_initial_steps_missing_file_and_ensure_directory() -> None:
    engine = PlanEngine()
    ok_missing, _, msg_missing = engine.build_initial_steps(_task("code.comment", ""))
    assert ok_missing is False
    assert "缺少目标文件名" in msg_missing

    task_dir = TaskSpec(
        raw_description="d",
        normalized_description="d",
        capability_id="workflow.ensure_directory",
        parameters={"base_dir": ".", "dir_name": "AI"},
    )
    ok_dir, steps_dir, _ = engine.build_initial_steps(task_dir)
    assert ok_dir is True
    assert "mkdir -p" in steps_dir[0].command


def test_discovery_and_workflow_decisions(tmp_path: Path, monkeypatch) -> None:
    engine = PlanEngine()
    target = tmp_path / "Sam.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")
    task = _task("workflow.code_fix", "NotExist.c")

    monkeypatch.chdir(tmp_path)
    decision_start = engine.derive_workflow_decision(task, [])
    assert decision_start is not None
    assert decision_start.action == "next"
    assert "find . -type f -name" in decision_start.command

    transcript = [
        {
            "command": "find . -type f -name Sam.c",
            "stdout": str(target),
            "stderr": "",
            "exit_code": 0,
        }
    ]
    decision_after_find = engine.derive_workflow_decision(task, transcript)
    assert decision_after_find is not None
    assert decision_after_find.action == "next"
    assert decision_after_find.command.startswith("test -f ") or "ai code check" in decision_after_find.command


def test_retry_and_fallback_and_validation() -> None:
    engine = PlanEngine(step_timeout_seconds=30)
    timeout_transcript = [
        {"command": "ai code comment demo.c --start 1 --end 2", "exit_code": 124, "stderr": "命令超时（30s）"}
    ]
    retry = engine.derive_retry_decision(timeout_transcript)
    assert retry is not None
    assert "--yes" in retry.command

    timeout_wrap = engine.derive_retry_decision([{"command": "echo x", "exit_code": 124, "stderr": "命令超时"}])
    assert timeout_wrap is not None
    assert timeout_wrap.command.startswith("timeout 30s")

    ok, msg = engine.validate_ai_code_command("ai code check demo.c")
    assert ok is False
    assert "--start" in msg and "--end" in msg

    fallback = engine.fallback_next_from_suggestions(["echo hi"], [], _task("code.check", "demo.c"))
    assert fallback is not None
    assert fallback.action == "next"
