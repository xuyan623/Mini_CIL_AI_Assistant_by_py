from __future__ import annotations

from pathlib import Path

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import PlanDecision, TaskSpec


def _task(capability_id: str | None, file_value: str = "") -> TaskSpec:
    params = {"file": file_value} if file_value else {}
    return TaskSpec(
        raw_description="raw",
        normalized_description="normalized",
        capability_id=capability_id,
        parameters=params,
    )


def test_internal_match_helpers_and_find_file_paths(tmp_path: Path, monkeypatch) -> None:
    engine = PlanEngine()
    sample = tmp_path / "src" / "Sam.c"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("int main(){return 0;}\n", encoding="utf-8")

    transcript = [{"command": "echo hi", "exit_code": 0}, {"command": "ai code check demo.c --start 1 --end 2", "exit_code": 1}]
    assert engine._command_seen("echo   hi", transcript) is True
    assert engine._command_result("ai code check demo.c --start 1 --end 2", transcript) == transcript[-1]
    assert engine._command_result("missing", transcript) is None

    dedup = engine._extract_find_matches(f"{sample}\n{sample}\n\nnot_a_path")
    assert dedup == [str(sample)]

    assert engine._find_file_matches(str(sample)) == [sample.resolve()]
    rel_file = tmp_path / "demo.py"
    rel_file.write_text("print(1)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert engine._find_file_matches("./demo.py") == [rel_file.resolve()]


def test_initial_step_building_and_resolution_branches(tmp_path: Path, monkeypatch) -> None:
    engine = PlanEngine()
    target = tmp_path / "Sam.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert engine.build_initial_steps(_task(None))[0] is False
    assert engine.build_initial_steps(_task("code.comment", ""))[0] is False

    ensure_task = TaskSpec(
        raw_description="d",
        normalized_description="d",
        capability_id="workflow.ensure_directory",
        parameters={"base_dir": ".", "dir_name": "AI"},
    )
    ok_ensure, steps_ensure, _ = engine.build_initial_steps(ensure_task)
    assert ok_ensure is True
    assert "mkdir -p" in steps_ensure[0].command

    missing_dir_name = TaskSpec(
        raw_description="d",
        normalized_description="d",
        capability_id="workflow.ensure_directory",
        parameters={"base_dir": "."},
    )
    ok_missing, _, msg_missing = engine.build_initial_steps(missing_dir_name)
    assert ok_missing is False
    assert "目录名称" in msg_missing

    multiple = [tmp_path / "a" / "same.c", tmp_path / "b" / "same.c"]
    multiple[0].parent.mkdir(parents=True, exist_ok=True)
    multiple[1].parent.mkdir(parents=True, exist_ok=True)
    multiple[0].write_text("1\n", encoding="utf-8")
    multiple[1].write_text("2\n", encoding="utf-8")
    monkeypatch.setattr(engine, "_find_file_matches", lambda _candidate: multiple)
    path, msg = engine._resolve_single_file("same.c")
    assert path is None
    assert "多个同名文件" in msg

    monkeypatch.setattr(engine, "_find_file_matches", lambda _candidate: [])
    path_missing, msg_missing_file = engine._resolve_single_file("none.c")
    assert path_missing is None
    assert "未找到目标文件" in msg_missing_file

    assert engine._resolve_existing_file(str(target)) == str(target.resolve())
    assert engine._resolve_existing_file("none.c") == ""


def test_workflow_decision_branches_and_fallbacks(tmp_path: Path, monkeypatch) -> None:
    engine = PlanEngine()
    target = tmp_path / "Sam.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")
    task = _task("workflow.code_fix", "Sam.c")
    monkeypatch.chdir(tmp_path)

    unresolved = engine.derive_workflow_decision(_task("code.check", ""), [])
    assert unresolved is not None
    assert unresolved.action == "need_input"

    missing_state = engine.derive_workflow_decision(_task("workflow.code_fix", "none.c"), [])
    assert missing_state is not None
    assert missing_state.action == "next"

    transcript_missing = [{"command": "find . -type f -name none.c", "stdout": "", "stderr": "", "exit_code": 0}]
    missing_after_find = engine.derive_workflow_decision(_task("workflow.code_fix", "none.c"), transcript_missing)
    assert missing_after_find is not None
    assert missing_after_find.action == "need_input"

    transcript_ambiguous = [
        {
            "command": "find . -type f -name none.c",
            "stdout": f"{tmp_path / 'a' / 'none.c'}\n{tmp_path / 'b' / 'none.c'}\n",
            "stderr": "",
            "exit_code": 0,
        }
    ]
    ambiguous = engine.derive_workflow_decision(_task("workflow.code_fix", "none.c"), transcript_ambiguous)
    assert ambiguous is not None
    assert ambiguous.action == "need_input"

    original_build_sequence = engine._build_sequence_for_mode
    monkeypatch.setattr(engine, "_build_sequence_for_mode", lambda mode, file_path: [])
    unsupported = engine.derive_workflow_decision(task, [])
    assert unsupported is not None
    assert unsupported.action == "abort"
    monkeypatch.setattr(engine, "_build_sequence_for_mode", original_build_sequence)

    sequence = engine._build_sequence_for_mode("fix", str(target.resolve()))
    fail_testf = engine.derive_workflow_decision(
        task,
        [{"command": sequence[0].command, "exit_code": 1, "stderr": "not found", "stdout": ""}],
    )
    assert fail_testf is not None
    assert fail_testf.action == "need_input"

    fail_other = engine.derive_workflow_decision(
        task,
        [{"command": sequence[1].command, "exit_code": 1, "stderr": "boom", "stdout": ""}],
    )
    assert fail_other is not None
    assert fail_other.action == "abort"

    done = engine.derive_workflow_decision(
        task,
        [{"command": step.command, "exit_code": 0, "stderr": "", "stdout": ""} for step in sequence],
    )
    assert done is not None
    assert done.action == "done"

    ok_placeholder, _, msg_placeholder = engine.resolve_placeholders("<FILE_PATH>", [], task)
    assert ok_placeholder is False
    assert "未解析占位符" in msg_placeholder

    fallback_empty = engine.fallback_next_from_suggestions(["   "], [], task)
    assert fallback_empty is None
    fallback_placeholder = engine.fallback_next_from_suggestions(["<FILE_PATH>"], [], task)
    assert fallback_placeholder is not None
    assert fallback_placeholder.action == "need_input"
    fallback_abort = engine.fallback_next_from_suggestions(["ai code check demo.c"], [], task)
    assert fallback_abort is not None
    assert fallback_abort.action == "abort"

    validate_short_ok, _ = engine.validate_ai_code_command("echo hi")
    assert validate_short_ok is True
