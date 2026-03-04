from __future__ import annotations

from ai_assistant.planner.task_interpreter import TaskInterpreter


def test_local_capability_match_for_reference_based_modify_request() -> None:
    interpreter = TaskInterpreter()

    task = interpreter.interpret("根据建议修改这个文件", [])

    assert task.capability_id == "workflow.code_fix"
    assert "file" in task.missing_parameters
    assert task.note == ""


def test_local_capability_match_for_reference_based_backup_request() -> None:
    interpreter = TaskInterpreter()

    task = interpreter.interpret("帮我备份这个文件", [])

    assert task.capability_id == "backup.create"
    assert task.missing_parameters == []
    assert task.note == ""


def test_local_capability_match_with_explicit_file_path() -> None:
    interpreter = TaskInterpreter()

    task = interpreter.interpret("根据建议修改Sam.c", [])

    assert task.capability_id == "workflow.code_fix"
    assert task.parameters["file"].endswith("Sam.c")
    assert task.missing_parameters == []
