from __future__ import annotations

from pathlib import Path

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import TaskSpec


def _task(file_path: str) -> TaskSpec:
    return TaskSpec(
        raw_description="修复代码",
        normalized_description="修复代码",
        capability_id="workflow.code_fix",
        parameters={"file": file_path},
    )


def test_skip_duplicate_test_file_step(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("int main(void){return 0;}\n", encoding="utf-8")
    task = _task(str(target))

    transcript = [
        {
            "command": f"test -f {target}",
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
        }
    ]
    facts = engine.extract_execution_facts(transcript, task)
    skip, reason = engine.should_skip_redundant_step(f"test -f {target}", facts, task)

    assert skip is True
    assert "已确认存在" in reason


def test_skip_find_when_target_path_already_resolved(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("int main(void){return 0;}\n", encoding="utf-8")
    task = _task(str(target))

    transcript = [
        {
            "command": f"test -f {target}",
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
        }
    ]
    facts = engine.extract_execution_facts(transcript, task)
    skip, reason = engine.should_skip_redundant_step("find . -type f -name Sam.c", facts, task)

    assert skip is True
    assert str(target) in reason


def test_do_not_skip_side_effect_command(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("int main(void){return 0;}\n", encoding="utf-8")
    task = _task(str(target))
    facts = engine.extract_execution_facts([], task)

    skip, reason = engine.should_skip_redundant_step(
        f"ai code optimize {target} --start 1 --end 10 --yes",
        facts,
        task,
    )
    assert skip is False
    assert reason == ""
