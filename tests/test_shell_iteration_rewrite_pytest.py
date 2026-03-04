from __future__ import annotations

from pathlib import Path

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import TaskSpec


def _task(file_path: str) -> TaskSpec:
    return TaskSpec(
        raw_description="检查文件",
        normalized_description="检查文件",
        capability_id="code.check",
        parameters={"file": file_path},
    )


def test_rewrite_command_reuses_previous_line_count(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("line\n" * 23, encoding="utf-8")
    task = _task(str(target))

    transcript = [
        {
            "command": f'lines=$(wc -l < {target}); echo "$lines"',
            "stdout": "23\n",
            "stderr": "",
            "exit_code": 0,
        }
    ]
    facts = engine.extract_execution_facts(transcript, task)

    command = (
        f'ai code check {target} --start 1 --end '
        f'"$(lines=$(wc -l < {target}); [ "$lines" -gt 0 ] && echo "$lines" || echo 1)"'
    )
    rewrite = engine.rewrite_command_with_facts(command, facts, task)

    assert rewrite.rewritten is True
    assert "--end 23" in rewrite.command
    assert "wc -l <" not in rewrite.command
    assert "已复用上一步行数: 23" in rewrite.reason


def test_rewrite_command_skips_cache_after_mutation(tmp_path: Path) -> None:
    engine = PlanEngine()
    target = (tmp_path / "Sam.c").resolve()
    target.write_text("line\n" * 9, encoding="utf-8")
    task = _task(str(target))

    transcript = [
        {
            "command": f'lines=$(wc -l < {target}); echo "$lines"',
            "stdout": "9\n",
            "stderr": "",
            "exit_code": 0,
        },
        {
            "command": f"ai code optimize {target} --start 1 --end 9 --yes",
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
        },
    ]
    facts = engine.extract_execution_facts(transcript, task)
    assert str(target) not in facts.file_line_count
    assert facts.file_version_token[str(target)] == 1

    command = (
        f'ai code check {target} --start 1 --end '
        f'"$(lines=$(wc -l < {target}); [ "$lines" -gt 0 ] && echo "$lines" || echo 1)"'
    )
    rewrite = engine.rewrite_command_with_facts(command, facts, task)

    assert rewrite.rewritten is False
    assert rewrite.command == command
