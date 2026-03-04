from __future__ import annotations

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import ExecutionFacts, TaskSpec


class ShellStepFilter:
    def __init__(self, plan_engine: PlanEngine) -> None:
        self.plan_engine = plan_engine

    def should_skip(self, command: str, facts: ExecutionFacts, task: TaskSpec) -> tuple[bool, str]:
        return self.plan_engine.should_skip_redundant_step(command, facts, task)

