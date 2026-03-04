from __future__ import annotations

from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.types import ExecutionFacts, StepRewriteResult, TaskSpec


class ShellCommandRewriter:
    def __init__(self, plan_engine: PlanEngine) -> None:
        self.plan_engine = plan_engine

    def build_facts(self, transcript: list[dict[str, object]], task: TaskSpec) -> ExecutionFacts:
        return self.plan_engine.extract_execution_facts(transcript, task)

    def rewrite(self, command: str, facts: ExecutionFacts, task: TaskSpec) -> StepRewriteResult:
        return self.plan_engine.rewrite_command_with_facts(command, facts, task)

