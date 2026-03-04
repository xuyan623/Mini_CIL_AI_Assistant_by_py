from __future__ import annotations

from typing import Any

from ai_assistant.planner.types import PlanDecision, TaskSpec


class ShellPlannerAdapter:
    def __init__(self, service: object) -> None:
        self.service = service

    def plan_from_description(self, description: str, trace_id: str) -> tuple[bool, TaskSpec, list[str], str]:
        return self.service._plan_from_description(description, trace_id)  # type: ignore[attr-defined]

    def plan_next(
        self,
        description: str,
        transcript: list[dict[str, Any]],
        suggested_steps: list[str],
        trace_id: str,
    ) -> tuple[bool, PlanDecision]:
        return self.service._plan_next_with_ai(description, transcript, suggested_steps, trace_id)  # type: ignore[attr-defined]

