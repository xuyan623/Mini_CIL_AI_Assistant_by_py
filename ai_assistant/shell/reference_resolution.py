from __future__ import annotations

from ai_assistant.planner.types import TaskSpec


class ShellReferenceResolution:
    def __init__(self, service: object) -> None:
        self.service = service

    def resolve(self, task: TaskSpec, trace_id: str) -> tuple[bool, TaskSpec, str]:
        return self.service._resolve_references(task, trace_id)  # type: ignore[attr-defined]

