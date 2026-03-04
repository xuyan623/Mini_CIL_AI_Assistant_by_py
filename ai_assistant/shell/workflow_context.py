from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_assistant.planner.types import TaskSpec


@dataclass
class ShellTraceContext:
    trace_id: str
    description: str
    task: TaskSpec
    note: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)
    suggested_steps: list[str] = field(default_factory=list)
    profile_order: list[str] = field(default_factory=list)
    extra_system_messages: list[str] = field(default_factory=list)
    history_batch: Any = None
