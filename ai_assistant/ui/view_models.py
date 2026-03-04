from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutputBlock:
    block_id: str
    block_type: str
    title: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    trace_id: str = ""


@dataclass
class StepBlock:
    step_index: int
    command: str
    status: str
    stdout_preview: str = ""
    stderr_preview: str = ""
    duration_ms: int = 0
    next_hint: str = ""


@dataclass
class ErrorBlock:
    code: str
    message: str
    suggestion: str = ""
    trace_id: str = ""

