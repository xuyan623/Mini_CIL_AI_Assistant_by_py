from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityParameter:
    name: str
    required: bool
    description: str
    example: str = ""


@dataclass(frozen=True)
class CommandCapability:
    capability_id: str
    module: str
    action: str
    summary: str
    command_template: str
    aliases: tuple[str, ...] = ()
    required_parameters: tuple[CapabilityParameter, ...] = ()
    interactive: bool = False
    risk_level: str = "normal"


@dataclass
class TaskSpec:
    raw_description: str
    normalized_description: str
    capability_id: str | None
    parameters: dict[str, str] = field(default_factory=dict)
    missing_parameters: list[str] = field(default_factory=list)
    note: str = ""
    source: str = "rule"
    retry_note: str = ""


@dataclass
class WorkflowState:
    target_file: str = ""
    resolution_status: str = "unresolved"
    candidates: list[str] = field(default_factory=list)
    resolved_file: str = ""
    last_step: str = ""


@dataclass
class PlanStep:
    command: str
    purpose: str = ""


@dataclass
class PlanDecision:
    action: str
    command: str = ""
    message: str = ""


@dataclass
class ShellExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    ok: bool


@dataclass
class AIResponseEnvelope:
    ok: bool
    content: str
    error_code: str = ""
    raw: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    used_profile: str = ""


@dataclass
class EntityRecord:
    entity_id: str
    entity_type: str
    value: str
    normalized_value: str
    source_event_id: str
    trace_id: str
    created_at: str
    confidence: float = 1.0
    platform: str = "alpine"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceResolutionResult:
    status: str
    selected_entity: EntityRecord | None = None
    candidates: list[EntityRecord] = field(default_factory=list)
    reason: str = ""


@dataclass
class PlannerEnvelope:
    summary: str
    steps: list[PlanStep] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class NextDecisionEnvelope:
    action: str
    command: str = ""
    message: str = ""
    confidence: float = 0.0
