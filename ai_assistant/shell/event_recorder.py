from __future__ import annotations

from typing import Any

from ai_assistant.planner.types import AIResponseEnvelope


class ShellEventRecorder:
    def __init__(self, service: object) -> None:
        self.service = service

    def record_event(
        self,
        event_type: str,
        input_text: str,
        output_text: str,
        ok: bool,
        exit_code: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.service._record_event(  # type: ignore[attr-defined]
            event_type=event_type,
            input_text=input_text,
            output_text=output_text,
            ok=ok,
            exit_code=exit_code,
            metadata=metadata,
        )

    def record_planner_trace(
        self,
        *,
        trace_id: str,
        stage: str,
        request_text: str,
        response: AIResponseEnvelope,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.service._record_planner_trace(  # type: ignore[attr-defined]
            trace_id=trace_id,
            stage=stage,
            request_text=request_text,
            response=response,
            metadata=metadata,
        )

    def record_interrupt(
        self,
        *,
        trace_id: str,
        stage: str,
        reason: str,
        step: int | None = None,
        command: str = "",
    ) -> None:
        self.service._record_interrupt(  # type: ignore[attr-defined]
            trace_id=trace_id,
            stage=stage,
            reason=reason,
            step=step,
            command=command,
        )

