from __future__ import annotations

import re
import uuid
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
        batch: Any = None,
    ) -> str:
        if not hasattr(self.service, "history_service"):
            return self.service._record_event(  # type: ignore[attr-defined]
                event_type=event_type,
                input_text=input_text,
                output_text=output_text,
                ok=ok,
                exit_code=exit_code,
                metadata=metadata,
            )
        metadata_dict = dict(metadata or {})
        event_id = str(metadata_dict.get("event_id") or uuid.uuid4().hex)
        metadata_dict["event_id"] = event_id
        try:
            if batch is None:
                self.service.history_service.append_event(
                    event_type=event_type,
                    input_text=input_text,
                    output_text=output_text,
                    ok=ok,
                    exit_code=exit_code,
                    metadata=metadata_dict,
                )
            else:
                self.service.history_service.append_event_in_batch(
                    batch=batch,
                    event_type=event_type,
                    input_text=input_text,
                    output_text=output_text,
                    ok=ok,
                    exit_code=exit_code,
                    metadata=metadata_dict,
                )
        except Exception:
            return event_id
        return event_id

    def record_planner_trace(
        self,
        *,
        trace_id: str,
        stage: str,
        request_text: str,
        response: AIResponseEnvelope,
        metadata: dict[str, Any] | None = None,
        batch: Any = None,
    ) -> None:
        if not hasattr(self.service, "history_service"):
            self.service._record_planner_trace(  # type: ignore[attr-defined]
                trace_id=trace_id,
                stage=stage,
                request_text=request_text,
                response=response,
                metadata=metadata,
            )
            return
        try:
            if batch is None:
                self.service.history_service.append_planner_trace(
                    trace_id=trace_id,
                    stage=stage,
                    request=request_text,
                    response=response.content,
                    ok=response.ok,
                    error_code=response.error_code,
                    used_profile=response.used_profile,
                    attempts=response.attempts,
                    metadata=metadata,
                )
            else:
                self.service.history_service.append_planner_trace_in_batch(
                    batch=batch,
                    trace_id=trace_id,
                    stage=stage,
                    request=request_text,
                    response=response.content,
                    ok=response.ok,
                    error_code=response.error_code,
                    used_profile=response.used_profile,
                    attempts=response.attempts,
                    metadata=metadata,
                )
        except Exception:
            return

    def record_interrupt(
        self,
        *,
        trace_id: str,
        stage: str,
        reason: str,
        step: int | None = None,
        command: str = "",
        batch: Any = None,
    ) -> None:
        if not hasattr(self.service, "history_service"):
            self.service._record_interrupt(  # type: ignore[attr-defined]
                trace_id=trace_id,
                stage=stage,
                reason=reason,
                step=step,
                command=command,
            )
            return
        self.record_event(
            event_type="interrupt",
            input_text=command or stage,
            output_text=f"用户中断：{reason}",
            ok=False,
            exit_code=130,
            metadata={
                "module": "shell",
                "phase": "control",
                "trace_id": trace_id,
                "stage": stage,
                "reason": reason,
                "step": step if step is not None else 0,
                "command": command,
            },
            batch=batch,
        )

    @staticmethod
    def extract_paths_from_text(text: str) -> list[str]:
        candidates: set[str] = set()
        for raw_line in (text or "").splitlines():
            line = raw_line.strip().strip("\"'")
            if not line:
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            if line.startswith("└─ "):
                line = line[3:].strip()
            if line.startswith(".") or line.startswith("/") or re.match(r"^[A-Za-z]:\\", line):
                if "/" in line or "\\" in line:
                    candidates.add(line)
        return sorted(candidates)

    def append_file_entity(
        self,
        *,
        value: str,
        source_event_id: str,
        trace_id: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
        batch: Any = None,
    ) -> None:
        normalized = self.service._normalize_file_value(value)
        if not normalized:
            return
        platform = "windows" if re.match(r"^[A-Za-z]:\\", normalized) else "alpine"
        if batch is None:
            self.service.history_service.append_entity(
                entity_type="file",
                value=value,
                normalized_value=normalized,
                source_event_id=source_event_id,
                trace_id=trace_id,
                confidence=confidence,
                platform=platform,
                metadata=metadata or {},
            )
            return
        self.service.history_service.append_entity_in_batch(
            batch=batch,
            entity_type="file",
            value=value,
            normalized_value=normalized,
            source_event_id=source_event_id,
            trace_id=trace_id,
            confidence=confidence,
            platform=platform,
            metadata=metadata or {},
        )

    def extract_entities_from_step_output(
        self,
        *,
        command: str,
        stdout: str,
        stderr: str,
        source_event_id: str,
        trace_id: str,
        batch: Any = None,
    ) -> None:
        file_path_match = re.search(r"\btest\s+-f\s+(.+)$", command)
        if file_path_match:
            raw_file = file_path_match.group(1).strip().strip("\"'")
            self.service._append_file_entity(
                value=raw_file,
                source_event_id=source_event_id,
                trace_id=trace_id,
                confidence=0.95,
                metadata={"source": "test -f"},
            )

        backup_match = re.search(r"\bai\s+backup\s+create\s+([^\s]+)", command)
        if backup_match:
            raw_file = backup_match.group(1).strip().strip("\"'")
            self.service._append_file_entity(
                value=raw_file,
                source_event_id=source_event_id,
                trace_id=trace_id,
                confidence=0.95,
                metadata={"source": "ai backup create"},
            )

        code_match = re.search(r"\bai\s+code\s+\w+\s+([^\s]+)", command)
        if code_match:
            raw_file = code_match.group(1).strip().strip("\"'")
            self.service._append_file_entity(
                value=raw_file,
                source_event_id=source_event_id,
                trace_id=trace_id,
                confidence=0.9,
                metadata={"source": "ai code"},
            )

        if "find " in command:
            for path in self.extract_paths_from_text(stdout):
                self.service._append_file_entity(
                    value=path,
                    source_event_id=source_event_id,
                    trace_id=trace_id,
                    confidence=0.9,
                    metadata={"source": "find"},
                )

        if "ai file find " in command:
            for path in self.extract_paths_from_text(stdout):
                self.service._append_file_entity(
                    value=path,
                    source_event_id=source_event_id,
                    trace_id=trace_id,
                    confidence=0.85,
                    metadata={"source": "ai file find"},
                )
