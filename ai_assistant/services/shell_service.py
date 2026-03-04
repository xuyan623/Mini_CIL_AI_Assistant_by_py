from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from ai_assistant.command_rules import build_cli_command_rules_prompt
from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.step_executor import StepExecutor
from ai_assistant.planner.task_interpreter import TaskInterpreter
from ai_assistant.planner.types import AIResponseEnvelope, EntityRecord, PlanDecision, TaskSpec
from ai_assistant.shell import (
    ShellCommandRewriter,
    ShellCommandValidator,
    ShellEventRecorder,
    ShellExecutionRuntime,
    ShellOrchestrator,
    ShellPlannerAdapter,
    ShellReferenceResolution,
    ShellStepFilter,
)
from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.ai_gateway import AIGateway
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryBatch, HistoryService
from ai_assistant.services.reference_resolver import ReferenceResolver
from ai_assistant.ui import OutputRenderer, RuntimeFeedback


@dataclass
class ShellSafetyReport:
    safe: bool
    warnings: list[str]


class ShellService:
    def __init__(
        self,
        ai_client: AIClient | None = None,
        history_service: HistoryService | None = None,
        context_service: ContextService | None = None,
    ) -> None:
        self.ai_client = ai_client or AIClient()
        self.ai_gateway = AIGateway(self.ai_client)
        self.history_service = history_service or HistoryService()
        self.context_service = context_service or ContextService()
        self.task_interpreter = TaskInterpreter()
        self.reference_resolver = ReferenceResolver()
        self.max_steps = 10
        self.step_timeout_seconds = 30
        self.plan_engine = PlanEngine(step_timeout_seconds=self.step_timeout_seconds)
        self.step_executor = StepExecutor(timeout_seconds=self.step_timeout_seconds)
        self.output_renderer = OutputRenderer()
        self.execution_runtime = ShellExecutionRuntime(self.output_renderer)
        self.command_rewriter = ShellCommandRewriter(self.plan_engine)
        self.step_filter = ShellStepFilter(self.plan_engine)
        self.command_validator = ShellCommandValidator()
        self.event_recorder = ShellEventRecorder(self)
        self.reference_resolution = ShellReferenceResolution(self)
        self.planner_adapter = ShellPlannerAdapter(self)
        self.orchestrator = ShellOrchestrator(self)
        self._active_trace_context: dict[str, Any] | None = None
        self._trace_profile_order: dict[str, list[str]] = {}
        self._trace_extra_system_messages: dict[str, list[str]] = {}
        self._trace_history_batches: dict[str, HistoryBatch] = {}
        self._runtime_feedback_callback = RuntimeFeedback(enabled=True).as_attempt_callback()
        self.dangerous_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\brm\s+-rf\s+/\b"), "会删除整个系统"),
            (re.compile(r"\brm\s+-rf\s+~\b"), "会删除用户主目录"),
            (re.compile(r"\bdd\s+if="), "可能覆盖磁盘数据"),
            (re.compile(r":\(\)\s*\{\s*:\|:&\s*;\s*\}"), "fork 炸弹"),
            (re.compile(r"\bmkfs\.[^\s]+"), "会格式化磁盘"),
            (re.compile(r"\bchmod\s+-R\s+777\s+/\b"), "会放开系统权限"),
            (re.compile(r"\bchown\s+-R\b"), "可能破坏权限"),
            (re.compile(r"\bwget\b.*\|\s*(bash|sh)\b"), "下载并执行远程脚本"),
            (re.compile(r"\bcurl\b.*\|\s*(bash|sh)\b"), "下载并执行远程脚本"),
        ]

    def _build_extra_system_messages(self, task_description: str) -> list[str]:
        messages = [build_cli_command_rules_prompt()]
        context_block = self.context_service.render_context_block(max_chars=8000)
        if context_block:
            messages.append(f"当前激活了代码上下文。若用户描述与项目相关，请先结合上下文生成命令：\n{context_block}")
        related_events = self.history_service.format_related_events(task_description)
        if related_events:
            messages.append(related_events)
        return messages

    def safety_check(self, command: str) -> ShellSafetyReport:
        warnings: list[str] = []
        for pattern, reason in self.dangerous_patterns:
            if pattern.search(command):
                warnings.append(f"⚠️ 危险模式：{reason}")
        if re.search(r"\bsudo\b", command):
            warnings.append("⚠️ 命令需要 root 权限")
        return ShellSafetyReport(safe=(len(warnings) == 0), warnings=warnings)

    def _record_event(
        self,
        event_type: str,
        input_text: str,
        output_text: str,
        ok: bool,
        exit_code: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata_dict = dict(metadata or {})
        trace_id = str(metadata_dict.get("trace_id", "")).strip()
        batch = self._trace_history_batches.get(trace_id) if trace_id else None
        return self.event_recorder.record_event(
            event_type=event_type,
            input_text=input_text,
            output_text=output_text,
            ok=ok,
            exit_code=exit_code,
            metadata=metadata_dict,
            batch=batch,
        )

    def _record_planner_trace(self, *, trace_id: str, stage: str, request_text: str, response: AIResponseEnvelope, metadata: dict[str, Any] | None = None) -> None:
        batch = self._trace_history_batches.get(trace_id)
        self.event_recorder.record_planner_trace(
            trace_id=trace_id,
            stage=stage,
            request_text=request_text,
            response=response,
            metadata=metadata,
            batch=batch,
        )

    def _record_interrupt(self, *, trace_id: str, stage: str, reason: str, step: int | None = None, command: str = "") -> None:
        batch = self._trace_history_batches.get(trace_id)
        self.event_recorder.record_interrupt(
            trace_id=trace_id,
            stage=stage,
            reason=reason,
            step=step,
            command=command,
            batch=batch,
        )

    def _confirm_with_prompt(self, prompt: str) -> tuple[bool, str]:
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                return False, "__eof__"
            if answer in {"y", "n"}:
                return answer == "y", answer
            if self._active_trace_context:
                self._record_event(
                    event_type="shell_control",
                    input_text=prompt,
                    output_text=answer,
                    ok=False,
                    exit_code=0,
                    metadata={
                        "module": "shell",
                        "phase": "control",
                        "trace_id": self._active_trace_context.get("trace_id", ""),
                        "stage": "input_validation",
                        "reason": "invalid_yes_no",
                        "step": self._active_trace_context.get("step"),
                    },
                )
            print("请输入 y 或 n", flush=True)

    @staticmethod
    def _emit_runtime_output(text: str) -> None:
        print(text, flush=True)

    def _load_json_object(self, raw_response: str) -> dict[str, Any] | None:
        return self.planner_adapter.load_json_object(raw_response)

    def _parse_planner_steps_json(self, raw_response: str) -> list[str]:
        return self.planner_adapter.parse_planner_steps_json(raw_response)

    def _parse_initial_steps(self, raw_response: str) -> list[str]:
        return self.planner_adapter.parse_initial_steps(raw_response)

    def _is_natural_language_line(self, command: str) -> bool:
        return self.command_validator.is_natural_language_line(command)

    def _contains_placeholder_token(self, command: str) -> bool:
        return self.command_validator.contains_placeholder_token(command)

    def _validate_shell_command(self, command: str) -> tuple[bool, str]:
        return self.command_validator.validate(command)

    def _repair_planner_output(self, *, trace_id: str, stage: str, task_description: str, raw_content: str, expect: str) -> AIResponseEnvelope:
        return self.planner_adapter.repair_planner_output(
            trace_id=trace_id,
            stage=stage,
            task_description=task_description,
            raw_content=raw_content,
            expect=expect,
        )

    def _build_initial_prompt(self, description: str) -> str:
        return self.planner_adapter.build_initial_prompt(description)

    def _candidate_to_dict(self, candidate: EntityRecord) -> dict[str, Any]:
        return self.reference_resolution.candidate_to_dict(candidate)

    def _build_reference_vote_prompt(self, description: str, candidates: list[EntityRecord]) -> str:
        return self.reference_resolution.build_reference_vote_prompt(description, candidates)

    def _vote_reference_with_ai(self, *, description: str, candidates: list[EntityRecord], trace_id: str) -> tuple[str, float, str]:
        return self.reference_resolution.vote_reference_with_ai(
            description=description,
            candidates=candidates,
            trace_id=trace_id,
        )

    def _normalize_file_value(self, raw_value: str) -> str:
        return self.reference_resolution.normalize_file_value(raw_value)

    def _resolve_references(self, task: TaskSpec, trace_id: str) -> tuple[bool, TaskSpec, str]:
        return self.reference_resolution.resolve(task, trace_id)

    def _build_replan_prompt(self, description: str, transcript: list[dict[str, Any]], suggested_steps: list[str]) -> str:
        return self.planner_adapter.build_replan_prompt(description, transcript, suggested_steps)

    def _compose_profile_order_for_trace(self, trace_id: str) -> list[str]:
        return self.planner_adapter.compose_profile_order_for_trace(trace_id)

    def _update_trace_profile_order(self, trace_id: str, response: AIResponseEnvelope) -> list[str]:
        return self.planner_adapter.update_trace_profile_order(trace_id, response)

    def _request_ai(self, *, prompt: str, trace_id: str, stage: str, task_description: str, max_tokens: int, temperature: float, timeout: int) -> AIResponseEnvelope:
        return self.planner_adapter.request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage=stage,
            task_description=task_description,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    def _format_attempt_failure_summary(self, response: AIResponseEnvelope) -> str:
        return self.planner_adapter.format_attempt_failure_summary(response)

    def _interpret_task_with_ai(self, base_task: TaskSpec, trace_id: str) -> TaskSpec | None:
        return self.planner_adapter.interpret_task_with_ai(base_task, trace_id)

    def _plan_from_description(self, description: str, trace_id: str) -> tuple[bool, TaskSpec, list[str], str]:
        return self.planner_adapter.plan_from_description(description, trace_id)

    def generate_initial_steps(self, description: str) -> tuple[bool, list[str], str]:
        trace_id = uuid.uuid4().hex
        ok, _task, commands, note = self._plan_from_description(description, trace_id)
        if not ok:
            return False, [], note
        return True, commands, note

    def generate_command(self, description: str) -> tuple[bool, str]:
        trace_id = uuid.uuid4().hex
        ok, task, commands, note = self._plan_from_description(description, trace_id)
        if not ok:
            return False, note
        lines: list[str] = []
        if task.retry_note:
            lines.append(f"📌 {task.retry_note}")
        if note:
            lines.append(f"📌 {note}")
        lines.extend(commands)
        return True, "\n".join(lines)

    def _plan_next_with_ai(self, description: str, transcript: list[dict[str, Any]], suggested_steps: list[str], trace_id: str) -> tuple[bool, PlanDecision]:
        return self.planner_adapter.plan_next(description, transcript, suggested_steps, trace_id)

    def _extract_paths_from_text(self, text: str) -> list[str]:
        return self.event_recorder.extract_paths_from_text(text)

    def _append_file_entity(self, *, value: str, source_event_id: str, trace_id: str, confidence: float, metadata: dict[str, Any] | None = None) -> None:
        batch = self._trace_history_batches.get(trace_id)
        self.event_recorder.append_file_entity(
            value=value,
            source_event_id=source_event_id,
            trace_id=trace_id,
            confidence=confidence,
            metadata=metadata,
            batch=batch,
        )

    def _extract_entities_from_step_output(self, *, command: str, stdout: str, stderr: str, source_event_id: str, trace_id: str) -> None:
        batch = self._trace_history_batches.get(trace_id)
        self.event_recorder.extract_entities_from_step_output(
            command=command,
            stdout=stdout,
            stderr=stderr,
            source_event_id=source_event_id,
            trace_id=trace_id,
            batch=batch,
        )

    def _run_workflow(self, description: str) -> tuple[bool, str]:
        return self.orchestrator.run_workflow(description)

    def run(self, description: str) -> tuple[bool, str]:
        return self.orchestrator.run(description)
