from __future__ import annotations

import json
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from ai_assistant.command_rules import build_cli_command_rules_prompt
from ai_assistant.planner.plan_engine import PlanEngine
from ai_assistant.planner.step_executor import StepExecutor
from ai_assistant.planner.task_interpreter import TaskInterpreter
from ai_assistant.planner.types import AIResponseEnvelope, PlanDecision, TaskSpec
from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.ai_gateway import AIGateway
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService


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
        self.max_steps = 10
        self.step_timeout_seconds = 30
        self.plan_engine = PlanEngine(step_timeout_seconds=self.step_timeout_seconds)
        self.step_executor = StepExecutor(timeout_seconds=self.step_timeout_seconds)
        self._active_trace_context: dict[str, Any] | None = None
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
    ) -> None:
        try:
            self.history_service.append_event(
                event_type=event_type,
                input_text=input_text,
                output_text=output_text,
                ok=ok,
                exit_code=exit_code,
                metadata=metadata,
            )
        except Exception:
            return

    def _record_planner_trace(
        self,
        *,
        trace_id: str,
        stage: str,
        request_text: str,
        response: AIResponseEnvelope,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.history_service.append_planner_trace(
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

    @staticmethod
    def _load_json_object(raw_response: str) -> dict[str, Any] | None:
        cleaned = (raw_response or "").strip()
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _extract_commands(raw_response: str) -> list[str]:
        cleaned = (raw_response or "").strip()
        if not cleaned:
            return []
        commands: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```") or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].strip()
            line = re.sub(r"^\d+\.\s+", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            if line:
                commands.append(line)
        return commands

    @staticmethod
    def _extract_first_command(raw_response: str) -> str:
        commands = ShellService._extract_commands(raw_response)
        if not commands:
            return ""
        return commands[0]

    def _parse_initial_steps(self, raw_response: str) -> list[str]:
        parsed = self._load_json_object(raw_response)
        if parsed:
            if isinstance(parsed.get("steps"), list):
                steps: list[str] = []
                for item in parsed["steps"]:
                    if isinstance(item, str):
                        candidate = item.strip()
                    elif isinstance(item, dict):
                        candidate = str(item.get("command", "")).strip()
                    else:
                        candidate = ""
                    if candidate:
                        steps.append(candidate)
                if steps:
                    return steps
            if parsed.get("action") is not None:
                return []
        return self._extract_commands(raw_response)

    def _confirm_with_prompt(self, prompt: str) -> tuple[bool, str]:
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                return False, ""
            if answer in {"y", "n"}:
                return answer == "y", answer
            if self._active_trace_context:
                metadata = {
                    "module": "shell",
                    "phase": "control",
                    "trace_id": self._active_trace_context.get("trace_id", ""),
                    "stage": "input_validation",
                    "reason": "invalid_yes_no",
                    "step": self._active_trace_context.get("step"),
                }
                self._record_event(
                    event_type="shell_control",
                    input_text=prompt,
                    output_text=answer,
                    ok=False,
                    exit_code=0,
                    metadata=metadata,
                )
            print("请输入 y 或 n", flush=True)

    @staticmethod
    def _emit_runtime_output(text: str) -> None:
        print(text, flush=True)

    def _build_initial_prompt(self, description: str) -> str:
        return (
            "你是 Alpine Linux (/bin/sh + BusyBox) 终端命令规划助手。"
            "请根据用户目标生成首批可执行命令步骤。\n"
            "必须返回严格 JSON：\n"
            '{"summary":"<简短目标总结>","steps":[{"command":"<可直接执行命令>","purpose":"<该步目的>"}]}\n'
            "约束：\n"
            "1. command 必须能直接执行，不能使用 <FILE_PATH> 之类占位符。\n"
            "2. 若信息不足，先给发现信息的命令（find/sed/head/wc/test）。\n"
            "3. 步骤需可验证、可迭代，不超过 6 步。\n"
            "4. 只返回 JSON，不要解释，不要 Markdown。\n"
            f"用户描述：{description}"
        )

    def _build_replan_prompt(
        self,
        description: str,
        transcript: list[dict[str, Any]],
        suggested_steps: list[str],
    ) -> str:
        summary_lines = ["已执行步骤记录（按顺序）："]
        for index, record in enumerate(transcript[-8:], 1):
            summary_lines.append(
                f"{index}. cmd={record.get('command')} | code={record.get('exit_code')} | "
                f"stdout={str(record.get('stdout', ''))[:240]} | stderr={str(record.get('stderr', ''))[:240]}"
            )
        suggested_text = "\n".join(f"- {item}" for item in suggested_steps[:5]) if suggested_steps else "无"
        summary_block = "\n".join(summary_lines)
        return (
            "你是 Alpine Linux (/bin/sh + BusyBox) 的命令规划助手。"
            "请根据用户目标和执行结果决定下一步。\n"
            "必须返回严格 JSON：\n"
            '{"action":"next|done|need_input|abort","command":"<当 action=next 时必填>","message":"<简短说明>","confidence":0.0}\n'
            "规则：\n"
            "1. action=next 时 command 必须是可直接执行的单行命令，不能有占位符。\n"
            "2. 若上一步失败，优先给修复/诊断命令，不要重复失败命令。\n"
            "3. 信息不足或候选不唯一时返回 need_input，并在 message 指出缺失信息。\n"
            "4. 目标完成时返回 done，无法继续时返回 abort。\n"
            "5. 只返回 JSON。\n"
            f"用户目标：{description}\n"
            f"{summary_block}\n"
            f"可参考草案步骤：\n{suggested_text}"
        )

    def _request_ai(
        self,
        *,
        prompt: str,
        trace_id: str,
        stage: str,
        task_description: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
    ) -> AIResponseEnvelope:
        self.history_service.trim_and_summarize(self.ai_gateway.summarize_messages)
        messages = self.history_service.build_messages_for_request(
            user_prompt=prompt,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=self._build_extra_system_messages(task_description),
        )
        response = self.ai_gateway.chat(
            messages,
            stream_override=False,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            print_stream=False,
            allow_fallback=True,
        )
        self._record_planner_trace(
            trace_id=trace_id,
            stage=stage,
            request_text=prompt,
            response=response,
            metadata={"module": "shell", "phase": "plan", "task": task_description},
        )
        self._record_event(
            event_type="shell_planner",
            input_text=prompt,
            output_text=response.content,
            ok=response.ok,
            exit_code=0 if response.ok else 1,
            metadata={
                "module": "shell",
                "phase": "plan",
                "stage": stage,
                "trace_id": trace_id,
                "error_code": response.error_code,
                "used_profile": response.used_profile,
                "attempt_count": len(response.attempts),
            },
        )
        return response

    def _interpret_task_with_ai(self, base_task: TaskSpec, trace_id: str) -> TaskSpec | None:
        if not self.task_interpreter.should_try_ai_language_parse(base_task):
            return None
        prompt = self.task_interpreter.build_ai_parse_prompt(base_task.normalized_description)
        response = self._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage="interpret",
            task_description=base_task.normalized_description,
            max_tokens=512,
            temperature=0.1,
            timeout=60,
        )
        if not response.ok:
            return None
        return self.task_interpreter.parse_ai_task(
            raw_description=base_task.raw_description,
            normalized_description=base_task.normalized_description,
            retry_note=base_task.retry_note,
            raw_response=response.content,
        )

    def _plan_from_description(self, description: str, trace_id: str) -> tuple[bool, TaskSpec, list[str], str]:
        events = self.history_service.list_events()
        base_task = self.task_interpreter.interpret(description, events)
        if base_task.capability_id == "__invalid__" or base_task.note.startswith("❌"):
            return False, base_task, [], base_task.note

        parsed_task = self._interpret_task_with_ai(base_task, trace_id)
        task = parsed_task or base_task
        if task.note.startswith("❌"):
            return False, task, [], task.note

        if task.capability_id:
            local_ok, local_steps, local_note = self.plan_engine.build_initial_steps(task)
            if local_ok:
                commands = [step.command for step in local_steps]
                if not commands:
                    return False, task, [], "❌ 未生成有效命令"
                note = local_note or task.note
                return True, task, commands, note

        prompt = self._build_initial_prompt(task.normalized_description)
        response = self._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage="initial",
            task_description=task.normalized_description,
            max_tokens=640,
            temperature=0.2,
            timeout=60,
        )
        if not response.ok:
            if response.error_code == "empty_content":
                return False, task, [], "❌ 未生成有效命令"
            return False, task, [], response.content

        commands = self._parse_initial_steps(response.content)
        if not commands:
            return False, task, [], "❌ 未生成有效命令"

        for command in commands:
            valid, validation_message = self.plan_engine.validate_ai_code_command(command)
            if not valid:
                return False, task, [], validation_message
        return True, task, commands, task.note

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

    def _plan_next_with_ai(
        self,
        description: str,
        transcript: list[dict[str, Any]],
        suggested_steps: list[str],
        trace_id: str,
    ) -> tuple[bool, PlanDecision]:
        prompt = self._build_replan_prompt(description, transcript, suggested_steps)
        response = self._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage="replan",
            task_description=description,
            max_tokens=512,
            temperature=0.2,
            timeout=60,
        )
        if not response.ok:
            return False, PlanDecision(action="abort", message=response.content)

        parsed = self._load_json_object(response.content)
        if parsed:
            action = str(parsed.get("action", "")).strip().lower()
            command = str(parsed.get("command", "")).strip()
            message = str(parsed.get("message", "")).strip()
            if action in {"next", "done", "need_input", "abort"}:
                return True, PlanDecision(action=action, command=command, message=message)

        fallback_command = self._extract_first_command(response.content)
        if fallback_command:
            return True, PlanDecision(action="next", command=fallback_command, message="")
        return False, PlanDecision(action="abort", message="未能解析下一步决策")

    def run(self, description: str) -> tuple[bool, str]:
        trace_id = uuid.uuid4().hex
        ok, task, initial_steps, note = self._plan_from_description(description, trace_id)
        if not ok:
            return False, note
        if not initial_steps:
            return False, "❌ 未生成有效步骤"

        plan_lines = ["📋 生成步骤："]
        if task.retry_note:
            plan_lines.append(f"  📌 {task.retry_note}")
        if note:
            plan_lines.append(f"  📌 {note}")
        for step in initial_steps:
            plan_lines.append(f"  $ {step}")
        plan_text = "\n".join(plan_lines)

        self._record_event(
            event_type="shell_plan",
            input_text=task.normalized_description,
            output_text="\n".join(initial_steps),
            ok=True,
            exit_code=0,
            metadata={
                "module": "shell",
                "phase": "plan",
                "trace_id": trace_id,
                "note": note,
                "step_count": len(initial_steps),
                "requested_description": description,
                "retry_note": task.retry_note,
                "capability_id": task.capability_id or "",
                "source": task.source,
            },
        )

        if not sys.stdin.isatty():
            return True, f"{plan_text}\nℹ️ 当前终端非交互模式，仅生成步骤草案，未执行"

        self._active_trace_context = None
        try:
            self._emit_runtime_output(plan_text)
            start_prompt = "是否开始分步骤尝试执行？(y/n): "
            self._active_trace_context = {"trace_id": trace_id, "step": None}
            start_ok, start_answer = self._confirm_with_prompt(start_prompt)
            self._record_event(
                event_type="shell_control",
                input_text=start_prompt,
                output_text=start_answer,
                ok=start_ok,
                exit_code=0,
                metadata={"module": "shell", "phase": "control", "trace_id": trace_id, "stage": "start"},
            )
            if not start_ok:
                return True, "✅ 已取消执行"

            transcript: list[dict[str, Any]] = []
            suggested_steps = initial_steps[1:]
            current_command = initial_steps[0]
            step_index = 0

            while step_index < self.max_steps:
                resolved_ok, resolved_command, resolved_error = self.plan_engine.resolve_placeholders(
                    current_command, transcript, task
                )
                if not resolved_ok:
                    return False, f"❌ {resolved_error}"

                valid, validation_message = self.plan_engine.validate_ai_code_command(resolved_command)
                if not valid:
                    return False, validation_message

                step_index += 1
                report = self.safety_check(resolved_command)
                step_lines = [f"\n🔹 第 {step_index} 步", f"  $ {resolved_command}"]
                if report.warnings:
                    step_lines.append("  ⚠️ 安全警告：")
                    step_lines.extend([f"    - {item}" for item in report.warnings])
                self._emit_runtime_output("\n".join(step_lines))

                confirm_prompt = (
                    f"是否执行第{step_index}步？(y/n): "
                    if report.safe
                    else f"第{step_index}步存在风险，仍要执行？(y/n): "
                )
                self._active_trace_context = {"trace_id": trace_id, "step": step_index}
                confirmed, confirm_answer = self._confirm_with_prompt(confirm_prompt)
                self._record_event(
                    event_type="shell_control",
                    input_text=confirm_prompt,
                    output_text=confirm_answer,
                    ok=confirmed,
                    exit_code=0,
                    metadata={
                        "module": "shell",
                        "phase": "control",
                        "trace_id": trace_id,
                        "stage": "step",
                        "step": step_index,
                        "command": resolved_command,
                        "safe": report.safe,
                    },
                )
                if not confirmed:
                    return True, f"✅ 第 {step_index} 步已取消，流程停止"

                step_result = self.step_executor.execute(resolved_command)
                step_result_lines = [f"  🧾 退出码：{step_result.exit_code}"]
                if step_result.stdout:
                    step_result_lines.append("  📤 标准输出：")
                    step_result_lines.append(step_result.stdout.rstrip())
                if step_result.stderr:
                    step_result_lines.append("  ⚠️ 标准错误：")
                    step_result_lines.append(step_result.stderr.rstrip())
                self._emit_runtime_output("\n".join(step_result_lines))

                event_output = (
                    f"exit_code={step_result.exit_code}\n"
                    f"stdout:\n{step_result.stdout}\n"
                    f"stderr:\n{step_result.stderr}"
                )
                self._record_event(
                    event_type="shell_step",
                    input_text=resolved_command,
                    output_text=event_output,
                    ok=step_result.ok,
                    exit_code=step_result.exit_code,
                    metadata={
                        "module": "shell",
                        "phase": "execute",
                        "trace_id": trace_id,
                        "step": step_index,
                        "stdout": step_result.stdout,
                        "stderr": step_result.stderr,
                    },
                )
                transcript.append(
                    {
                        "step": step_index,
                        "command": resolved_command,
                        "exit_code": step_result.exit_code,
                        "stdout": step_result.stdout,
                        "stderr": step_result.stderr,
                        "ok": step_result.ok,
                    }
                )

                ai_ok, ai_decision = self._plan_next_with_ai(task.normalized_description, transcript, suggested_steps, trace_id)
                if ai_ok:
                    decision = ai_decision
                else:
                    retry_decision = self.plan_engine.derive_retry_decision(transcript)
                    if retry_decision is not None:
                        decision = retry_decision
                    else:
                        fallback = self.plan_engine.fallback_next_from_suggestions(suggested_steps, transcript, task)
                        if fallback is not None:
                            decision = fallback
                        elif step_result.ok:
                            return True, "✅ 执行完成"
                        else:
                            return False, f"❌ 上一步失败且无法生成下一步：{ai_decision.message}"

                if decision.action == "done":
                    done_message = decision.message or "执行完成"
                    return True, f"✅ {done_message}"
                if decision.action == "need_input":
                    message = decision.message or "缺少必要信息，请补充后重试"
                    return False, f"❌ {message}"
                if decision.action == "abort":
                    message = decision.message or "流程已终止"
                    return False, f"❌ {message}"

                if decision.message:
                    self._emit_runtime_output(f"  📌 {decision.message}")

                next_command = decision.command.strip()
                if not next_command:
                    retry_decision = self.plan_engine.derive_retry_decision(transcript)
                    if retry_decision is not None:
                        next_command = retry_decision.command
                    else:
                        fallback = self.plan_engine.fallback_next_from_suggestions(suggested_steps, transcript, task)
                        if fallback is None:
                            if step_result.ok:
                                return True, "✅ 执行完成"
                            return False, "❌ 当前步骤失败，且未生成可执行的下一步命令"
                        if fallback.action != "next":
                            return False, f"❌ {fallback.message}"
                        next_command = fallback.command
                current_command = next_command

            return False, f"❌ 已达到最大步骤数（{self.max_steps}），流程停止"
        finally:
            self._active_trace_context = None
