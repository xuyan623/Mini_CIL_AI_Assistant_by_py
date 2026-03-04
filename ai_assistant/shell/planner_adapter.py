from __future__ import annotations

import json
from typing import Any

from ai_assistant.planner.types import AIResponseEnvelope, PlanDecision, TaskSpec


class ShellPlannerAdapter:
    def __init__(self, service: object) -> None:
        self.service = service

    @staticmethod
    def load_json_object(raw_response: str) -> dict[str, Any] | None:
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

    def parse_planner_steps_json(self, raw_response: str) -> list[str]:
        parsed = self.load_json_object(raw_response)
        if not parsed:
            return []
        steps_raw = parsed.get("steps")
        if not isinstance(steps_raw, list):
            return []
        steps: list[str] = []
        for item in steps_raw:
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, dict):
                candidate = str(item.get("command", "")).strip()
            else:
                candidate = ""
            if candidate:
                steps.append(candidate)
        return steps

    def parse_initial_steps(self, raw_response: str) -> list[str]:
        return self.parse_planner_steps_json(raw_response)

    @staticmethod
    def build_initial_prompt(description: str) -> str:
        return (
            "你是 Alpine Linux (/bin/sh + BusyBox) 终端命令规划助手。"
            "请根据用户目标生成首批可执行命令步骤。\n"
            "必须返回严格 JSON：\n"
            '{"summary":"检查 Sam.c 是否有 bug","steps":[{"command":"test -f /home/mycode/Sam.c","purpose":"确认文件存在"},{"command":"ai code check /home/mycode/Sam.c --start 1 --end \\"$(wc -l < /home/mycode/Sam.c)\\"","purpose":"执行检查"}]}\n'
            "约束：\n"
            "1. command 必须能直接执行，禁止输出任何 <...> 占位符。\n"
            "2. 若信息不足，先给发现信息的命令（find/sed/head/wc/test）。\n"
            "3. 步骤需可验证、可迭代，不超过 6 步。\n"
            "4. 只返回 JSON，不要解释，不要 Markdown。\n"
            f"用户描述：{description}"
        )

    @staticmethod
    def build_replan_prompt(
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

    def compose_profile_order_for_trace(self, trace_id: str) -> list[str]:
        trace_order = self.service._trace_profile_order.get(trace_id, [])
        if trace_order:
            return [item for item in trace_order if item]
        try:
            current_profile = self.service.ai_gateway.ai_client.config_service.get_active_profile().profile_id
            profile_ids = self.service.ai_gateway.ai_client.config_service.list_profile_ids()
        except Exception:
            return []
        ordered = [current_profile, *[item for item in profile_ids if item != current_profile]]
        deduped: list[str] = []
        for item in ordered:
            if item in deduped:
                continue
            deduped.append(item)
        return deduped

    def update_trace_profile_order(self, trace_id: str, response: AIResponseEnvelope) -> list[str]:
        try:
            profile_ids = self.service.ai_gateway.ai_client.config_service.list_profile_ids()
        except Exception:
            profile_ids = []
        attempted = [str(item.get("profile_id", "")).strip() for item in response.attempts if str(item.get("profile_id", "")).strip()]
        preferred = str(response.used_profile or "")
        ordered: list[str] = []
        if preferred:
            ordered.append(preferred)
        ordered.extend(attempted)
        ordered.extend([item for item in profile_ids if item not in ordered])
        deduped: list[str] = []
        for item in ordered:
            if item in deduped:
                continue
            deduped.append(item)
        if deduped:
            self.service._trace_profile_order[trace_id] = deduped
        return deduped

    def request_ai(
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
        self.service.history_service.trim_and_summarize(self.service.ai_gateway.summarize_messages)
        profile_order = self.compose_profile_order_for_trace(trace_id)
        cached_system_messages = self.service._trace_extra_system_messages.get(trace_id)
        if cached_system_messages is None:
            cached_system_messages = self.service._build_extra_system_messages(task_description)
            self.service._trace_extra_system_messages[trace_id] = list(cached_system_messages)
        messages = self.service.history_service.build_messages_for_request(
            user_prompt=prompt,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=cached_system_messages,
        )
        response = self.service.ai_gateway.chat(
            messages,
            stream_override=False,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            print_stream=False,
            attempt_callback=self.service._runtime_feedback_callback,
            allow_fallback=True,
            fallback_profiles=profile_order or None,
        )
        profile_order_used = self.update_trace_profile_order(trace_id, response)
        history_batch = self.service._trace_history_batches.get(trace_id)
        self.service.event_recorder.record_planner_trace(
            trace_id=trace_id,
            stage=stage,
            request_text=prompt,
            response=response,
            metadata={
                "module": "shell",
                "phase": "plan",
                "task": task_description,
                "profile_order_used": profile_order_used,
            },
            batch=history_batch,
        )
        self.service.event_recorder.record_event(
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
                "attempts": response.attempts,
                "profile_order_used": profile_order_used,
            },
            batch=history_batch,
        )
        return response

    @staticmethod
    def format_attempt_failure_summary(response: AIResponseEnvelope) -> str:
        if not response.attempts:
            return ""
        lines: list[str] = []
        for attempt in response.attempts:
            if bool(attempt.get("ok")):
                continue
            profile_id = str(attempt.get("profile_id", "")).strip() or "(active)"
            error_code = str(attempt.get("error_code", "")).strip() or "failed"
            error_preview = str(attempt.get("error_preview", "")).strip()
            if error_preview:
                lines.append(f"- {profile_id}: {error_code} | {error_preview}")
            else:
                lines.append(f"- {profile_id}: {error_code}")
        if not lines:
            return ""
        return "模型调用尝试摘要：\n" + "\n".join(lines)

    def interpret_task_with_ai(self, base_task: TaskSpec, trace_id: str) -> TaskSpec | None:
        if not self.service.task_interpreter.should_try_ai_language_parse(base_task):
            return None
        prompt = self.service.task_interpreter.build_ai_parse_prompt(base_task.normalized_description)
        response = self.service._request_ai(
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
        return self.service.task_interpreter.parse_ai_task(
            raw_description=base_task.raw_description,
            normalized_description=base_task.normalized_description,
            retry_note=base_task.retry_note,
            raw_response=response.content,
        )

    def repair_planner_output(
        self,
        *,
        trace_id: str,
        stage: str,
        task_description: str,
        raw_content: str,
        expect: str,
    ) -> AIResponseEnvelope:
        if expect == "initial":
            expected_schema = '{"summary":"...","steps":[{"command":"...","purpose":"..."}]}'
        elif expect == "reference_vote":
            expected_schema = '{"selected_entity_id":"<候选ID或空字符串>","confidence":0.0,"reason":"..."}'
        else:
            expected_schema = '{"action":"next|done|need_input|abort","command":"...","message":"...","confidence":0.0}'
        prompt = (
            "你刚才的输出不符合 JSON 协议。"
            "请将下列原始输出修复为严格 JSON，且不要包含任何解释或 Markdown。\n"
            f"目标 schema：{expected_schema}\n"
            f"原始输出：\n{raw_content}"
        )
        return self.service._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage=f"{stage}_repair",
            task_description=task_description,
            max_tokens=512,
            temperature=0.0,
            timeout=45,
        )

    def plan_from_description(self, description: str, trace_id: str) -> tuple[bool, TaskSpec, list[str], str]:
        events = self.service.history_service.list_events()
        base_task = self.service.task_interpreter.interpret(description, events)
        if base_task.capability_id == "__invalid__" or base_task.note.startswith("❌"):
            return False, base_task, [], base_task.note

        resolved_ok, resolved_task, resolved_message = self.service.reference_resolution.resolve(base_task, trace_id)
        if not resolved_ok:
            return False, resolved_task, [], resolved_message
        base_task = resolved_task

        parsed_task = self.service._interpret_task_with_ai(base_task, trace_id)
        task = parsed_task or base_task
        if task.note.startswith("❌"):
            return False, task, [], task.note

        if task.capability_id:
            local_ok, local_steps, local_note = self.service.plan_engine.build_initial_steps(task)
            if local_ok:
                commands = [step.command for step in local_steps]
                if not commands:
                    return False, task, [], "❌ 未生成有效命令"
                note = local_note or task.note
                return True, task, commands, note

        prompt = self.service._build_initial_prompt(task.normalized_description)
        response = self.service._request_ai(
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
            summary = self.service._format_attempt_failure_summary(response)
            message = response.content
            if summary:
                message = f"{message}\n{summary}".strip()
            return False, task, [], message

        commands = self.service._parse_initial_steps(response.content)
        repair_attempted = False
        if not commands:
            repaired = self.service._repair_planner_output(
                trace_id=trace_id,
                stage="initial",
                task_description=task.normalized_description,
                raw_content=response.content,
                expect="initial",
            )
            repair_attempted = True
            if repaired.ok:
                commands = self.service._parse_initial_steps(repaired.content)
        if not commands:
            return False, task, [], "❌ 规划输出不合规：未返回 JSON steps"

        def validate_commands(command_list: list[str]) -> tuple[bool, str]:
            for command in command_list:
                executable, executable_message = self.service._validate_shell_command(command)
                if not executable:
                    return False, executable_message
                valid, validation_message = self.service.plan_engine.validate_ai_code_command(command)
                if not valid:
                    return False, validation_message
            return True, ""

        valid_commands, validation_error = validate_commands(commands)
        if not valid_commands and not repair_attempted:
            repaired = self.service._repair_planner_output(
                trace_id=trace_id,
                stage="initial",
                task_description=task.normalized_description,
                raw_content=response.content,
                expect="initial",
            )
            repair_attempted = True
            if repaired.ok:
                repaired_commands = self.service._parse_initial_steps(repaired.content)
                if repaired_commands:
                    commands = repaired_commands
                    valid_commands, validation_error = validate_commands(commands)
        if not valid_commands:
            return False, task, [], validation_error
        return True, task, commands, task.note

    def plan_next(
        self,
        description: str,
        transcript: list[dict[str, Any]],
        suggested_steps: list[str],
        trace_id: str,
    ) -> tuple[bool, PlanDecision]:
        prompt = self.service._build_replan_prompt(description, transcript, suggested_steps)
        response = self.service._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage="replan",
            task_description=description,
            max_tokens=512,
            temperature=0.2,
            timeout=60,
        )
        if not response.ok:
            summary = self.service._format_attempt_failure_summary(response)
            message = response.content
            if summary:
                message = f"{message}\n{summary}".strip()
            return False, PlanDecision(action="abort", message=message)

        parsed = self.service._load_json_object(response.content)
        if not parsed:
            repaired = self.service._repair_planner_output(
                trace_id=trace_id,
                stage="replan",
                task_description=description,
                raw_content=response.content,
                expect="decision",
            )
            if repaired.ok:
                parsed = self.service._load_json_object(repaired.content)
            else:
                return False, PlanDecision(action="abort", message=repaired.content)
        if parsed:
            action = str(parsed.get("action", "")).strip().lower()
            command = str(parsed.get("command", "")).strip()
            message = str(parsed.get("message", "")).strip()
            if action in {"next", "done", "need_input", "abort"}:
                if action == "next":
                    executable, executable_message = self.service._validate_shell_command(command)
                    if not executable:
                        return False, PlanDecision(action="abort", message=executable_message)
                return True, PlanDecision(action=action, command=command, message=message)

        return False, PlanDecision(action="abort", message="未能解析下一步决策 JSON")
