from __future__ import annotations

import sys
import uuid
from typing import Any

from ai_assistant.planner.types import ShellExecutionResult
from ai_assistant.shell.workflow_context import ShellTraceContext


class ShellOrchestrator:
    def __init__(self, service: object) -> None:
        self.service = service

    def _commit_trace_batch(self, trace_id: str) -> None:
        batch = self.service._trace_history_batches.get(trace_id)
        if batch is None:
            return
        self.service.history_service.commit_batch(batch)

    def run_workflow(self, description: str) -> tuple[bool, str]:
        trace_id = uuid.uuid4().hex
        self.service._trace_profile_order.pop(trace_id, None)
        self.service._trace_extra_system_messages.pop(trace_id, None)
        self.service._trace_history_batches[trace_id] = self.service.history_service.begin_batch()
        ok, task, initial_steps, note = self.service.planner_adapter.plan_from_description(description, trace_id)
        self._commit_trace_batch(trace_id)
        if not ok:
            return False, note
        if not initial_steps:
            return False, "❌ 未生成有效步骤"

        trace_context = ShellTraceContext(
            trace_id=trace_id,
            description=description,
            task=task,
            note=note,
            transcript=[],
            suggested_steps=initial_steps[1:],
            profile_order=self.service._trace_profile_order.get(trace_id, []),
            extra_system_messages=self.service._trace_extra_system_messages.get(trace_id, []),
            history_batch=self.service._trace_history_batches.get(trace_id),
        )
        plan_text = self.service.execution_runtime.render_plan(
            retry_note=task.retry_note,
            note=note,
            steps=initial_steps,
        )

        self.service._record_event(
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
        self._commit_trace_batch(trace_id)

        stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if not (stdin_is_tty and stdout_is_tty):
            return True, f"{plan_text}\nℹ️ 当前终端非交互模式，仅生成步骤草案，未执行"

        self.service._active_trace_context = None
        current_command = ""
        try:
            self.service._emit_runtime_output(plan_text)
            start_prompt = "是否开始分步骤尝试执行？(y/n): "
            self.service._active_trace_context = {"trace_id": trace_id, "step": None}
            start_ok, start_answer = self.service._confirm_with_prompt(start_prompt)
            self.service._record_event(
                event_type="shell_control",
                input_text=start_prompt,
                output_text=start_answer,
                ok=start_ok,
                exit_code=0,
                metadata={"module": "shell", "phase": "control", "trace_id": trace_id, "stage": "start"},
            )
            self._commit_trace_batch(trace_id)
            if not start_ok:
                if start_answer == "__eof__":
                    self.service._record_interrupt(trace_id=trace_id, stage="start", reason="eof", step=None)
                    self._commit_trace_batch(trace_id)
                return True, "✅ 已取消执行"

            current_command = initial_steps[0]
            step_index = 0

            while step_index < self.service.max_steps:
                resolved_ok, resolved_command, resolved_error = self.service.plan_engine.resolve_placeholders(
                    current_command, trace_context.transcript, task
                )
                if not resolved_ok:
                    return False, f"❌ {resolved_error}"

                facts = self.service.command_rewriter.build_facts(trace_context.transcript, task)
                rewrite_result = self.service.command_rewriter.rewrite(resolved_command, facts, task)
                if rewrite_result.rewritten:
                    resolved_command = rewrite_result.command

                executable, executable_message = self.service._validate_shell_command(resolved_command)
                if not executable:
                    return False, executable_message

                valid, validation_message = self.service.plan_engine.validate_ai_code_command(resolved_command)
                if not valid:
                    return False, validation_message

                skip_step, skip_reason = self.service.step_filter.should_skip(resolved_command, facts, task)
                step_index += 1
                report = self.service.safety_check(resolved_command)
                step_lines = [self.service.execution_runtime.render_step_intro(step_index, resolved_command, bool(report.warnings))]
                if report.warnings:
                    step_lines.append("[WARN] 安全警告：")
                    step_lines.extend([f"  - {item}" for item in report.warnings])
                if rewrite_result.rewritten and rewrite_result.reason:
                    step_lines.append(f"[INFO] {rewrite_result.reason}")
                if skip_step and skip_reason:
                    step_lines.append(f"[INFO] 跳过冗余步骤：{skip_reason}")
                self.service._emit_runtime_output("\n".join(step_lines))

                if skip_step:
                    step_result = ShellExecutionResult(
                        command=resolved_command,
                        exit_code=0,
                        stdout="",
                        stderr="",
                        ok=True,
                    )
                    self.service._emit_runtime_output(
                        self.service.execution_runtime.render_step_result(
                            step_index=step_index,
                            command=resolved_command,
                            exit_code=0,
                            stdout_text="",
                            stderr_text="",
                            next_hint="已跳过冗余步骤",
                        )
                    )
                else:
                    confirm_prompt = (
                        f"是否执行第{step_index}步？(y/n): "
                        if report.safe
                        else f"第{step_index}步存在风险，仍要执行？(y/n): "
                    )
                    self.service._active_trace_context = {"trace_id": trace_id, "step": step_index}
                    confirmed, confirm_answer = self.service._confirm_with_prompt(confirm_prompt)
                    self.service._record_event(
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
                    self._commit_trace_batch(trace_id)
                    if not confirmed:
                        if confirm_answer == "__eof__":
                            self.service._record_interrupt(
                                trace_id=trace_id,
                                stage="step_confirm",
                                reason="eof",
                                step=step_index,
                                command=resolved_command,
                            )
                            self._commit_trace_batch(trace_id)
                        return True, f"✅ 第 {step_index} 步已取消，流程停止"

                    step_result = self.service.step_executor.execute(resolved_command)
                    self.service._emit_runtime_output(
                        self.service.execution_runtime.render_step_result(
                            step_index=step_index,
                            command=resolved_command,
                            exit_code=step_result.exit_code,
                            stdout_text=step_result.stdout.rstrip(),
                            stderr_text=step_result.stderr.rstrip(),
                        )
                    )

                event_output = (
                    f"exit_code={step_result.exit_code}\n"
                    f"stdout:\n{step_result.stdout}\n"
                    f"stderr:\n{step_result.stderr}"
                )
                step_event_id = self.service._record_event(
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
                        "rewrite_reason": rewrite_result.reason,
                        "skipped": bool(skip_step),
                        "skip_reason": skip_reason,
                        "profile_order_used": self.service._trace_profile_order.get(trace_id, []),
                    },
                )
                self.service._extract_entities_from_step_output(
                    command=resolved_command,
                    stdout=step_result.stdout,
                    stderr=step_result.stderr,
                    source_event_id=step_event_id,
                    trace_id=trace_id,
                )
                self._commit_trace_batch(trace_id)
                trace_context.transcript.append(
                    {
                        "step": step_index,
                        "command": resolved_command,
                        "exit_code": step_result.exit_code,
                        "stdout": step_result.stdout,
                        "stderr": step_result.stderr,
                        "ok": step_result.ok,
                        "skipped": bool(skip_step),
                        "rewrite_reason": rewrite_result.reason,
                        "skip_reason": skip_reason,
                    }
                )

                retry_decision = self.service.plan_engine.derive_retry_decision(trace_context.transcript)
                if retry_decision is not None:
                    decision = retry_decision
                else:
                    workflow_decision = self.service.plan_engine.derive_workflow_decision(task, trace_context.transcript)
                    if workflow_decision is not None:
                        decision = workflow_decision
                    else:
                        ai_ok, ai_decision = self.service.planner_adapter.plan_next(
                            task.normalized_description, trace_context.transcript, trace_context.suggested_steps, trace_id
                        )
                        self._commit_trace_batch(trace_id)
                        if ai_ok:
                            decision = ai_decision
                        else:
                            fallback = self.service.plan_engine.fallback_next_from_suggestions(
                                trace_context.suggested_steps, trace_context.transcript, task
                            )
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
                    self.service._emit_runtime_output(f"[INFO] {decision.message}")

                next_command = decision.command.strip()
                if not next_command:
                    retry_decision = self.service.plan_engine.derive_retry_decision(trace_context.transcript)
                    if retry_decision is not None:
                        next_command = retry_decision.command
                    else:
                        fallback = self.service.plan_engine.fallback_next_from_suggestions(
                            trace_context.suggested_steps, trace_context.transcript, task
                        )
                        if fallback is None:
                            if step_result.ok:
                                return True, "✅ 执行完成"
                            return False, "❌ 当前步骤失败，且未生成可执行的下一步命令"
                        if fallback.action != "next":
                            return False, f"❌ {fallback.message}"
                        next_command = fallback.command
                current_command = next_command

            return False, f"❌ 已达到最大步骤数（{self.service.max_steps}），流程停止"
        except KeyboardInterrupt:
            self.service._record_interrupt(
                trace_id=trace_id,
                stage="runtime",
                reason="ctrl_c",
                step=self.service._active_trace_context.get("step") if self.service._active_trace_context else None,
                command=current_command,
            )
            self._commit_trace_batch(trace_id)
            raise
        finally:
            self.service._active_trace_context = None
            self.service._trace_profile_order.pop(trace_id, None)
            self.service._trace_extra_system_messages.pop(trace_id, None)
            self._commit_trace_batch(trace_id)
            self.service._trace_history_batches.pop(trace_id, None)

    def run(self, description: str) -> tuple[bool, str]:
        return self.run_workflow(description)
