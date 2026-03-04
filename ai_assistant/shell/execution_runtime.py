from __future__ import annotations

from ai_assistant.ui import OutputRenderer, StepBlock


class ShellExecutionRuntime:
    def __init__(self, output_renderer: OutputRenderer | None = None) -> None:
        self.output_renderer = output_renderer or OutputRenderer()

    def render_plan(self, *, retry_note: str, note: str, steps: list[str]) -> str:
        summary_parts: list[str] = []
        if retry_note:
            summary_parts.append(retry_note)
        if note:
            summary_parts.append(note)
        summary = " | ".join(summary_parts)
        return self.output_renderer.render_plan(steps, note=summary)

    def render_step_intro(self, step_index: int, command: str, has_warning: bool = False) -> str:
        status = "warn" if has_warning else "info"
        return self.output_renderer.render_execution_step(
            StepBlock(
                step_index=step_index,
                command=command,
                status=status,
            )
        )

    def render_step_result(
        self,
        *,
        step_index: int,
        command: str,
        exit_code: int,
        stdout_text: str,
        stderr_text: str,
        next_hint: str = "",
    ) -> str:
        status = "ok" if int(exit_code) == 0 else "error"
        return self.output_renderer.render_execution_step(
            StepBlock(
                step_index=step_index,
                command=command,
                status=status,
                stdout_preview=stdout_text,
                stderr_preview=stderr_text,
                next_hint=next_hint,
            )
        )

