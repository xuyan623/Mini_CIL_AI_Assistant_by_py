from __future__ import annotations

from dataclasses import dataclass

from ai_assistant.ui.view_models import ErrorBlock, OutputBlock, StepBlock


@dataclass
class RenderOptions:
    style: str = "rich_cards"
    density: str = "adaptive"
    tag_mode: str = "text"


class OutputRenderer:
    def __init__(self, options: RenderOptions | None = None) -> None:
        self.options = options or RenderOptions()

    @staticmethod
    def _status_tag(status: str) -> str:
        normalized = (status or "").strip().lower()
        if normalized in {"ok", "success", "done"}:
            return "[OK]"
        if normalized in {"warn", "warning"}:
            return "[WARN]"
        if normalized in {"error", "err", "failed", "abort"}:
            return "[ERR]"
        return "[INFO]"

    @staticmethod
    def _divider() -> str:
        return "-" * 64

    def render_block(self, block: OutputBlock) -> str:
        tag = self._status_tag(block.status)
        lines = [self._divider(), f"{tag} {block.title}", self._divider(), block.summary]
        details_required = self.options.density == "always_detailed" or tag in {"[ERR]", "[WARN]"}
        if details_required and block.details:
            lines.append("")
            lines.append("Details:")
            for item in block.details:
                lines.append(f"  - {item}")
        if block.actions:
            lines.append("")
            lines.append("Actions:")
            for item in block.actions:
                lines.append(f"  $ {item}")
        if block.trace_id:
            lines.append("")
            lines.append(f"trace_id: {block.trace_id}")
        return "\n".join(lines)

    def render_plan(self, plan_steps: list[str], note: str = "") -> str:
        summary = "已生成执行步骤" if plan_steps else "未生成执行步骤"
        if note:
            summary = f"{summary} | {note}"
        block = OutputBlock(
            block_id="plan",
            block_type="plan",
            title="Execution Plan",
            status="info",
            summary=summary,
            actions=plan_steps,
        )
        return self.render_block(block)

    def render_execution_step(self, step: StepBlock) -> str:
        tag = self._status_tag(step.status)
        lines = [self._divider(), f"{tag} Step {step.step_index}", self._divider(), f"$ {step.command}"]
        if step.duration_ms > 0:
            lines.append(f"duration: {step.duration_ms}ms")
        if step.stdout_preview:
            lines.append("stdout:")
            lines.append(step.stdout_preview.rstrip())
        if step.stderr_preview:
            lines.append("stderr:")
            lines.append(step.stderr_preview.rstrip())
        if step.next_hint:
            lines.append(f"next: {step.next_hint}")
        return "\n".join(lines)

    def render_error(self, error: ErrorBlock) -> str:
        block = OutputBlock(
            block_id="error",
            block_type="error",
            title=f"Error ({error.code})",
            status="error",
            summary=error.message,
            details=[error.suggestion] if error.suggestion else [],
            trace_id=error.trace_id,
        )
        return self.render_block(block)
