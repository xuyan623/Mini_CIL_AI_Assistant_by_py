from __future__ import annotations

from ai_assistant.ui import ErrorBlock, OutputBlock, OutputRenderer, StepBlock


def test_render_block_with_details_and_actions() -> None:
    renderer = OutputRenderer()
    block = OutputBlock(
        block_id="b1",
        block_type="plan",
        title="Plan",
        status="warn",
        summary="summary",
        details=["d1", "d2"],
        actions=["echo a", "echo b"],
        trace_id="trace-1",
    )
    text = renderer.render_block(block)
    assert "[WARN]" in text
    assert "Details:" in text
    assert "Actions:" in text
    assert "trace_id: trace-1" in text


def test_render_plan_and_step_and_error() -> None:
    renderer = OutputRenderer()
    plan_text = renderer.render_plan(["echo hello"], note="note")
    assert "Execution Plan" in plan_text
    assert "$ echo hello" in plan_text

    step_text = renderer.render_execution_step(
        StepBlock(
            step_index=1,
            command="echo hello",
            status="ok",
            stdout_preview="hello",
            stderr_preview="",
            next_hint="done",
        )
    )
    assert "[OK]" in step_text
    assert "stdout:" in step_text
    assert "next: done" in step_text

    err_text = renderer.render_error(ErrorBlock(code="E1", message="failed", suggestion="retry"))
    assert "[ERR]" in err_text
    assert "retry" in err_text

