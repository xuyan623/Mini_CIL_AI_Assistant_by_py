from __future__ import annotations

from types import SimpleNamespace

from ai_assistant.planner.types import AIResponseEnvelope
from ai_assistant.shell import ShellEventRecorder


def test_shell_event_recorder_delegation() -> None:
    captured = {"event": None, "trace": None, "interrupt": None}

    service = SimpleNamespace(
        _record_event=lambda **kwargs: captured.update({"event": kwargs}) or "eid",
        _record_planner_trace=lambda **kwargs: captured.update({"trace": kwargs}),
        _record_interrupt=lambda **kwargs: captured.update({"interrupt": kwargs}),
    )
    recorder = ShellEventRecorder(service)

    event_id = recorder.record_event("shell", "in", "out", True, 0, {"k": "v"})
    assert event_id == "eid"
    assert captured["event"]["event_type"] == "shell"

    recorder.record_planner_trace(
        trace_id="t1",
        stage="s1",
        request_text="req",
        response=AIResponseEnvelope(ok=True, content="ok"),
        metadata={"m": 1},
    )
    assert captured["trace"]["stage"] == "s1"

    recorder.record_interrupt(trace_id="t2", stage="run", reason="ctrl_c", step=1, command="echo")
    assert captured["interrupt"]["reason"] == "ctrl_c"

