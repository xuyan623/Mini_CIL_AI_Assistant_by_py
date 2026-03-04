from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_assistant.ui.runtime_feedback import RuntimeFeedback


class _FakeStderr:
    def __init__(self) -> None:
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def test_render_loop_start_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_stderr = _FakeStderr()
    monkeypatch.setattr("sys.stderr", fake_stderr)
    feedback = RuntimeFeedback(enabled=True)

    def _sleep(_seconds: float) -> None:
        feedback._running = False

    monkeypatch.setattr("time.sleep", _sleep)
    feedback._running = True
    feedback._render_loop()
    assert any("正在思考" in text for text in fake_stderr.buffer)

    feedback.start_thinking()
    feedback.stop_thinking()
    assert any("\r" in text for text in fake_stderr.buffer)


def test_model_switch_and_gateway_events(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_stderr = _FakeStderr()
    monkeypatch.setattr("sys.stderr", fake_stderr)
    feedback = RuntimeFeedback(enabled=True)
    monkeypatch.setattr(feedback, "start_thinking", lambda: None)
    monkeypatch.setattr(feedback, "stop_thinking", lambda: None)

    feedback.emit_model_switch("a", "b", "empty_content")
    assert any("已切换到 'b'" in text for text in fake_stderr.buffer)

    feedback.handle_gateway_event({"event": "chat_start"})
    feedback.handle_gateway_event({"event": "chat_end"})
    feedback.handle_gateway_event({"event": "fallback_switch", "from_profile": "a", "to_profile": "b", "reason": "x"})
    callback = feedback.as_attempt_callback()
    callback({"event": "chat_end"})

    disabled = RuntimeFeedback(enabled=False)
    disabled.emit_model_switch("a", "b", "reason")
