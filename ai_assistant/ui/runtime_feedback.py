from __future__ import annotations

import sys
import threading
import time
from typing import Any


class RuntimeFeedback:
    _FRAMES = ("-", "\\", "|", "/")

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and sys.stderr.isatty())
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_line_length = 0

    def _render_loop(self) -> None:
        frame_index = 0
        while self._running:
            frame = self._FRAMES[frame_index % len(self._FRAMES)]
            line = f"\r[{frame}] 正在思考..."
            with self._lock:
                self._last_line_length = len(line) - 1
                sys.stderr.write(line)
                sys.stderr.flush()
            frame_index += 1
            time.sleep(0.12)

    def start_thinking(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop_thinking(self) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        with self._lock:
            clear_line = f"\r{' ' * self._last_line_length}\r"
            sys.stderr.write(clear_line)
            sys.stderr.flush()

    def emit_model_switch(self, from_profile: str, to_profile: str, reason: str) -> None:
        if not self.enabled:
            return
        self.stop_thinking()
        from_text = from_profile or "(active)"
        sys.stderr.write(
            f"[INFO] 模型配置 '{from_text}' 调用失败（{reason}），已切换到 '{to_profile}' 重试...\n"
        )
        sys.stderr.flush()
        self.start_thinking()

    def handle_gateway_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "chat_start":
            self.start_thinking()
            return
        if event_type == "chat_end":
            self.stop_thinking()
            return
        if event_type == "fallback_switch":
            self.emit_model_switch(
                from_profile=str(event.get("from_profile", "")),
                to_profile=str(event.get("to_profile", "")),
                reason=str(event.get("reason", "")),
            )

    def as_attempt_callback(self) -> callable:
        return self.handle_gateway_event

