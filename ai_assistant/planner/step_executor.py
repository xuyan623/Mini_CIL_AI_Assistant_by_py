from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from ai_assistant.planner.types import ShellExecutionResult


class StepExecutor:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _ensure_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        return str(value)

    def execute(self, command: str) -> ShellExecutionResult:
        process_env = os.environ.copy()
        process_env.setdefault("AI_SUPPRESS_PLAINTEXT_WARN", "1")
        timeout_seconds = self.timeout_seconds
        if re.search(r"\bai\s+code\b", command):
            timeout_seconds = max(timeout_seconds, 90)
        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=process_env,
            )
            return ShellExecutionResult(
                command=command,
                exit_code=process.returncode,
                stdout=self._ensure_text(process.stdout),
                stderr=self._ensure_text(process.stderr),
                ok=(process.returncode == 0),
            )
        except subprocess.TimeoutExpired as exc:
            return ShellExecutionResult(
                command=command,
                exit_code=124,
                stdout=self._ensure_text(exc.stdout),
                stderr=f"命令超时（{timeout_seconds}s）",
                ok=False,
            )
        except Exception as exc:
            return ShellExecutionResult(command=command, exit_code=1, stdout="", stderr=str(exc), ok=False)
