from __future__ import annotations

import time
from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellService


class _DummyAIClient:
    def chat(self, messages, **kwargs):  # noqa: ANN001,ARG002
        return True, '{"summary":"x","steps":[{"command":"echo hello","purpose":"x"}]}'


def test_shell_planning_latency_budget(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    history = HistoryService(manager)
    context = ContextService(manager)
    service = ShellService(ai_client=_DummyAIClient(), history_service=history, context_service=context)  # type: ignore[arg-type]

    start = time.perf_counter()
    for _ in range(20):
        ok, text = service.generate_command("打印 hello")
        assert ok is True
        assert "echo hello" in text
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / 20) * 1000
    assert avg_ms < 200.0
