from __future__ import annotations


class ShellOrchestrator:
    def __init__(self, service: object) -> None:
        self.service = service

    def run(self, description: str) -> tuple[bool, str]:
        return self.service._run_workflow(description)  # type: ignore[attr-defined]

