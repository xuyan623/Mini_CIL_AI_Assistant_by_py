from __future__ import annotations

from ai_assistant.planner.capabilities import build_capability_cli_reference


def build_cli_command_rules_prompt() -> str:
    return build_capability_cli_reference()
