from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    ok: bool
    message: str
    exit_code: int = 0
    data: dict[str, Any] | None = None


@dataclass
class ProfileConfig:
    profile_id: str
    name: str
    api_key: str
    api_url: str
    model: str
    stream: bool = False
