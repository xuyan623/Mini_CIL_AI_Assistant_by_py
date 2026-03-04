from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ai_assistant.services.ai_gateway import AIGateway


class _DummyConfigService:
    def __init__(self) -> None:
        self._active = "step"
        self._profiles = ["step", "deepseek", "qwen3.5"]

    def get_active_profile(self) -> SimpleNamespace:
        return SimpleNamespace(profile_id=self._active)

    def list_profile_ids(self) -> list[str]:
        return list(self._profiles)


class _DummyAIClient:
    def __init__(self) -> None:
        self.config_service = _DummyConfigService()
        self._responses: dict[str, list[tuple[bool, str]]] = {
            "step": [(False, "step timeout"), (False, "step timeout again")],
            "deepseek": [(False, "deepseek failed"), (False, "deepseek failed again")],
            "qwen3.5": [(True, '{"result":"ok-1"}'), (True, '{"result":"ok-2"}')],
        }

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[bool, str]:  # noqa: ARG002
        profile_id = kwargs.get("profile_override") or self.config_service.get_active_profile().profile_id
        queue = self._responses.get(profile_id, [(False, "unknown profile")])
        if len(queue) > 1:
            value = queue.pop(0)
        else:
            value = queue[0]
        return value


def test_gateway_prefers_last_successful_profile_in_follow_up_calls() -> None:
    gateway = AIGateway(ai_client=_DummyAIClient())
    payload = [{"role": "user", "content": "hello"}]

    first = gateway.chat(payload, allow_fallback=True, attempt_callback=lambda event: None)
    assert first.ok is True
    assert first.used_profile == "qwen3.5"
    assert [item["profile_id"] for item in first.attempts] == ["step", "deepseek", "qwen3.5"]
    assert first.attempts[0]["error_preview"] == "step timeout"
    assert first.attempts[1]["error_preview"] == "deepseek failed"

    second = gateway.chat(payload, allow_fallback=True, attempt_callback=lambda event: None)
    assert second.ok is True
    assert second.used_profile == "qwen3.5"
    assert [item["profile_id"] for item in second.attempts] == ["qwen3.5"]


def test_gateway_failure_attempts_contain_error_preview() -> None:
    client = _DummyAIClient()
    client._responses["qwen3.5"] = [(False, "qwen failed")]
    gateway = AIGateway(ai_client=client)
    payload = [{"role": "user", "content": "hello"}]

    result = gateway.chat(payload, allow_fallback=True, attempt_callback=lambda event: None)
    assert result.ok is False
    assert len(result.attempts) == 3
    assert all(str(item.get("error_preview", "")).strip() for item in result.attempts)
