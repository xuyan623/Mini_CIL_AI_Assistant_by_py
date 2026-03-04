from __future__ import annotations

from types import SimpleNamespace

from ai_assistant.services.ai_gateway import AIGateway


def test_profile_attempt_order_and_callback_guard() -> None:
    gateway = AIGateway(ai_client=SimpleNamespace(config_service=None))
    assert gateway._profile_attempt_order(allow_fallback=True, fallback_profiles=["a", "", "b"]) == ["a", "b"]
    assert gateway._profile_attempt_order(allow_fallback=False, fallback_profiles=None) == [""]

    gateway._emit_attempt_event(None, {"event": "x"})
    gateway._emit_attempt_event(lambda payload: (_ for _ in ()).throw(RuntimeError("bad")), {"event": "x"})


def test_chat_and_summarize_error_branch() -> None:
    profile = SimpleNamespace(profile_id="p1")
    config = SimpleNamespace(
        get_active_profile=lambda: profile,
        list_profile_ids=lambda: ["p1"],
    )
    ai_client = SimpleNamespace(config_service=config, chat=lambda messages, **kwargs: (False, "boom"))  # noqa: ARG005
    gateway = AIGateway(ai_client=ai_client)

    response = gateway.chat([{"role": "user", "content": "hello"}], allow_fallback=False, attempt_callback=lambda payload: None)
    assert response.ok is False
    assert response.error_code == "request_failed"

    summary = gateway.summarize_messages([{"role": "user", "content": "x"}])
    assert summary == ""
