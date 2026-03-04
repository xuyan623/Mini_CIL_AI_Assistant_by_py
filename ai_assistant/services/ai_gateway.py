from __future__ import annotations

from typing import Any

from ai_assistant.planner.types import AIResponseEnvelope
from ai_assistant.services.ai_client import AIClient


class AIGateway:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self.ai_client = ai_client or AIClient()
        self._preferred_profile_id = ""

    @staticmethod
    def _unique_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            token = str(item or "")
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _profile_attempt_order(self, allow_fallback: bool, fallback_profiles: list[str] | None = None) -> list[str]:
        if fallback_profiles:
            return self._unique_order([item for item in fallback_profiles if item])

        config_service = getattr(self.ai_client, "config_service", None)
        if config_service is None:
            return [""]

        active_profile = config_service.get_active_profile().profile_id
        if not allow_fallback:
            return [active_profile]

        ordered = [active_profile]
        for profile_id in config_service.list_profile_ids():
            if profile_id != active_profile:
                ordered.append(profile_id)
        preferred = str(self._preferred_profile_id or "")
        if preferred and preferred in ordered:
            ordered = [preferred, *[item for item in ordered if item != preferred]]
        return self._unique_order(ordered)

    @staticmethod
    def _emit_attempt_event(callback: callable | None, payload: dict[str, Any]) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            return

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream_override: bool | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        timeout: int = 60,
        print_stream: bool = False,
        attempt_callback: callable | None = None,
        allow_fallback: bool = True,
        fallback_profiles: list[str] | None = None,
    ) -> AIResponseEnvelope:
        attempts: list[dict[str, Any]] = []
        attempt_order = self._profile_attempt_order(allow_fallback=allow_fallback, fallback_profiles=fallback_profiles)

        self._emit_attempt_event(
            attempt_callback,
            {"event": "chat_start", "attempt_order": attempt_order, "print_stream": bool(print_stream)},
        )
        last_error_text = "❌ API 返回空内容"
        last_error_code = "empty_content"
        response_envelope = AIResponseEnvelope(
            ok=False,
            content=last_error_text,
            error_code=last_error_code,
            attempts=attempts,
            used_profile="",
        )

        try:
            for index, profile_id in enumerate(attempt_order):
                request_kwargs: dict[str, Any] = {
                    "stream_override": stream_override,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    "print_stream": print_stream,
                }
                if profile_id:
                    request_kwargs["profile_override"] = profile_id

                ok, content = self.ai_client.chat(messages, **request_kwargs)
                text = (content or "").strip()
                attempt: dict[str, Any] = {
                    "profile_id": profile_id,
                    "ok": bool(ok and text),
                    "error_code": "",
                    "content_preview": text[:240],
                    "error_preview": "",
                }

                if not ok:
                    attempt["error_code"] = "request_failed"
                    attempt["error_preview"] = str(content or "").strip()[:180]
                    attempts.append(attempt)
                    last_error_text = content
                    last_error_code = "request_failed"
                    next_profile = attempt_order[index + 1] if index + 1 < len(attempt_order) else ""
                    if next_profile and next_profile != profile_id:
                        self._emit_attempt_event(
                            attempt_callback,
                            {
                                "event": "fallback_switch",
                                "from_profile": profile_id,
                                "to_profile": next_profile,
                                "reason": "request_failed",
                            },
                        )
                    continue
                if not text:
                    attempt["error_code"] = "empty_content"
                    attempt["error_preview"] = "API 返回空内容"
                    attempts.append(attempt)
                    last_error_text = "❌ API 返回空内容"
                    last_error_code = "empty_content"
                    next_profile = attempt_order[index + 1] if index + 1 < len(attempt_order) else ""
                    if next_profile and next_profile != profile_id:
                        self._emit_attempt_event(
                            attempt_callback,
                            {
                                "event": "fallback_switch",
                                "from_profile": profile_id,
                                "to_profile": next_profile,
                                "reason": "empty_content",
                            },
                        )
                    continue

                attempts.append(attempt)
                response_envelope = AIResponseEnvelope(
                    ok=True,
                    content=content,
                    error_code="",
                    attempts=attempts,
                    used_profile=profile_id,
                )
                if profile_id:
                    self._preferred_profile_id = profile_id
                return response_envelope

            response_envelope = AIResponseEnvelope(
                ok=False,
                content=last_error_text,
                error_code=last_error_code,
                attempts=attempts,
                used_profile="",
            )
            return response_envelope
        finally:
            self._emit_attempt_event(
                attempt_callback,
                {
                    "event": "chat_end",
                    "ok": response_envelope.ok,
                    "used_profile": response_envelope.used_profile,
                    "error_code": response_envelope.error_code,
                },
            )

    def summarize_messages(self, messages: list[dict[str, str]]) -> str:
        prompt = [
            {
                "role": "user",
                "content": f"请将以下对话总结为 200 字以内要点：{messages}",
            }
        ]
        response = self.chat(
            prompt,
            stream_override=False,
            temperature=0.2,
            max_tokens=512,
            timeout=60,
            print_stream=False,
            attempt_callback=None,
            allow_fallback=True,
        )
        if not response.ok:
            return ""
        return response.content
