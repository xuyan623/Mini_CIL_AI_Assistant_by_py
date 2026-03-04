from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace
from typing import Any

import pytest

from ai_assistant.services.ai_client import AIClient


class _DummyResponse:
    def __init__(self, payload: str = "", stream_lines: list[str] | None = None) -> None:
        self._payload = payload
        self._stream_lines = stream_lines or []

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __iter__(self):
        for line in self._stream_lines:
            yield line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _config_service(stream: bool = False, api_key: str = "sk-test") -> Any:
    profile = SimpleNamespace(
        profile_id="p1",
        name="n",
        api_key=api_key,
        api_url="https://example.com/v1/chat/completions",
        model="m",
        stream=stream,
    )
    return SimpleNamespace(
        get_active_profile=lambda: profile,
        get_profile=lambda profile_id: profile if profile_id == "p1" else None,
    )


def test_chat_missing_api_key_returns_error() -> None:
    client = AIClient(config_service=_config_service(api_key=""))
    ok, message = client.chat([{"role": "user", "content": "hello"}])
    assert ok is False
    assert "缺少 API Key" in message


def test_chat_profile_override_not_found() -> None:
    client = AIClient(config_service=_config_service())
    ok, message = client.chat([{"role": "user", "content": "hello"}], profile_override="missing")
    assert ok is False
    assert "配置不存在" in message


def test_chat_non_stream_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "hello"}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=60: _DummyResponse(payload=json.dumps(payload)),  # noqa: ARG005
    )
    client = AIClient(config_service=_config_service(stream=False))
    ok, content = client.chat([{"role": "user", "content": "hello"}], stream_override=False)
    assert ok is True
    assert "hello" in content


def test_chat_stream_success(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"he"}}]}',
        'data: {"choices":[{"delta":{"content":"llo"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=60: _DummyResponse(stream_lines=lines),  # noqa: ARG005
    )
    client = AIClient(config_service=_config_service(stream=True))
    ok, content = client.chat([{"role": "user", "content": "hello"}], stream_override=True)
    assert ok is True
    assert content == "hello"


def test_chat_non_stream_empty_fallback_to_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _DummyResponse(payload=json.dumps({"choices": [{"message": {"content": ""}}]})),
        _DummyResponse(stream_lines=['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"]),
    ]

    def _urlopen(request, timeout=60):  # noqa: ARG001
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    client = AIClient(config_service=_config_service(stream=False))
    ok, content = client.chat([{"role": "user", "content": "hello"}], stream_override=False)
    assert ok is True
    assert content == "ok"


def test_chat_stream_plain_text_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["plain fragment", "data: [DONE]"]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=60: _DummyResponse(stream_lines=lines),  # noqa: ARG005
    )
    client = AIClient(config_service=_config_service(stream=True))
    ok, content = client.chat([{"role": "user", "content": "hello"}], stream_override=True, print_stream=False)
    assert ok is True
    assert "plain fragment" in content


def test_chat_http_error_and_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    http_error = urllib.error.HTTPError(
        url="https://example.com",
        code=500,
        msg="Server Error",
        hdrs=None,
        fp=io.BytesIO(b"body"),
    )

    def _raise_http(request, timeout=60):  # noqa: ARG001
        raise http_error

    monkeypatch.setattr("urllib.request.urlopen", _raise_http)
    client = AIClient(config_service=_config_service(stream=False))
    ok_http, message_http = client.chat([{"role": "user", "content": "hello"}], stream_override=False)
    assert ok_http is False
    assert "HTTP 错误" in message_http

    def _raise_url(request, timeout=60):  # noqa: ARG001
        raise urllib.error.URLError("dns")

    monkeypatch.setattr("urllib.request.urlopen", _raise_url)
    ok_url, message_url = client.chat([{"role": "user", "content": "hello"}], stream_override=False)
    assert ok_url is False
    assert "网络错误" in message_url


def test_ai_client_helper_extractors_and_clean_code() -> None:
    assert AIClient.clean_code_block("```py\nx=1\n```") == "x=1"
    assert AIClient._extract_text_fragment("x") == "x"
    assert AIClient._extract_text_fragment(1) == "1"
    assert AIClient._extract_text_fragment(["a", {"text": "b"}]) == "ab"
    assert AIClient._extract_text_fragment({"content": "c"}) == "c"

    non_stream = AIClient._extract_non_stream_content({"choices": [{"message": {"content": "hello"}}]})
    assert non_stream == "hello"
    non_stream_fallback = AIClient._extract_non_stream_content({"text": "t"})
    assert non_stream_fallback == "t"

    stream = AIClient._extract_stream_chunk_content({"choices": [{"delta": {"content": "x"}}]})
    assert stream == "x"
    stream_fallback = AIClient._extract_stream_chunk_content({"output_text": "y"})
    assert stream_fallback == "y"


def test_ensure_thinking_instruction_and_summarize(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "content": "hello"}]
    ensured = AIClient._ensure_thinking_instruction(messages)
    assert ensured[0]["role"] == "system"
    assert "内部要求" in ensured[0]["content"]

    existing = [{"role": "system", "content": "内部思考 不要输出思考过程"}, {"role": "user", "content": "x"}]
    ensured_existing = AIClient._ensure_thinking_instruction(existing)
    assert ensured_existing[0]["role"] == "system"
    assert len(ensured_existing) == len(existing)

    client = AIClient(config_service=_config_service())
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: (False, "x"))
    assert client.summarize_messages([{"role": "user", "content": "x"}]) == ""
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: (True, "summary"))
    assert client.summarize_messages([{"role": "user", "content": "x"}]) == "summary"
