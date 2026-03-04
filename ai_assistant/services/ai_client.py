from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ai_assistant.services.config_service import ConfigService


class AIClient:
    THINKING_SYSTEM_INSTRUCTION = (
        "内部要求：请先进行充分的内部思考再回答。"
        "不要输出思考过程或推理链，只输出最终可读答案。"
    )

    def __init__(self, config_service: ConfigService | None = None) -> None:
        self.config_service = config_service or ConfigService()

    @staticmethod
    def clean_code_block(content: str) -> str:
        text = content.strip()
        fence_pattern = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(?P<body>[\s\S]*?)\n```$", re.MULTILINE)
        match = fence_pattern.match(text)
        if match:
            return match.group("body").rstrip()
        return text

    @staticmethod
    def _extract_text_fragment(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                fragment = AIClient._extract_text_fragment(item)
                if fragment:
                    chunks.append(fragment)
            return "".join(chunks)
        if isinstance(value, dict):
            for key in ("text", "content", "value", "output_text", "output"):
                fragment = AIClient._extract_text_fragment(value.get(key))
                if fragment:
                    return fragment
            return ""
        return str(value)

    @staticmethod
    def _extract_non_stream_content(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}

            if isinstance(message, dict):
                content = AIClient._extract_text_fragment(message.get("content"))
                if content:
                    return content

            text_choice = AIClient._extract_text_fragment(first_choice.get("text"))
            if text_choice:
                return text_choice

        for key in ("output_text", "text", "content", "output", "response"):
            extracted = AIClient._extract_text_fragment(data.get(key))
            if extracted:
                return extracted
        return ""

    @staticmethod
    def _extract_stream_chunk_content(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            if isinstance(first_choice, dict):
                delta_payload = first_choice.get("delta")
                extracted_delta = AIClient._extract_text_fragment(delta_payload)
                if extracted_delta:
                    return extracted_delta
                for key in ("content", "text", "output_text"):
                    extracted = AIClient._extract_text_fragment(first_choice.get(key))
                    if extracted:
                        return extracted

        for key in ("output_text", "text", "content", "output", "response"):
            extracted = AIClient._extract_text_fragment(data.get(key))
            if extracted:
                return extracted
        return ""

    @classmethod
    def _ensure_thinking_instruction(cls, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", ""))
            if role and content:
                normalized.append({"role": role, "content": content})

        for message in normalized:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            if "内部思考" in content and "不要输出思考过程" in content:
                return normalized

        return [{"role": "system", "content": cls.THINKING_SYSTEM_INSTRUCTION}, *normalized]

    def chat(
        self,
        messages: list[dict[str, str]],
        stream_override: bool | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        timeout: int = 60,
        print_stream: bool = False,
        profile_override: str | None = None,
    ) -> tuple[bool, str]:
        if profile_override:
            profile = self.config_service.get_profile(profile_override)
            if profile is None:
                return False, f"❌ 配置不存在：{profile_override}"
        else:
            profile = self.config_service.get_active_profile()
        stream = profile.stream if stream_override is None else stream_override

        if not profile.api_key:
            return False, "❌ 当前配置缺少 API Key，请先执行 config add 或切换配置"

        request_messages = self._ensure_thinking_instruction(messages)

        request_payload: dict[str, Any] = {
            "model": profile.model,
            "messages": request_messages,
            "temperature": temperature,
            "stream": bool(stream),
        }
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        payload_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            profile.api_url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {profile.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if not stream:
                    raw = response.read().decode("utf-8", errors="replace")
                    data = json.loads(raw)
                    if "error" in data:
                        return False, f"❌ API 错误：{data['error'].get('message', 'unknown')}"
                    content = self._extract_non_stream_content(data)
                    if not content.strip():
                        # 部分兼容层在非流式下返回空文本，回退尝试流式读取一次。
                        retry_ok, retry_content = self.chat(
                            messages,
                            stream_override=True,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            print_stream=False,
                            profile_override=profile_override,
                        )
                        if retry_ok and retry_content.strip():
                            return True, retry_content
                        return False, "❌ API 返回空内容"
                    return True, content

                chunks: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        payload = line[5:].strip()
                    else:
                        payload = line

                    if payload == "[DONE]":
                        break

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        # Some providers may stream plain text fragments.
                        if print_stream:
                            print(payload, end="", flush=True)
                        chunks.append(payload)
                        continue

                    if "error" in data:
                        return False, f"❌ API 错误：{data['error'].get('message', 'unknown')}"

                    delta = self._extract_stream_chunk_content(data)
                    if delta:
                        if print_stream:
                            print(delta, end="", flush=True)
                        chunks.append(delta)

                if print_stream and chunks:
                    print()
                return True, "".join(chunks)

        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return False, f"❌ HTTP 错误：{exc.code} {body}".strip()
        except urllib.error.URLError as exc:
            return False, f"❌ 网络错误：{exc.reason}"
        except Exception as exc:
            return False, f"❌ 请求失败：{exc}"

    def summarize_messages(self, messages: list[dict[str, str]]) -> str:
        prompt = [
            {
                "role": "user",
                "content": f"请将以下对话总结为 200 字以内要点：{json.dumps(messages, ensure_ascii=False)}",
            }
        ]
        ok, content = self.chat(prompt, stream_override=False, temperature=0.2, max_tokens=512, timeout=60)
        if not ok:
            return ""
        return content
