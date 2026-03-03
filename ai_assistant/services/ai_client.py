from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ai_assistant.services.config_service import ConfigService


class AIClient:
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

    def chat(
        self,
        messages: list[dict[str, str]],
        stream_override: bool | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        timeout: int = 60,
        print_stream: bool = False,
    ) -> tuple[bool, str]:
        profile = self.config_service.get_active_profile()
        stream = profile.stream if stream_override is None else stream_override

        if not profile.api_key:
            return False, "❌ 当前配置缺少 API Key，请先执行 config add 或切换配置"

        request_payload: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
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
                    content = data["choices"][0]["message"]["content"]
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

                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
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
