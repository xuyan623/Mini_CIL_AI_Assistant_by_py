from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ai_assistant.storage import atomic_write_json, file_lock, safe_load_json
from ai_assistant.paths import PathManager, get_path_manager


DEFAULT_SYSTEM_PROMPT = (
    "你是 Alpine Linux 终端中的 AI 助手。"
    "回答要准确、直接、可执行。"
)


@dataclass
class HistorySettings:
    summary_threshold: int = 10
    keep_recent_rounds: int = 3
    recent_history_limit: int = 12
    recent_event_limit: int = 20
    event_output_preview_chars: int = 240
    max_events: int = 1000


class HistoryService:
    def __init__(self, path_manager: PathManager | None = None, settings: HistorySettings | None = None) -> None:
        self.path_manager = path_manager or get_path_manager()
        self.settings = settings or HistorySettings()
        self.lock_path = self.path_manager.state_dir / "history.lock"

    def _default_payload(self) -> dict[str, Any]:
        return {
            "version": 2,
            "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            "events": [],
        }

    def load_payload(self) -> dict[str, Any]:
        with file_lock(self.lock_path):
            payload = safe_load_json(self.path_manager.history_path, self._default_payload())
            if isinstance(payload, list):
                payload = {"version": 1, "messages": payload}
            if not isinstance(payload, dict):
                payload = self._default_payload()
            payload["version"] = 2
            payload.setdefault("messages", [])
            payload.setdefault("events", [])
            if not isinstance(payload["messages"], list):
                payload["messages"] = []
            if not isinstance(payload["events"], list):
                payload["events"] = []
            if not payload["messages"]:
                payload["messages"] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            atomic_write_json(self.path_manager.history_path, payload)
            return payload

    def save_payload(self, payload: dict[str, Any]) -> None:
        with file_lock(self.lock_path):
            atomic_write_json(self.path_manager.history_path, payload)

    def clear(self) -> None:
        self.save_payload(self._default_payload())

    def list_messages(self) -> list[dict[str, str]]:
        payload = self.load_payload()
        return payload["messages"]

    def list_events(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        return payload.get("events", [])

    def append_message(self, role: str, content: str) -> None:
        payload = self.load_payload()
        payload["messages"].append({"role": role, "content": content})
        self.save_payload(payload)

    def append_exchange(self, user_message: str, assistant_message: str) -> None:
        payload = self.load_payload()
        payload["messages"].append({"role": "user", "content": user_message})
        payload["messages"].append({"role": "assistant", "content": assistant_message})
        self.save_payload(payload)

    def append_event(
        self,
        event_type: str,
        input_text: str,
        output_text: str,
        ok: bool,
        exit_code: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = self.load_payload()
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "input": input_text,
            "output": output_text,
            "ok": bool(ok),
            "exit_code": int(exit_code),
        }
        if metadata:
            event["metadata"] = metadata

        events = payload.setdefault("events", [])
        events.append(event)
        if len(events) > self.settings.max_events:
            payload["events"] = events[-self.settings.max_events :]
        self.save_payload(payload)

    def append_command_record(
        self,
        command_line: str,
        output_text: str,
        ok: bool,
        exit_code: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.append_event(
            event_type="command",
            input_text=command_line,
            output_text=output_text,
            ok=ok,
            exit_code=exit_code,
            metadata=metadata,
        )

    def get_recent_non_system_messages(
        self,
        limit: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        source = payload if payload is not None else self.load_payload()
        messages = source.get("messages", [])
        recent_limit = self.settings.recent_history_limit if limit is None else limit
        non_system = [message for message in messages if message.get("role") != "system"]
        return non_system[-recent_limit:]

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[:max_chars]}..."

    def format_recent_events(
        self,
        limit: int | None = None,
        max_output_chars: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        source = payload if payload is not None else self.load_payload()
        events = source.get("events", [])
        if not events:
            return ""

        recent_limit = self.settings.recent_event_limit if limit is None else limit
        preview_limit = self.settings.event_output_preview_chars if max_output_chars is None else max_output_chars

        selected = events[-recent_limit:]
        lines = ["最近命令输入与输出（供参考）："]
        for index, event in enumerate(selected, 1):
            input_text = self._compact_text(str(event.get("input", "")), 120)
            output_text = self._compact_text(str(event.get("output", "")), preview_limit)
            lines.append(
                f"{index}. 输入: {input_text} | 输出: {output_text} | ok={event.get('ok', False)} code={event.get('exit_code', 1)}"
            )
        return "\n".join(lines)

    def build_messages_for_request(
        self,
        user_prompt: str,
        include_recent_history: bool = True,
        include_recent_events: bool = True,
        extra_system_messages: list[str] | None = None,
    ) -> list[dict[str, str]]:
        payload = self.load_payload()
        system_messages = [message for message in payload["messages"] if message.get("role") == "system"]
        if not system_messages:
            system_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        assembled_messages = [*system_messages]

        if include_recent_events:
            events_note = self.format_recent_events(payload=payload)
            if events_note:
                assembled_messages.append({"role": "system", "content": events_note})

        if extra_system_messages:
            for extra in extra_system_messages:
                if extra and extra.strip():
                    assembled_messages.append({"role": "system", "content": extra.strip()})

        if include_recent_history:
            assembled_messages.extend(self.get_recent_non_system_messages(payload=payload))

        assembled_messages.append({"role": "user", "content": user_prompt})
        return assembled_messages

    def trim_and_summarize(self, summarize_callback: callable | None = None) -> None:
        payload = self.load_payload()
        messages = payload["messages"]
        non_system = [message for message in messages if message.get("role") != "system"]
        if len(non_system) <= self.settings.summary_threshold:
            return

        if summarize_callback is None:
            return

        summary_text = summarize_callback(non_system)
        if not summary_text:
            return

        system_messages = [message for message in messages if message.get("role") == "system"]
        keep_count = self.settings.keep_recent_rounds * 2
        recent_messages = non_system[-keep_count:]
        payload["messages"] = [
            *system_messages[:1],
            {"role": "system", "content": f"历史总结：{summary_text}"},
            *recent_messages,
        ]
        self.save_payload(payload)
