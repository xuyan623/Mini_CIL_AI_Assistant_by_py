from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid
from typing import Any

from ai_assistant.paths import PathManager, get_path_manager
from ai_assistant.state import JsonStateStore


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
    related_event_limit: int = 6
    event_output_preview_chars: int = 240
    max_events: int | None = None
    max_planner_traces: int | None = None
    max_entities: int = 2000


class HistoryService:
    def __init__(
        self,
        path_manager: PathManager | None = None,
        settings: HistorySettings | None = None,
        state_store: JsonStateStore | None = None,
    ) -> None:
        self.path_manager = path_manager or get_path_manager()
        self.settings = settings or HistorySettings()
        self.state_store = state_store or JsonStateStore()

    def _default_payload(self) -> dict[str, Any]:
        return {
            "version": 6,
            "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            "events": [],
            "planner_traces": [],
            "entities": [],
        }

    def _normalize_payload(self, raw_payload: Any) -> dict[str, Any]:
        payload = raw_payload
        if isinstance(payload, list):
            payload = {"version": 1, "messages": payload}
        if not isinstance(payload, dict):
            payload = self._default_payload()

        payload["version"] = 6
        payload.setdefault("messages", [])
        payload.setdefault("events", [])
        payload.setdefault("planner_traces", [])
        payload.setdefault("entities", [])
        if not isinstance(payload["messages"], list):
            payload["messages"] = []
        if not isinstance(payload["events"], list):
            payload["events"] = []
        if not isinstance(payload["planner_traces"], list):
            payload["planner_traces"] = []
        if not isinstance(payload["entities"], list):
            payload["entities"] = []
        if not payload["messages"]:
            payload["messages"] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        normalized_events: list[dict[str, Any]] = []
        for raw_event in payload["events"]:
            normalized_event = self._normalize_event(raw_event)
            if normalized_event is not None:
                normalized_events.append(normalized_event)
        payload["events"] = normalized_events

        normalized_traces: list[dict[str, Any]] = []
        for raw_trace in payload["planner_traces"]:
            normalized_trace = self._normalize_planner_trace(raw_trace)
            if normalized_trace is not None:
                normalized_traces.append(normalized_trace)
        payload["planner_traces"] = normalized_traces

        normalized_entities: list[dict[str, Any]] = []
        for raw_entity in payload["entities"]:
            normalized_entity = self._normalize_entity(raw_entity)
            if normalized_entity is not None:
                normalized_entities.append(normalized_entity)
        payload["entities"] = normalized_entities
        return payload

    @staticmethod
    def _normalize_event(raw_event: Any) -> dict[str, Any] | None:
        if not isinstance(raw_event, dict):
            return None

        metadata = raw_event.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata.setdefault("ui_block_id", "")
        metadata.setdefault("display_level", "")
        metadata.setdefault("decision_source", "")
        metadata.setdefault("io_stats", {})

        event_id = str(raw_event.get("event_id") or uuid.uuid4().hex)
        trace_id = str(raw_event.get("trace_id") or metadata.get("trace_id") or event_id)
        module = str(raw_event.get("module") or metadata.get("module") or "")
        phase = str(raw_event.get("phase") or metadata.get("phase") or metadata.get("stage") or "")
        parent_event_id = str(raw_event.get("parent_event_id") or metadata.get("parent_event_id") or "")

        normalized: dict[str, Any] = {
            "event_id": event_id,
            "trace_id": trace_id,
            "timestamp": raw_event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "event_type": str(raw_event.get("event_type", "unknown")),
            "module": module,
            "phase": phase,
            "parent_event_id": parent_event_id,
            "input": str(raw_event.get("input", "")),
            "output": str(raw_event.get("output", "")),
            "stdout": str(raw_event.get("stdout", "")),
            "stderr": str(raw_event.get("stderr", "")),
            "duration_ms": int(raw_event.get("duration_ms", metadata.get("duration_ms", 0)) or 0),
            "decision_source": str(raw_event.get("decision_source", metadata.get("decision_source", ""))),
            "ok": bool(raw_event.get("ok", False)),
            "exit_code": int(raw_event.get("exit_code", 1)),
            "metadata": metadata,
        }
        return normalized

    @staticmethod
    def _normalize_planner_trace(raw_trace: Any) -> dict[str, Any] | None:
        if not isinstance(raw_trace, dict):
            return None
        metadata = raw_trace.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "trace_id": str(raw_trace.get("trace_id") or uuid.uuid4().hex),
            "timestamp": raw_trace.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "stage": str(raw_trace.get("stage", "")),
            "request": str(raw_trace.get("request", "")),
            "response": str(raw_trace.get("response", "")),
            "ok": bool(raw_trace.get("ok", False)),
            "error_code": str(raw_trace.get("error_code", "")),
            "used_profile": str(raw_trace.get("used_profile", "")),
            "attempts": raw_trace.get("attempts", []),
            "metadata": metadata,
        }

    @staticmethod
    def _normalize_entity(raw_entity: Any) -> dict[str, Any] | None:
        if not isinstance(raw_entity, dict):
            return None
        metadata = raw_entity.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "entity_id": str(raw_entity.get("entity_id") or uuid.uuid4().hex),
            "entity_type": str(raw_entity.get("entity_type", "unknown")),
            "value": str(raw_entity.get("value", "")),
            "normalized_value": str(raw_entity.get("normalized_value", raw_entity.get("value", ""))),
            "source_event_id": str(raw_entity.get("source_event_id", "")),
            "trace_id": str(raw_entity.get("trace_id", "")),
            "created_at": str(raw_entity.get("created_at", datetime.now(timezone.utc).isoformat())),
            "confidence": float(raw_entity.get("confidence", 1.0) or 1.0),
            "platform": str(raw_entity.get("platform", "alpine")),
            "metadata": metadata,
        }

    def load_payload(self) -> dict[str, Any]:
        return self.state_store.read_json(
            self.path_manager.history_path,
            default_factory=self._default_payload,
            normalizer=self._normalize_payload,
        )

    def save_payload(self, payload: dict[str, Any]) -> None:
        normalized_payload = self._normalize_payload(payload)
        self.state_store.update_json(
            self.path_manager.history_path,
            updater=lambda _current: normalized_payload,
            default_factory=self._default_payload,
            normalizer=self._normalize_payload,
        )
        self.state_store.flush()

    def clear(self) -> None:
        self.save_payload(self._default_payload())

    def list_messages(self) -> list[dict[str, str]]:
        payload = self.load_payload()
        return payload["messages"]

    def list_events(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        return payload.get("events", [])

    def list_planner_traces(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        return payload.get("planner_traces", [])

    def list_entities(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            return []
        return entities

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
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        metadata_dict.setdefault("ui_block_id", "")
        metadata_dict.setdefault("display_level", "")
        metadata_dict.setdefault("decision_source", "")
        metadata_dict.setdefault("io_stats", self.state_store.get_io_stats())
        event_id = str(metadata_dict.get("event_id") or uuid.uuid4().hex)
        trace_id = str(metadata_dict.get("trace_id") or event_id)
        module = str(metadata_dict.get("module", ""))
        phase = str(metadata_dict.get("phase", metadata_dict.get("stage", "")))
        parent_event_id = str(metadata_dict.get("parent_event_id", ""))
        event: dict[str, Any] = {
            "event_id": event_id,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "module": module,
            "phase": phase,
            "parent_event_id": parent_event_id,
            "input": input_text,
            "output": output_text,
            "stdout": str(metadata_dict.get("stdout", "")),
            "stderr": str(metadata_dict.get("stderr", "")),
            "duration_ms": int(metadata_dict.get("duration_ms", 0) or 0),
            "decision_source": str(metadata_dict.get("decision_source", "")),
            "ok": bool(ok),
            "exit_code": int(exit_code),
            "metadata": metadata_dict,
        }

        events = payload.setdefault("events", [])
        events.append(event)
        if self.settings.max_events is not None and len(events) > self.settings.max_events:
            payload["events"] = events[-self.settings.max_events :]
        self.save_payload(payload)

    def append_planner_trace(
        self,
        *,
        trace_id: str,
        stage: str,
        request: str,
        response: str,
        ok: bool,
        error_code: str = "",
        used_profile: str = "",
        attempts: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = self.load_payload()
        traces = payload.setdefault("planner_traces", [])
        traces.append(
            {
                "trace_id": str(trace_id or uuid.uuid4().hex),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": str(stage),
                "request": str(request),
                "response": str(response),
                "ok": bool(ok),
                "error_code": str(error_code),
                "used_profile": str(used_profile),
                "attempts": attempts or [],
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
        if self.settings.max_planner_traces is not None and len(traces) > self.settings.max_planner_traces:
            payload["planner_traces"] = traces[-self.settings.max_planner_traces :]
        self.save_payload(payload)

    def append_resolution_trace(
        self,
        *,
        trace_id: str,
        request: str,
        response: str,
        ok: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        self.append_planner_trace(
            trace_id=trace_id,
            stage="reference_resolution",
            request=request,
            response=response,
            ok=ok,
            error_code="" if ok else "resolution_failed",
            used_profile="",
            attempts=[],
            metadata=metadata_dict,
        )
        self.append_event(
            event_type="resolution",
            input_text=request,
            output_text=response,
            ok=ok,
            exit_code=0 if ok else 1,
            metadata={
                "module": "shell",
                "phase": "plan",
                "trace_id": trace_id,
                "source": metadata_dict.get("source", ""),
                "status": metadata_dict.get("status", ""),
                "candidate_count": metadata_dict.get("candidate_count", 0),
                "decision_source": metadata_dict.get("source", ""),
                **metadata_dict,
            },
        )

    def append_entity(
        self,
        *,
        entity_type: str,
        value: str,
        normalized_value: str | None = None,
        source_event_id: str = "",
        trace_id: str = "",
        confidence: float = 1.0,
        platform: str = "alpine",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load_payload()
        entity = {
            "entity_id": uuid.uuid4().hex,
            "entity_type": str(entity_type),
            "value": str(value),
            "normalized_value": str(normalized_value if normalized_value is not None else value),
            "source_event_id": str(source_event_id),
            "trace_id": str(trace_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence": float(confidence),
            "platform": str(platform),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        entities = payload.setdefault("entities", [])
        entities.append(entity)
        if len(entities) > self.settings.max_entities:
            payload["entities"] = entities[-self.settings.max_entities :]
        self.save_payload(payload)
        return entity

    def find_entities(
        self,
        *,
        entity_type: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        entities = self.list_entities()
        filtered = entities
        if entity_type:
            filtered = [item for item in filtered if str(item.get("entity_type", "")) == entity_type]
        if keyword:
            lowered = keyword.lower()
            filtered = [
                item
                for item in filtered
                if lowered in str(item.get("value", "")).lower()
                or lowered in str(item.get("normalized_value", "")).lower()
            ]
        filtered = sorted(filtered, key=lambda item: str(item.get("created_at", "")), reverse=True)
        return filtered[:limit]

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

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9_./\\-]+|[\u4e00-\u9fff]+", text.lower())
        return {token for token in tokens if len(token) > 1}

    def format_related_events(self, query: str, limit: int | None = None, payload: dict[str, Any] | None = None) -> str:
        source = payload if payload is not None else self.load_payload()
        events = source.get("events", [])
        if not events:
            return ""

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return ""

        scored: list[tuple[int, dict[str, Any]]] = []
        for event in events:
            input_text = str(event.get("input", ""))
            output_text = str(event.get("output", ""))
            event_tokens = self._tokenize(f"{input_text}\n{output_text}")
            score = len(query_tokens.intersection(event_tokens))
            if score > 0:
                scored.append((score, event))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        max_items = limit if limit is not None else self.settings.related_event_limit
        selected = [item[1] for item in scored[:max_items]]
        lines = ["与当前任务相关的历史事件："]
        for index, event in enumerate(selected, 1):
            lines.append(
                f"{index}. 输入: {self._compact_text(str(event.get('input', '')), 120)} "
                f"| 输出: {self._compact_text(str(event.get('output', '')), 200)} "
                f"| ok={event.get('ok', False)} code={event.get('exit_code', 1)}"
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
