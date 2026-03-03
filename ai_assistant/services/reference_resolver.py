from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ai_assistant.planner.types import EntityRecord, ReferenceResolutionResult


class ReferenceResolver:
    file_pronoun_tokens = (
        "这个文件",
        "该文件",
        "这个代码",
        "它",
        "刚刚找到的文件",
        "上一步文件",
        "this file",
        "that file",
        "the file",
        "it",
    )

    explicit_file_pattern = re.compile(r"([A-Za-z0-9_.:\-/\\]+\.[A-Za-z0-9_]+)")

    @classmethod
    def has_file_pronoun(cls, description: str) -> bool:
        lowered = description.lower()
        return any(token in lowered for token in cls.file_pronoun_tokens)

    @classmethod
    def has_explicit_file(cls, description: str) -> bool:
        return bool(cls.explicit_file_pattern.search(description))

    @staticmethod
    def _looks_like_windows_path(value: str) -> bool:
        return bool(re.match(r"^[A-Za-z]:\\", value))

    @staticmethod
    def _clean_path_text(value: str) -> str:
        return str(value).strip().strip("\"' ")

    def _normalize_entity(self, raw_entity: dict[str, Any]) -> EntityRecord | None:
        try:
            entity_id = str(raw_entity.get("entity_id", "")).strip()
            entity_type = str(raw_entity.get("entity_type", "")).strip()
            value = self._clean_path_text(str(raw_entity.get("value", "")))
            normalized_value = self._clean_path_text(str(raw_entity.get("normalized_value", value)))
            source_event_id = str(raw_entity.get("source_event_id", ""))
            trace_id = str(raw_entity.get("trace_id", ""))
            created_at = str(raw_entity.get("created_at", ""))
            confidence = float(raw_entity.get("confidence", 1.0) or 1.0)
            platform = str(raw_entity.get("platform", "alpine"))
            metadata = raw_entity.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not entity_id or not entity_type or not value:
                return None
            return EntityRecord(
                entity_id=entity_id,
                entity_type=entity_type,
                value=value,
                normalized_value=normalized_value or value,
                source_event_id=source_event_id,
                trace_id=trace_id,
                created_at=created_at,
                confidence=confidence,
                platform=platform,
                metadata=metadata,
            )
        except Exception:
            return None

    def _is_platform_mismatch(self, entity: EntityRecord) -> bool:
        if os.name == "nt":
            return False
        return self._looks_like_windows_path(entity.normalized_value) or self._looks_like_windows_path(entity.value)

    def _is_file_reachable(self, entity: EntityRecord) -> bool:
        candidate_text = entity.normalized_value or entity.value
        candidate_text = self._clean_path_text(candidate_text)
        if not candidate_text:
            return False
        if self._looks_like_windows_path(candidate_text) and os.name != "nt":
            return False
        candidate_path = Path(candidate_text).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()
        else:
            candidate_path = candidate_path.resolve()
        return candidate_path.exists() and candidate_path.is_file()

    def resolve_file_reference(
        self,
        *,
        description: str,
        entities: list[dict[str, Any]],
    ) -> ReferenceResolutionResult:
        if self.has_explicit_file(description):
            return ReferenceResolutionResult(
                status="resolved",
                selected_entity=None,
                candidates=[],
                reason="用户输入已包含显式文件路径",
            )
        if not self.has_file_pronoun(description):
            return ReferenceResolutionResult(
                status="resolved",
                selected_entity=None,
                candidates=[],
                reason="未检测到文件指代词",
            )

        normalized_entities: list[EntityRecord] = []
        for raw_entity in entities:
            entity = self._normalize_entity(raw_entity)
            if entity is not None and entity.entity_type == "file":
                normalized_entities.append(entity)
        if not normalized_entities:
            return ReferenceResolutionResult(
                status="missing",
                selected_entity=None,
                candidates=[],
                reason="历史中没有可用文件实体，无法解析“这个文件”",
            )

        sorted_entities = sorted(normalized_entities, key=lambda item: item.created_at, reverse=True)
        deduplicated: list[EntityRecord] = []
        seen: set[str] = set()
        for entity in sorted_entities:
            key = entity.normalized_value or entity.value
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(entity)

        viable: list[EntityRecord] = []
        rejected: list[EntityRecord] = []
        for entity in deduplicated:
            if self._is_platform_mismatch(entity):
                entity.metadata.setdefault("rejected_reason", "platform_mismatch")
                rejected.append(entity)
                continue
            if not self._is_file_reachable(entity):
                entity.metadata.setdefault("rejected_reason", "file_not_found")
                rejected.append(entity)
                continue
            viable.append(entity)

        if len(viable) == 1:
            return ReferenceResolutionResult(
                status="resolved",
                selected_entity=viable[0],
                candidates=viable,
                reason="命中唯一可用文件实体",
            )
        if len(viable) > 1:
            return ReferenceResolutionResult(
                status="ambiguous",
                selected_entity=None,
                candidates=viable,
                reason="匹配到多个可用文件实体，需用户确认",
            )

        return ReferenceResolutionResult(
            status="missing",
            selected_entity=None,
            candidates=rejected,
            reason="未找到当前环境可用的文件实体",
        )
