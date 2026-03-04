from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai_assistant.planner.types import EntityRecord, TaskSpec


class ShellReferenceResolution:
    def __init__(self, service: object) -> None:
        self.service = service

    @staticmethod
    def normalize_file_value(raw_value: str) -> str:
        cleaned = str(raw_value).strip().strip("\"' ")
        if not cleaned:
            return ""
        if re.match(r"^[A-Za-z]:\\", cleaned):
            return cleaned
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return str(candidate)

    @staticmethod
    def candidate_to_dict(candidate: EntityRecord) -> dict[str, Any]:
        return {
            "entity_id": candidate.entity_id,
            "entity_type": candidate.entity_type,
            "value": candidate.value,
            "normalized_value": candidate.normalized_value,
            "created_at": candidate.created_at,
            "confidence": candidate.confidence,
            "metadata": candidate.metadata,
        }

    def build_reference_vote_prompt(self, description: str, candidates: list[EntityRecord]) -> str:
        candidate_payload = [self.candidate_to_dict(item) for item in candidates]
        return (
            "请从候选实体中选择“用户当前指代的文件”。\n"
            "仅返回 JSON，不要解释。\n"
            'JSON: {"selected_entity_id":"<候选ID或空字符串>","confidence":0.0,"reason":"<简短原因>"}\n'
            "规则：\n"
            "1. 只能从候选列表中选择；不确定请返回空字符串。\n"
            "2. confidence 取值 0~1。\n"
            f"用户描述：{description}\n"
            f"候选列表：{json.dumps(candidate_payload, ensure_ascii=False)}"
        )

    def vote_reference_with_ai(
        self,
        *,
        description: str,
        candidates: list[EntityRecord],
        trace_id: str,
    ) -> tuple[str, float, str]:
        if not candidates:
            return "", 0.0, "无候选可投票"
        prompt = self.build_reference_vote_prompt(description, candidates)
        response = self.service._request_ai(
            prompt=prompt,
            trace_id=trace_id,
            stage="reference_vote",
            task_description=description,
            max_tokens=300,
            temperature=0.0,
            timeout=45,
        )
        if not response.ok:
            return "", 0.0, response.content
        parsed = self.service._load_json_object(response.content)
        if not parsed:
            repaired = self.service._repair_planner_output(
                trace_id=trace_id,
                stage="reference_vote",
                task_description=description,
                raw_content=response.content,
                expect="reference_vote",
            )
            if not repaired.ok:
                return "", 0.0, repaired.content
            parsed = self.service._load_json_object(repaired.content) or {}
        selected_entity_id = str(parsed.get("selected_entity_id", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < 0:
            confidence = 0.0
        if confidence > 1:
            confidence = 1.0
        return selected_entity_id, confidence, reason

    def resolve(self, task: TaskSpec, trace_id: str) -> tuple[bool, TaskSpec, str]:
        if task.parameters.get("file"):
            return True, task, ""

        entities = self.service.history_service.list_entities()
        local_result = self.service.reference_resolver.resolve_file_reference(
            description=task.normalized_description,
            entities=entities,
        )
        local_status = local_result.status
        local_reason = local_result.reason

        selected_entity: EntityRecord | None = local_result.selected_entity
        if local_status == "ambiguous":
            candidate_lines = [f"  - {item.normalized_value or item.value}" for item in local_result.candidates[:8]]
            message = "❌ 检测到“这个文件/它”存在多个候选，请先明确目标路径：\n" + "\n".join(candidate_lines)
            self.service.history_service.append_resolution_trace(
                trace_id=trace_id,
                request=task.normalized_description,
                response=message,
                ok=False,
                metadata={"source": "local", "status": "ambiguous", "candidate_count": len(local_result.candidates)},
            )
            return False, task, message

        ai_selected_id = ""
        ai_confidence = 0.0
        ai_reason = ""
        if local_result.candidates:
            ai_selected_id, ai_confidence, ai_reason = self.service._vote_reference_with_ai(
                description=task.normalized_description,
                candidates=local_result.candidates,
                trace_id=trace_id,
            )

        if local_status == "resolved":
            if ai_selected_id:
                if not selected_entity or ai_selected_id != selected_entity.entity_id:
                    message = "❌ 指代解析冲突：本地与模型选择不一致，请明确文件路径"
                    self.service.history_service.append_resolution_trace(
                        trace_id=trace_id,
                        request=task.normalized_description,
                        response=message,
                        ok=False,
                        metadata={
                            "source": "vote",
                            "status": "ambiguous",
                            "local_entity": selected_entity.entity_id if selected_entity else "",
                            "model_entity": ai_selected_id,
                            "model_confidence": ai_confidence,
                            "model_reason": ai_reason,
                        },
                    )
                    return False, task, message
            if selected_entity:
                resolved_path = self.service._normalize_file_value(selected_entity.normalized_value or selected_entity.value)
                if resolved_path:
                    task.parameters["file"] = resolved_path
                    lowered = task.normalized_description.lower()
                    if task.capability_id is None and any(token in lowered for token in ("备份", "backup")):
                        task.capability_id = "backup.create"
                    if resolved_path not in task.normalized_description:
                        task.normalized_description = f"{task.normalized_description}（目标文件：{resolved_path}）"
                    note = f"已解析“这个文件”为：{resolved_path}"
                    task.note = note if not task.note else f"{task.note}；{note}"
            self.service.history_service.append_resolution_trace(
                trace_id=trace_id,
                request=task.normalized_description,
                response=task.parameters.get("file", ""),
                ok=True,
                metadata={
                    "source": "local",
                    "status": "resolved",
                    "reason": local_reason,
                    "entity_id": selected_entity.entity_id if selected_entity else "",
                    "model_entity": ai_selected_id,
                    "model_confidence": ai_confidence,
                    "model_reason": ai_reason,
                },
            )
            return True, task, ""

        if local_status == "missing":
            selected_from_ai: EntityRecord | None = None
            if ai_selected_id and ai_confidence >= 0.85:
                for candidate in local_result.candidates:
                    if candidate.entity_id == ai_selected_id and candidate.metadata.get("rejected_reason") != "platform_mismatch":
                        selected_from_ai = candidate
                        break
            if selected_from_ai:
                resolved_path = self.service._normalize_file_value(selected_from_ai.normalized_value or selected_from_ai.value)
                if resolved_path:
                    task.parameters["file"] = resolved_path
                    lowered = task.normalized_description.lower()
                    if task.capability_id is None and any(token in lowered for token in ("备份", "backup")):
                        task.capability_id = "backup.create"
                    if resolved_path not in task.normalized_description:
                        task.normalized_description = f"{task.normalized_description}（目标文件：{resolved_path}）"
                    note = f"已根据历史解析“这个文件”为：{resolved_path}"
                    task.note = note if not task.note else f"{task.note}；{note}"
                    self.service.history_service.append_resolution_trace(
                        trace_id=trace_id,
                        request=task.normalized_description,
                        response=resolved_path,
                        ok=True,
                        metadata={
                            "source": "model",
                            "status": "resolved",
                            "model_entity": ai_selected_id,
                            "model_confidence": ai_confidence,
                            "model_reason": ai_reason,
                        },
                    )
                    return True, task, ""

            message = "❌ 无法解析“这个文件”，请补充明确路径（例如：./mycode/Sam.c）"
            self.service.history_service.append_resolution_trace(
                trace_id=trace_id,
                request=task.normalized_description,
                response=message,
                ok=False,
                metadata={
                    "source": "local",
                    "status": "missing",
                    "reason": local_reason,
                    "model_entity": ai_selected_id,
                    "model_confidence": ai_confidence,
                    "model_reason": ai_reason,
                },
            )
            return False, task, message

        return True, task, ""
