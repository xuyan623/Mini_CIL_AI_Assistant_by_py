from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai_assistant.planner.capabilities import get_capability, list_capabilities
from ai_assistant.planner.types import TaskSpec


class TaskInterpreter:
    retry_tokens = {"重试", "再试", "再试一次", "retry", "try again", "继续"}

    @staticmethod
    def _extract_file_candidate(description: str) -> str | None:
        match = re.search(r"([A-Za-z0-9_.:\-/\\]+\.[A-Za-z0-9_]+)", description)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _load_json_object(raw_response: str) -> dict[str, Any] | None:
        cleaned = (raw_response or "").strip()
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    def _resolve_retry_description(self, description: str, events: list[dict[str, Any]]) -> tuple[str, str]:
        normalized = description.strip().lower()
        if normalized not in self.retry_tokens:
            return description.strip(), ""

        for event in reversed(events):
            if event.get("event_type") != "shell_plan":
                continue
            prior_input = str(event.get("input", "")).strip()
            if not prior_input or prior_input.lower() in self.retry_tokens:
                continue
            return prior_input, f"检测到“{description}”，改为重试上次任务：{prior_input}"
        return "", "❌ 未找到可重试任务，请先给出明确目标描述"

    @staticmethod
    def _task_missing_file(
        description: str,
        normalized_description: str,
        capability_id: str,
        retry_note: str,
    ) -> TaskSpec:
        return TaskSpec(
            raw_description=description,
            normalized_description=normalized_description,
            capability_id=capability_id,
            parameters={},
            missing_parameters=["file"],
            note="❌ 描述缺少目标文件名，请补充例如：test123.c",
            source="ai",
            retry_note=retry_note,
        )

    def should_try_ai_language_parse(self, task: TaskSpec) -> bool:
        if task.capability_id == "__invalid__":
            return False
        if task.parameters.get("file"):
            return True
        lowered = task.normalized_description.lower()
        parse_hints = (
            "目录",
            "文件夹",
            "folder",
            "directory",
            "code",
            "代码",
            "backup",
            "config",
            "context",
            "注释",
            "优化",
            "修复",
            "修改",
            "检查",
            "explain",
            "comment",
            "optimize",
            "fix",
            "check",
        )
        return any(token in lowered for token in parse_hints)

    def build_ai_parse_prompt(self, normalized_description: str) -> str:
        capabilities = []
        for capability in list_capabilities():
            required = [parameter.name for parameter in capability.required_parameters if parameter.required]
            capabilities.append(
                {
                    "capability_id": capability.capability_id,
                    "summary": capability.summary,
                    "required": required,
                }
            )
        return (
            "你是命令能力路由器。请把用户自然语言需求解析为结构化任务。\n"
            "仅返回 JSON，不要解释。\n"
            "JSON 格式："
            '{"capability_id":"<能力ID或null>","parameters":{"file":"","base_dir":"","dir_name":"","search_dir":"","file_name":""},'
            '"missing_parameters":["<缺失参数>"],"note":"<可选说明>"}\n'
            f"可选能力：{json.dumps(capabilities, ensure_ascii=False)}\n"
            f"用户输入：{normalized_description}"
        )

    def parse_ai_task(
        self,
        *,
        raw_description: str,
        normalized_description: str,
        retry_note: str,
        raw_response: str,
    ) -> TaskSpec | None:
        parsed = self._load_json_object(raw_response)
        if not parsed:
            return None

        capability_id_raw = parsed.get("capability_id")
        if capability_id_raw is None:
            return None
        capability_id = str(capability_id_raw).strip()
        if not capability_id:
            return None
        capability = get_capability(capability_id)
        if capability is None:
            return None

        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        normalized_parameters: dict[str, str] = {}
        for key, value in parameters.items():
            if isinstance(key, str) and isinstance(value, (str, int, float)):
                normalized_parameters[key] = str(value).strip()

        file_candidate = self._extract_file_candidate(normalized_description)
        ai_file_parameter = str(normalized_parameters.get("file", "")).strip()
        if file_candidate:
            if not ai_file_parameter:
                normalized_parameters["file"] = file_candidate
            else:
                ai_file_is_basename = ai_file_parameter == Path(ai_file_parameter).name
                candidate_is_path = any(separator in file_candidate for separator in ("/", "\\"))
                if ai_file_is_basename and candidate_is_path:
                    normalized_parameters["file"] = file_candidate

        missing_parameters = parsed.get("missing_parameters", [])
        if not isinstance(missing_parameters, list):
            missing_parameters = []
        normalized_missing: list[str] = []
        for item in missing_parameters:
            if isinstance(item, str) and item.strip():
                normalized_missing.append(item.strip())

        for parameter in capability.required_parameters:
            if not parameter.required:
                continue
            value = normalized_parameters.get(parameter.name, "")
            if not value and parameter.name not in normalized_missing:
                normalized_missing.append(parameter.name)

        if "file" in normalized_missing:
            return self._task_missing_file(raw_description, normalized_description, capability_id, retry_note)

        note = str(parsed.get("note", "")).strip()
        return TaskSpec(
            raw_description=raw_description,
            normalized_description=normalized_description,
            capability_id=capability_id,
            parameters=normalized_parameters,
            missing_parameters=normalized_missing,
            note=note,
            source="ai",
            retry_note=retry_note,
        )

    def interpret(self, description: str, events: list[dict[str, Any]]) -> TaskSpec:
        normalized_description, retry_note = self._resolve_retry_description(description, events)
        if not normalized_description:
            return TaskSpec(
                raw_description=description,
                normalized_description=description.strip(),
                capability_id="__invalid__",
                parameters={},
                missing_parameters=[],
                note=retry_note or "❌ 任务描述不能为空",
                source="ai",
                retry_note="",
            )

        file_candidate = self._extract_file_candidate(normalized_description)
        parameters: dict[str, str] = {}
        if file_candidate:
            normalized = Path(file_candidate).expanduser()
            parameters["file"] = str(normalized)

        return TaskSpec(
            raw_description=description,
            normalized_description=normalized_description,
            capability_id=None,
            parameters=parameters,
            missing_parameters=[],
            note="",
            source="ai",
            retry_note=retry_note,
        )
