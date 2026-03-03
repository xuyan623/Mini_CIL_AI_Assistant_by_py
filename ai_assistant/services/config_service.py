from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from ai_assistant.models import ProfileConfig
from ai_assistant.paths import get_path_manager, PathManager
from ai_assistant.storage import atomic_write_json, file_lock, safe_load_json


class ConfigService:
    _warned_plaintext = False

    def __init__(self, path_manager: PathManager | None = None) -> None:
        self.path_manager = path_manager or get_path_manager()
        self.path_manager.ensure_directories()
        self.lock_path = self.path_manager.config_dir / "profiles.lock"

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {
            "version": 2,
            "default_profile": "deepseek",
            "profiles": {
                "deepseek": {
                    "name": "DeepSeek Reasoner",
                    "api_key": "",
                    "api_url": "https://api.deepseek.com/v1/chat/completions",
                    "model": "deepseek-reasoner",
                    "stream": False,
                }
            },
        }

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("version") != 2:
            return self._default_payload()

        profiles = payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return self._default_payload()

        normalized_profiles: dict[str, dict[str, Any]] = {}
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not profile_id:
                continue
            if not isinstance(profile, dict):
                continue
            normalized_profiles[profile_id] = {
                "name": str(profile.get("name", profile_id)),
                "api_key": str(profile.get("api_key", "")),
                "api_url": str(profile.get("api_url", "")),
                "model": str(profile.get("model", "")),
                "stream": bool(profile.get("stream", False)),
            }

        if not normalized_profiles:
            return self._default_payload()

        default_profile = str(payload.get("default_profile", "deepseek"))
        if default_profile not in normalized_profiles:
            default_profile = next(iter(normalized_profiles.keys()))

        return {
            "version": 2,
            "default_profile": default_profile,
            "profiles": normalized_profiles,
        }

    def load_payload(self) -> dict[str, Any]:
        with file_lock(self.lock_path):
            payload = safe_load_json(self.path_manager.profiles_path, self._default_payload())
            if not isinstance(payload, dict):
                payload = self._default_payload()
            payload = self._normalize_payload(payload)
            atomic_write_json(self.path_manager.profiles_path, payload)
            self._warn_if_plaintext(payload)
            return payload

    def save_payload(self, payload: dict[str, Any]) -> None:
        with file_lock(self.lock_path):
            atomic_write_json(self.path_manager.profiles_path, payload)

    @staticmethod
    def _warn_if_plaintext(payload: dict[str, Any]) -> None:
        if os.environ.get("AI_SUPPRESS_PLAINTEXT_WARN") == "1":
            return
        if ConfigService._warned_plaintext:
            return
        for profile in payload.get("profiles", {}).values():
            api_key = str(profile.get("api_key", ""))
            if api_key.startswith("sk-"):
                print(
                    "[WARN] 明文 API Key 存在于配置文件。建议改用环境变量并避免提交到仓库。",
                    file=sys.stderr,
                )
                ConfigService._warned_plaintext = True
                break

    def list_profiles(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        active = self.get_active_profile_id(payload)
        profiles: list[dict[str, Any]] = []
        for profile_id, profile in payload["profiles"].items():
            profiles.append(
                {
                    "id": profile_id,
                    "name": profile.get("name", profile_id),
                    "model": profile.get("model", ""),
                    "api_url": profile.get("api_url", ""),
                    "stream": bool(profile.get("stream", False)),
                    "current": profile_id == active,
                }
            )
        return profiles

    def get_active_profile_id(self, payload: dict[str, Any] | None = None) -> str:
        payload = payload or self.load_payload()
        env_profile = os.environ.get("AI_CONFIG")
        if env_profile and env_profile in payload["profiles"]:
            return env_profile

        default_path = self.path_manager.default_profile_path
        if default_path.exists():
            candidate = default_path.read_text(encoding="utf-8").strip()
            if candidate in payload["profiles"]:
                return candidate

        default_profile = payload.get("default_profile", "deepseek")
        if default_profile in payload["profiles"]:
            return default_profile

        first_profile = next(iter(payload["profiles"].keys()))
        return first_profile

    def get_active_profile(self) -> ProfileConfig:
        payload = self.load_payload()
        profile_id = self.get_active_profile_id(payload)
        profile = payload["profiles"][profile_id]
        return ProfileConfig(
            profile_id=profile_id,
            name=profile.get("name", profile_id),
            api_key=profile.get("api_key", ""),
            api_url=profile.get("api_url", ""),
            model=profile.get("model", ""),
            stream=bool(profile.get("stream", False)),
        )

    def get_profile(self, profile_id: str) -> ProfileConfig | None:
        payload = self.load_payload()
        profile = payload["profiles"].get(profile_id)
        if not isinstance(profile, dict):
            return None
        return ProfileConfig(
            profile_id=profile_id,
            name=profile.get("name", profile_id),
            api_key=profile.get("api_key", ""),
            api_url=profile.get("api_url", ""),
            model=profile.get("model", ""),
            stream=bool(profile.get("stream", False)),
        )

    def list_profile_ids(self) -> list[str]:
        payload = self.load_payload()
        return list(payload.get("profiles", {}).keys())

    def add_profile(
        self,
        profile_id: str,
        name: str,
        api_key: str,
        api_url: str,
        model: str,
        stream: bool = False,
        overwrite: bool = False,
    ) -> tuple[bool, str]:
        payload = self.load_payload()
        if profile_id in payload["profiles"] and not overwrite:
            return False, f"❌ 配置 '{profile_id}' 已存在"

        payload["profiles"][profile_id] = {
            "name": name,
            "api_key": api_key,
            "api_url": api_url,
            "model": model,
            "stream": bool(stream),
        }
        self.save_payload(payload)
        return True, f"✅ 已保存配置 '{profile_id}'"

    def switch_profile(self, profile_id: str) -> tuple[bool, str]:
        payload = self.load_payload()
        if profile_id not in payload["profiles"]:
            return False, f"❌ 配置 '{profile_id}' 不存在"

        self.path_manager.default_profile_path.write_text(profile_id, encoding="utf-8")
        payload["default_profile"] = profile_id
        self.save_payload(payload)
        return True, f"✅ 已切换到配置 '{profile_id}'"

    def delete_profile(self, profile_id: str) -> tuple[bool, str]:
        payload = self.load_payload()
        if profile_id not in payload["profiles"]:
            return False, f"❌ 配置 '{profile_id}' 不存在"
        if len(payload["profiles"]) <= 1:
            return False, "❌ 至少保留一个配置"

        del payload["profiles"][profile_id]
        active = self.get_active_profile_id(payload)
        if active == profile_id:
            next_profile = next(iter(payload["profiles"].keys()))
            payload["default_profile"] = next_profile
            self.path_manager.default_profile_path.write_text(next_profile, encoding="utf-8")

        self.save_payload(payload)
        return True, f"✅ 已删除配置 '{profile_id}'"

    def set_stream(self, profile_id: str, enabled: bool) -> tuple[bool, str]:
        payload = self.load_payload()
        if profile_id not in payload["profiles"]:
            return False, f"❌ 配置 '{profile_id}' 不存在"

        payload["profiles"][profile_id]["stream"] = bool(enabled)
        self.save_payload(payload)
        status = "启用" if enabled else "禁用"
        return True, f"✅ 已为 '{profile_id}' {status}流式输出"

    def export_profile(self, profile_id: str, output_file: Path, redact: bool = False) -> tuple[bool, str]:
        payload = self.load_payload()
        if profile_id not in payload["profiles"]:
            return False, f"❌ 配置 '{profile_id}' 不存在"

        profile = dict(payload["profiles"][profile_id])
        if redact:
            api_key = profile.get("api_key", "")
            profile["api_key_display"] = f"{api_key[:8]}..." if api_key else ""
            profile["api_key"] = ""

        export_payload = {
            "version": 2,
            "profile_id": profile_id,
            "profile": profile,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, f"✅ 已导出配置到 {output_file}"

    def import_profile(self, input_file: Path, profile_id: str | None = None) -> tuple[bool, str]:
        if not input_file.exists():
            return False, f"❌ 文件不存在：{input_file}"

        try:
            payload = json.loads(input_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"❌ 无法解析配置文件：{exc}"

        if not isinstance(payload, dict) or payload.get("version") != 2 or "profile" not in payload:
            return False, "❌ 仅支持 v2 导出配置格式"

        imported_profile_id = str(payload.get("profile_id", "imported"))
        profile = payload["profile"]
        if not isinstance(profile, dict):
            return False, "❌ 配置文件格式无效：profile 必须是对象"

        target_profile_id = profile_id or imported_profile_id

        if not profile.get("api_key"):
            return False, "❌ 导入配置缺少 api_key（可能是脱敏导出文件）"

        required_fields = ["api_url", "model"]
        for field_name in required_fields:
            if not profile.get(field_name):
                return False, f"❌ 导入配置缺少 {field_name}"

        return self.add_profile(
            profile_id=target_profile_id,
            name=profile.get("name", target_profile_id),
            api_key=profile["api_key"],
            api_url=profile["api_url"],
            model=profile["model"],
            stream=bool(profile.get("stream", False)),
            overwrite=False,
        )
