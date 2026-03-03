from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_assistant.paths import PathManager, get_path_manager
from ai_assistant.storage import atomic_write_json, file_lock, safe_load_json


class BackupService:
    def __init__(self, path_manager: PathManager | None = None) -> None:
        self.path_manager = path_manager or get_path_manager()
        self.path_manager.ensure_directories()
        self.lock_path = self.path_manager.data_dir / "backup.lock"
        self._ensure_index()

    @staticmethod
    def _default_index() -> dict[str, Any]:
        return {"version": 1, "items": []}

    def _ensure_index(self) -> None:
        with file_lock(self.lock_path):
            index = safe_load_json(self.path_manager.backup_index_path, self._default_index())
            if not isinstance(index, dict):
                index = self._default_index()
            index.setdefault("version", 1)
            index.setdefault("items", [])

            known = {item.get("backup_file") for item in index["items"]}
            for backup_file in self.path_manager.backup_dir.glob("*.bak"):
                if backup_file.name in known:
                    continue
                parsed = self._parse_backup_filename(backup_file.name)
                index["items"].append(
                    {
                        "backup_file": backup_file.name,
                        "source_path": parsed.get("source_path", ""),
                        "source_id": parsed.get("source_id", "legacy"),
                        "ts": parsed.get("ts", ""),
                    }
                )
            atomic_write_json(self.path_manager.backup_index_path, index)

    @staticmethod
    def _normalize_path(path: Path) -> str:
        normalized = str(path.resolve())
        normalized = normalized.replace("\\", "/")
        if os.name == "nt":
            normalized = normalized.lower()
        return normalized

    def _source_id(self, source_path: Path) -> str:
        normalized = self._normalize_path(source_path)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _parse_backup_filename(filename: str) -> dict[str, str]:
        if "__" in filename:
            parts = filename.split("__")
            if len(parts) >= 3:
                source_id = parts[0]
                ts_with_ext = parts[-1]
                ts = ts_with_ext[:-4] if ts_with_ext.endswith(".bak") else ts_with_ext
                basename = "__".join(parts[1:-1])
                return {
                    "source_id": source_id,
                    "source_path": basename,
                    "ts": ts,
                }

        # legacy format: basename.YYYYMMDD_HHMMSS.bak
        if filename.endswith(".bak") and len(filename) > 20:
            base = filename[:-4]
            if len(base) > 16:
                ts = base[-15:]
                source_name = base[:-16]
                return {
                    "source_id": f"legacy-{hashlib.sha256(source_name.encode('utf-8')).hexdigest()[:12]}",
                    "source_path": source_name,
                    "ts": ts,
                }

        return {"source_id": "legacy", "source_path": filename, "ts": ""}

    def _load_index(self) -> dict[str, Any]:
        with file_lock(self.lock_path):
            index = safe_load_json(self.path_manager.backup_index_path, self._default_index())
            if not isinstance(index, dict):
                index = self._default_index()
            index.setdefault("version", 1)
            index.setdefault("items", [])
            return index

    def _save_index(self, index: dict[str, Any]) -> None:
        with file_lock(self.lock_path):
            atomic_write_json(self.path_manager.backup_index_path, index)

    def create_backup(self, file_path: str, keep: int = 5) -> tuple[bool, str]:
        source = Path(file_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            return False, f"❌ 无效文件：{source}"

        source_id = self._source_id(source)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_filename = f"{source_id}__{source.name}__{timestamp}.bak"
        backup_path = self.path_manager.backup_dir / backup_filename

        shutil.copy2(source, backup_path)

        index = self._load_index()
        index["items"].append(
            {
                "backup_file": backup_filename,
                "source_path": str(source),
                "source_id": source_id,
                "ts": timestamp,
            }
        )

        removed = self._rotate_index(index, source_id, keep)
        self._save_index(index)

        message = f"✅ 备份完成：{backup_path}"
        if removed:
            message += f"\n🧹 已清理旧备份 {removed} 个（保留 {keep} 个）"
        return True, message

    def _rotate_index(self, index: dict[str, Any], source_id: str, keep: int) -> int:
        source_items = [item for item in index["items"] if item.get("source_id") == source_id]
        source_items.sort(key=lambda item: item.get("ts", ""), reverse=True)

        removed = 0
        for old_item in source_items[keep:]:
            backup_file = old_item.get("backup_file", "")
            backup_path = self.path_manager.backup_dir / backup_file
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception:
                    continue
            index["items"].remove(old_item)
            removed += 1
        return removed

    def list_backups(self, file_path: str | None = None) -> list[dict[str, Any]]:
        index = self._load_index()
        items = list(index["items"])

        if file_path:
            source_id = self._source_id(Path(file_path).expanduser().resolve())
            items = [item for item in items if item.get("source_id") == source_id]

        items.sort(key=lambda item: item.get("ts", ""), reverse=True)
        return items

    def backup_status(self, file_path: str | None = None) -> str:
        items = self.list_backups(file_path)
        if not items:
            if file_path:
                return f"📝 文件没有备份：{Path(file_path).expanduser().resolve()}"
            return "📝 暂无备份"

        lines = [f"📝 备份统计：共 {len(items)} 个"]
        for index, item in enumerate(items[:5], 1):
            lines.append(f"  {index}. {item['backup_file']} ({item.get('ts', '')})")
        if len(items) > 5:
            lines.append(f"  ... 还有 {len(items) - 5} 个")
        return "\n".join(lines)

    def restore_backup(
        self,
        backup_file: str,
        target_path: str | None = None,
        backup_target_before_restore: bool = True,
    ) -> tuple[bool, str]:
        backup_candidate = Path(backup_file)
        if backup_candidate.exists():
            backup_path = backup_candidate.resolve()
        else:
            backup_path = (self.path_manager.backup_dir / backup_file).resolve()

        if not backup_path.exists() or not backup_path.is_file():
            return False, f"❌ 备份文件不存在：{backup_file}"

        index = self._load_index()
        entry = next((item for item in index["items"] if item.get("backup_file") == backup_path.name), None)

        if target_path:
            target = Path(target_path).expanduser().resolve()
        else:
            if entry and entry.get("source_path"):
                source_path = Path(str(entry["source_path"]))
                target = source_path.expanduser().resolve()
            else:
                return False, "❌ 无法推断恢复目标路径，请通过 --target 指定"

        if backup_target_before_restore and target.exists() and target.is_file():
            self.create_backup(str(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, target)
        return True, f"✅ 已恢复备份：{backup_path}\n📍 目标文件：{target}"

    def clean_backups(self, file_path: str, keep: int) -> tuple[bool, str]:
        source = Path(file_path).expanduser().resolve()
        source_id = self._source_id(source)
        index = self._load_index()
        removed = self._rotate_index(index, source_id, keep)
        self._save_index(index)
        return True, f"✅ 清理完成：删除 {removed} 个旧备份，保留 {keep} 个"
