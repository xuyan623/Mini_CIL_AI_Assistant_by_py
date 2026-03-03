from __future__ import annotations

import os
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".py",
    ".sh",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".conf",
    ".ini",
    ".log",
    ".csv",
    ".c",
    ".h",
}


class FileService:
    def __init__(self) -> None:
        self.sensitive_prefixes = [
            "/",
            "/root",
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/sys",
            "/proc",
            "/dev",
            "/boot",
            "/opt",
            "/var",
            "/tmp",
        ]

    @staticmethod
    def _resolve(path: str | None, default: str = ".") -> Path:
        raw = path if path else default
        return Path(raw).expanduser().resolve()

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
        try:
            path.read_text(encoding="utf-8")
            return True
        except Exception:
            return False

    def _is_sensitive(self, path: Path) -> bool:
        normalized = str(path).replace("\\", "/")
        if os.name == "nt":
            # Windows fallback: treat drive root as sensitive.
            drive, _ = os.path.splitdrive(normalized)
            if drive and normalized in {f"{drive}/", f"{drive}"}:
                return True
        for prefix in self.sensitive_prefixes:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return True
        return False

    @staticmethod
    def _confirm(prompt: str) -> bool:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return False
        return answer == "y"

    def list_directory(self, directory: str | None = None) -> str:
        base_path = self._resolve(directory)
        if not base_path.exists():
            return f"❌ 目录不存在：{base_path}"
        if not base_path.is_dir():
            return f"❌ 非目录：{base_path}"

        def render_tree(path: Path, prefix: str = "") -> list[str]:
            try:
                children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except PermissionError:
                return [prefix + "└─ [Permission Denied]"]

            lines: list[str] = []
            for index, child in enumerate(children):
                connector = "└─ " if index == len(children) - 1 else "├─ "
                name = child.name + ("/" if child.is_dir() else "")
                lines.append(prefix + connector + name)
                if child.is_dir():
                    extension = "   " if index == len(children) - 1 else "│  "
                    lines.extend(render_tree(child, prefix + extension))
            return lines

        tree_lines = render_tree(base_path)
        return f"📁 目录结构：{base_path}\n" + "\n".join(tree_lines)

    def read_file(self, file_path: str, max_chars: int = 20000) -> str:
        path = self._resolve(file_path)
        if not path.exists() or not path.is_file():
            return f"❌ 无效文件：{path}"
        if not self._is_text_file(path):
            return f"❌ 非文本文件：{path}"

        try:
            content = path.read_text(encoding="utf-8")
        except PermissionError:
            return f"❌ 无权限读取：{path}"
        except Exception as exc:
            return f"❌ 读取失败：{exc}"

        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        suffix = ""
        if truncated:
            suffix = f"\n\n⚠️ 内容已截断为前 {max_chars} 字符"
        return f"📄 文件：{path}\n内容：\n{content}{suffix}"

    def search_file(self, file_path: str, keyword: str) -> str:
        path = self._resolve(file_path)
        if not path.exists() or not path.is_file() or not self._is_text_file(path):
            return f"❌ 无效文件：{path}"
        if not keyword:
            return "❌ 关键词不能为空"

        matches: list[str] = []
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if keyword.lower() in line.lower():
                        matches.append(f"第{line_number}行：{line.rstrip()}")
        except Exception as exc:
            return f"❌ 搜索失败：{exc}"

        if not matches:
            return f"🔍 无匹配：{path}（关键词：{keyword}）"
        return f"🔍 匹配{len(matches)}处：{path}\n" + "\n".join(matches)

    def find_files(self, keyword: str, search_dir: str | None = None) -> str:
        base_path = self._resolve(search_dir)
        if not base_path.exists() or not base_path.is_dir():
            return f"❌ 无效目录：{base_path}"
        if not keyword:
            return "❌ 关键词不能为空"

        found: list[str] = []
        for root_dir, _, files in os.walk(base_path):
            for name in files:
                if keyword.lower() in name.lower():
                    found.append(str(Path(root_dir) / name))

        if not found:
            return f"🔍 未找到包含“{keyword}”的文件（目录：{base_path}）"

        sorted_files = sorted(found)
        lines = [f"🔍 找到{len(sorted_files)}个文件（目录：{base_path}）："]
        lines.extend([f"   └─ {item}" for item in sorted_files])
        return "\n".join(lines)

    def remove_file(self, file_path: str, force: bool = False) -> str:
        path = self._resolve(file_path)
        if not path.exists() or not path.is_file():
            return f"❌ 无效文件：{path}"

        if not force and self._is_sensitive(path):
            return f"⚠️ 敏感路径，若确认删除请使用 --force：{path}"

        if not self._confirm(f"⚠️ 确认删除文件（不可恢复）{path} ? 输入 y 确认："):
            return "✅ 已取消"

        try:
            path.unlink()
            return f"✅ 已删除文件：{path}"
        except Exception as exc:
            return f"❌ 删除失败：{exc}"

    def remove_directory(self, directory: str, force: bool = False) -> str:
        path = self._resolve(directory)
        if not path.exists() or not path.is_dir():
            return f"❌ 无效目录：{path}"

        if not force and self._is_sensitive(path):
            return f"⚠️ 敏感路径，若确认删除请使用 --force：{path}"

        if not self._confirm(f"⚠️ 确认删除目录（含子文件，不可恢复）{path} ? 输入 y 确认："):
            return "✅ 已取消"

        try:
            shutil.rmtree(path)
            return f"✅ 已删除目录：{path}"
        except Exception as exc:
            return f"❌ 删除失败：{exc}"
