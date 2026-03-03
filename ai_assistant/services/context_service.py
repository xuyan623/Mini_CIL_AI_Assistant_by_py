from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_assistant.paths import PathManager, get_path_manager
from ai_assistant.storage import atomic_write_json, safe_load_json


class ContextService:
    def __init__(self, path_manager: PathManager | None = None) -> None:
        self.path_manager = path_manager or get_path_manager()

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {"version": 1, "files": []}

    def load_payload(self) -> dict[str, Any]:
        payload = safe_load_json(self.path_manager.context_path, self._default_payload())
        if not isinstance(payload, dict):
            payload = self._default_payload()
        payload.setdefault("version", 1)
        payload.setdefault("files", [])
        return payload

    def save_payload(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.path_manager.context_path, payload)

    @staticmethod
    def _read_file_content(file_path: Path, start_line: int | None, end_line: int | None) -> tuple[str, str]:
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        content_lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if start_line is None or end_line is None:
            return "".join(content_lines), "全文"

        total_lines = len(content_lines)
        if start_line < 1 or end_line < start_line or end_line > total_lines:
            raise ValueError(f"行号无效：文件共 {total_lines} 行，收到 {start_line}-{end_line}")

        snippet = "".join(content_lines[start_line - 1 : end_line])
        return snippet, f"行 {start_line}-{end_line}"

    def set_context(self, file_path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        path = Path(file_path).expanduser().resolve()
        code_content, range_info = self._read_file_content(path, start_line, end_line)

        payload = {
            "version": 1,
            "files": [
                {
                    "path": str(path),
                    "range": range_info,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": code_content,
                }
            ],
        }
        self.save_payload(payload)
        return f"✅ 已设置上下文：{path}（{range_info}）"

    def add_context(self, file_path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        path = Path(file_path).expanduser().resolve()
        code_content, range_info = self._read_file_content(path, start_line, end_line)

        payload = self.load_payload()
        files = payload["files"]
        if any(item.get("path") == str(path) for item in files):
            return f"⚠️ 文件已在上下文中：{path}"

        files.append(
            {
                "path": str(path),
                "range": range_info,
                "start_line": start_line,
                "end_line": end_line,
                "code": code_content,
            }
        )
        self.save_payload(payload)
        return f"✅ 已追加上下文：{path}（{range_info}）"

    def list_context(self) -> str:
        payload = self.load_payload()
        files = payload.get("files", [])
        if not files:
            return "📝 当前没有代码上下文"

        lines = ["📝 当前代码上下文："]
        for index, file_item in enumerate(files, 1):
            lines.append(f"  {index}. {file_item['path']} ({file_item['range']})")
        return "\n".join(lines)

    def clear_context(self) -> str:
        self.save_payload(self._default_payload())
        return "✅ 已清除代码上下文"

    def get_context_files(self) -> list[dict[str, Any]]:
        payload = self.load_payload()
        files = payload.get("files", [])
        if not isinstance(files, list):
            return []
        return files

    def render_context_block(self, max_chars: int | None = None) -> str:
        files = self.get_context_files()
        if not files:
            return ""

        blocks: list[str] = []
        for index, item in enumerate(files, 1):
            blocks.append(
                f"【文件 {index}/{len(files)}】\n"
                f"路径：{item['path']}\n"
                f"范围：{item['range']}\n"
                f"```\n{item['code']}\n```"
            )

        context_text = f"=== 代码上下文 ===\n{'\n\n'.join(blocks)}"
        if max_chars is None or len(context_text) <= max_chars:
            return context_text
        return f"{context_text[:max_chars]}\n...(上下文已截断)"

    def build_prompt(self, question: str, recent_messages: list[dict[str, str]] | None = None) -> str:
        context_block = self.render_context_block()
        if not context_block:
            raise ValueError("未设置代码上下文，请先执行 context set")

        history_block = ""
        if recent_messages:
            history_text = "\n".join(
                [f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in recent_messages]
            )
            history_block = f"\n\n=== 最近对话 ===\n{history_text}"

        return (
            "你是代码助手，请结合以下代码上下文回答问题。\n\n"
            f"{context_block}"
            f"{history_block}\n\n"
            f"=== 问题 ===\n{question}"
        )
