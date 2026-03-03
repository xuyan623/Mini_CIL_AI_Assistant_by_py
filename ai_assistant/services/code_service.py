from __future__ import annotations

from pathlib import Path

from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.backup_service import BackupService
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService


class CodeService:
    def __init__(
        self,
        ai_client: AIClient | None = None,
        backup_service: BackupService | None = None,
        history_service: HistoryService | None = None,
        context_service: ContextService | None = None,
    ) -> None:
        self.ai_client = ai_client or AIClient()
        self.backup_service = backup_service or BackupService()
        self.history_service = history_service or HistoryService()
        self.context_service = context_service or ContextService()

    @staticmethod
    def _read_text_file(file_path: str) -> tuple[bool, str, Path]:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return False, f"❌ 无效文件：{path}", path
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            return False, f"❌ 读取文件失败：{exc}", path
        return True, content, path

    @staticmethod
    def _slice_lines(content: str, start_line: int, end_line: int) -> tuple[bool, str, list[str], int]:
        lines = content.splitlines(keepends=True)
        total = len(lines)
        if start_line < 1 or end_line < start_line or end_line > total:
            return False, f"❌ 行号无效：文件共 {total} 行，有效范围 1-{total}", lines, total
        snippet = "".join(lines[start_line - 1 : end_line])
        return True, snippet, lines, total

    @staticmethod
    def _preview_and_confirm(title: str, content: str) -> bool:
        print("=" * 60)
        print(f"📝 {title} 预览")
        print("=" * 60)
        print(content)
        print("=" * 60)
        try:
            answer = input("是否确认写入？(y/n): ").strip().lower()
        except EOFError:
            return False
        return answer == "y"

    def _build_extra_system_messages(self) -> list[str]:
        context_block = self.context_service.render_context_block(max_chars=12000)
        if not context_block:
            return []
        return [f"当前会话激活了代码上下文，回答时如果相关必须结合：\n{context_block}"]

    def _ask_ai(self, prompt: str, temperature: float = 0.2) -> tuple[bool, str]:
        self.history_service.trim_and_summarize(self.ai_client.summarize_messages)
        messages = self.history_service.build_messages_for_request(
            user_prompt=prompt,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=self._build_extra_system_messages(),
        )
        return self.ai_client.chat(messages, stream_override=False, temperature=temperature, max_tokens=4096, timeout=90)

    def check(self, file_path: str, start_line: int, end_line: int) -> str:
        ok, content_or_error, path = self._read_text_file(file_path)
        if not ok:
            return content_or_error

        valid, snippet_or_error, _, _ = self._slice_lines(content_or_error, start_line, end_line)
        if not valid:
            return snippet_or_error

        prompt = (
            "检查以下代码片段的语法/逻辑/性能/可维护性问题，给出明确问题点和修复建议：\n"
            f"```\n{snippet_or_error}\n```"
        )
        success, response = self._ask_ai(prompt, temperature=0.2)
        if not success:
            return response
        return f"📝 代码检查结果 | 文件：{path}（行 {start_line}-{end_line}）\n{response}"

    def explain(self, file_path: str, start_line: int, end_line: int) -> str:
        ok, content_or_error, path = self._read_text_file(file_path)
        if not ok:
            return content_or_error

        valid, snippet_or_error, _, _ = self._slice_lines(content_or_error, start_line, end_line)
        if not valid:
            return snippet_or_error

        prompt = f"解释以下代码片段，分点说明职责、流程、关键变量：\n```\n{snippet_or_error}\n```"
        success, response = self._ask_ai(prompt, temperature=0.2)
        if not success:
            return response
        return f"📝 代码解释结果 | 文件：{path}（行 {start_line}-{end_line}）\n{response}"

    def summarize(self, file_path: str) -> str:
        ok, content_or_error, path = self._read_text_file(file_path)
        if not ok:
            return content_or_error

        prompt = (
            "总结以下代码文件（200 字以内），包括主要职责、关键函数、依赖关系：\n"
            f"```\n{content_or_error}\n```"
        )
        success, response = self._ask_ai(prompt, temperature=0.2)
        if not success:
            return response
        return f"📝 文件总结 | 文件：{path}\n{response}"

    def comment(self, file_path: str, start_line: int, end_line: int) -> str:
        return self._modify_range(file_path, start_line, end_line, "comment")

    def optimize(self, file_path: str, start_line: int, end_line: int) -> str:
        return self._modify_range(file_path, start_line, end_line, "optimize")

    def _modify_range(self, file_path: str, start_line: int, end_line: int, mode: str) -> str:
        ok, content_or_error, path = self._read_text_file(file_path)
        if not ok:
            return content_or_error

        valid, snippet_or_error, all_lines, total_lines = self._slice_lines(content_or_error, start_line, end_line)
        if not valid:
            return snippet_or_error

        if mode == "comment":
            prompt = (
                "为以下代码添加清晰注释，仅输出修改后的纯代码，不要输出解释或 Markdown：\n"
                f"```\n{snippet_or_error}\n```"
            )
            preview_title = "注释"
        else:
            prompt = (
                "优化以下代码，保持语义不变，仅输出优化后的纯代码，不要输出解释或 Markdown：\n"
                f"```\n{snippet_or_error}\n```"
            )
            preview_title = "优化"

        success, ai_result = self._ask_ai(prompt, temperature=0.2)
        if not success:
            return ai_result

        new_snippet = self.ai_client.clean_code_block(ai_result)
        if not new_snippet:
            return "❌ AI 未返回有效代码"

        if not self._preview_and_confirm(preview_title, new_snippet):
            return "✅ 已取消"

        backup_ok, backup_message = self.backup_service.create_backup(str(path))
        if not backup_ok:
            return backup_message

        new_lines = new_snippet.splitlines(keepends=True)
        if new_snippet and not new_snippet.endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"

        merged = all_lines[: start_line - 1] + new_lines + all_lines[end_line:]
        path.write_text("".join(merged), encoding="utf-8")
        return f"{backup_message}\n✅ 文件已更新：{path}（行 {start_line}-{end_line}）"

    def generate(self, file_path: str, start_line: int, end_line: int, description: str) -> str:
        ok, content_or_error, path = self._read_text_file(file_path)
        if not ok:
            return content_or_error

        lines = content_or_error.splitlines(keepends=True)
        total = len(lines)
        if start_line < 1 or end_line < start_line or end_line > total + 1:
            return f"❌ 行号无效：文件共 {total} 行，可用范围 1-{total + 1}"

        prompt = (
            "请基于完整文件上下文生成代码片段。仅输出纯代码，不要解释。\n"
            f"完整文件：\n```\n{content_or_error}\n```\n"
            f"需求：{description}"
        )
        success, ai_result = self._ask_ai(prompt, temperature=0.4)
        if not success:
            return ai_result

        generated = self.ai_client.clean_code_block(ai_result)
        if not generated:
            return "❌ AI 未生成有效代码"

        if not self._preview_and_confirm("代码生成", generated):
            return "✅ 已取消"

        backup_ok, backup_message = self.backup_service.create_backup(str(path))
        if not backup_ok:
            return backup_message

        new_lines = generated.splitlines(keepends=True)
        if generated and not generated.endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"

        if start_line == end_line:
            merged = lines[: start_line - 1] + new_lines + lines[start_line - 1 :]
        else:
            merged = lines[: start_line - 1] + new_lines + lines[end_line:]

        path.write_text("".join(merged), encoding="utf-8")
        return f"{backup_message}\n✅ 代码生成写入完成：{path}（行 {start_line}-{end_line}）"
