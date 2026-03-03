from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService


@dataclass
class ShellSafetyReport:
    safe: bool
    warnings: list[str]


class ShellService:
    def __init__(
        self,
        ai_client: AIClient | None = None,
        history_service: HistoryService | None = None,
        context_service: ContextService | None = None,
    ) -> None:
        self.ai_client = ai_client or AIClient()
        self.history_service = history_service or HistoryService()
        self.context_service = context_service or ContextService()
        self.dangerous_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\brm\s+-rf\s+/\b"), "会删除整个系统"),
            (re.compile(r"\brm\s+-rf\s+~\b"), "会删除用户主目录"),
            (re.compile(r"\bdd\s+if="), "可能覆盖磁盘数据"),
            (re.compile(r":\(\)\s*\{\s*:\|:&\s*;\s*\}"), "fork 炸弹"),
            (re.compile(r"\bmkfs\.[^\s]+"), "会格式化磁盘"),
            (re.compile(r"\bchmod\s+-R\s+777\s+/\b"), "会放开系统权限"),
            (re.compile(r"\bchown\s+-R\b"), "可能破坏权限"),
            (re.compile(r"\bwget\b.*\|\s*(bash|sh)\b"), "下载并执行远程脚本"),
            (re.compile(r"\bcurl\b.*\|\s*(bash|sh)\b"), "下载并执行远程脚本"),
        ]

    def _build_extra_system_messages(self) -> list[str]:
        context_block = self.context_service.render_context_block(max_chars=8000)
        if not context_block:
            return []
        return [f"当前激活了代码上下文。若用户描述与项目相关，请先结合上下文生成命令：\n{context_block}"]

    def safety_check(self, command: str) -> ShellSafetyReport:
        warnings: list[str] = []
        for pattern, reason in self.dangerous_patterns:
            if pattern.search(command):
                warnings.append(f"⚠️ 危险模式：{reason}")

        if re.search(r"\bsudo\b", command):
            warnings.append("⚠️ 命令需要 root 权限")
        return ShellSafetyReport(safe=(len(warnings) == 0), warnings=warnings)

    def generate_command(self, description: str) -> tuple[bool, str]:
        prompt = (
            "你是 Linux shell 命令生成助手。"
            "根据用户描述只返回一个可执行命令，不要解释，不要 Markdown。"
            f"\n用户描述：{description}"
        )
        self.history_service.trim_and_summarize(self.ai_client.summarize_messages)
        messages = self.history_service.build_messages_for_request(
            user_prompt=prompt,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=self._build_extra_system_messages(),
        )
        ok, response = self.ai_client.chat(
            messages,
            stream_override=False,
            temperature=0.2,
            max_tokens=512,
            timeout=60,
        )
        if not ok:
            return False, response

        command = self.ai_client.clean_code_block(response).strip().splitlines()[0].strip()
        if not command:
            return False, "❌ 未生成有效命令"
        return True, command

    def run(self, description: str, execute: bool = False) -> tuple[bool, str]:
        ok, command_or_error = self.generate_command(description)
        if not ok:
            return False, command_or_error

        command = command_or_error
        report = self.safety_check(command)

        lines = ["📋 生成命令：", f"  $ {command}"]
        if report.warnings:
            lines.append("⚠️ 安全警告：")
            lines.extend([f"  - {item}" for item in report.warnings])

        if not execute:
            lines.append("ℹ️ 未执行命令（使用 --execute 执行）")
            return True, "\n".join(lines)

        if not self._confirm_execution(report.safe):
            lines.append("✅ 已取消执行")
            return False, "\n".join(lines)

        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        lines.append(f"\n🧾 退出码：{process.returncode}")
        if process.stdout:
            lines.append("\n📤 标准输出：")
            lines.append(process.stdout.rstrip())
        if process.stderr:
            lines.append("\n⚠️ 标准错误：")
            lines.append(process.stderr.rstrip())

        return process.returncode == 0, "\n".join(lines)

    @staticmethod
    def _confirm_execution(is_safe: bool) -> bool:
        question = "是否执行该命令？(y/n): " if is_safe else "命令包含风险，仍要执行？(y/n): "
        try:
            answer = input(question).strip().lower()
        except EOFError:
            return False
        return answer == "y"
