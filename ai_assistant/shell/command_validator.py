from __future__ import annotations

import re


class ShellCommandValidator:
    @staticmethod
    def is_natural_language_line(command: str) -> bool:
        candidate = command.strip()
        if not candidate:
            return True
        if len(candidate) > 180:
            return True
        rejected_prefixes = (
            "首先",
            "关键点",
            "用户描述",
            "回顾",
            "可能的情况",
            "目标是",
            "作为助手",
            "总结",
            "说明",
            "请注意",
        )
        if candidate.startswith(rejected_prefixes):
            return True
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", candidate)
        if len(chinese_chars) >= 6 and not re.search(r"[|;$><]", candidate):
            if not re.search(r"\b(find|test|sed|head|wc|cp|mv|mkdir|ls|cat|ai)\b", candidate):
                return True
        return False

    @staticmethod
    def contains_placeholder_token(command: str) -> bool:
        candidate = (command or "").strip()
        if not candidate:
            return False
        direct_placeholders = (
            "<FILE_PATH>",
            "<FILE_NAME>",
            "<END_LINE>",
            "<可直接执行命令>",
            "<该步目的>",
            "<简短目标总结>",
            "<YOUR_COMMAND>",
            "<COMMAND_HERE>",
        )
        if any(token in candidate for token in direct_placeholders):
            return True
        if re.fullmatch(r"<[^<>]{1,80}>", candidate):
            return True
        for match in re.finditer(r"<([^<>]{1,80})>", candidate):
            inner = match.group(1).strip()
            if not inner:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner):
                return True
            if re.search(r"[\u4e00-\u9fff]", inner):
                return True
            if " " in inner:
                return True
        return False

    def validate(self, command: str) -> tuple[bool, str]:
        candidate = (command or "").strip()
        if not candidate:
            return False, "❌ 生成的命令为空，已中止执行"
        if self.contains_placeholder_token(candidate):
            return False, f"❌ 生成命令包含未替换占位符：{candidate[:120]}"
        if self.is_natural_language_line(candidate):
            return False, f"❌ 生成内容不是可执行命令：{candidate[:120]}"
        return True, ""
