from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from ai_assistant.planner.types import PlanDecision, PlanStep, TaskSpec


class PlanEngine:
    def __init__(self, step_timeout_seconds: int = 30) -> None:
        self.step_timeout_seconds = step_timeout_seconds

    @staticmethod
    def _contains_placeholder(command: str) -> bool:
        return bool(re.search(r"<[A-Z0-9_]+>", command))

    @staticmethod
    def _extract_find_matches(output_text: str) -> list[str]:
        matches: list[str] = []
        for raw_line in (output_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(".") or line.startswith("/") or line.startswith("\\"):
                matches.append(line)
        return matches

    @staticmethod
    def _iter_files(search_root: Path) -> list[Path]:
        ignore_dirs = {".git", "__pycache__", "assistant-config", "assistant-state", "assistant-data"}
        files: list[Path] = []
        for current_root, dir_names, file_names in os.walk(search_root):
            dir_names[:] = [name for name in dir_names if name not in ignore_dirs]
            root_path = Path(current_root)
            for file_name in file_names:
                files.append((root_path / file_name).resolve())
        return files

    def _find_file_matches(self, candidate: str) -> list[Path]:
        raw_candidate = Path(candidate)
        if raw_candidate.is_absolute() and raw_candidate.exists() and raw_candidate.is_file():
            return [raw_candidate.resolve()]

        if any(separator in candidate for separator in ["/", "\\"]):
            direct = Path(candidate).expanduser().resolve()
            if direct.exists() and direct.is_file():
                return [direct]

        target_name = raw_candidate.name
        matches: list[Path] = []
        for file_path in self._iter_files(Path.cwd().resolve()):
            if file_path.name == target_name:
                matches.append(file_path)
        return matches

    @staticmethod
    def _end_line_expression(quoted_file: str) -> str:
        return f"\"$(lines=$(wc -l < {quoted_file}); [ \"$lines\" -gt 0 ] && echo \"$lines\" || echo 1)\""

    def _build_code_steps(self, file_path: str, mode: str) -> list[PlanStep]:
        quoted = shlex.quote(file_path)
        end_expression = self._end_line_expression(quoted)
        if mode == "check":
            return [
                PlanStep(command=f"test -f {quoted}", purpose="确认文件存在"),
                PlanStep(command=f"ai code check {quoted} --start 1 --end {end_expression}", purpose="执行代码检查"),
            ]
        if mode == "comment":
            return [
                PlanStep(command=f"test -f {quoted}", purpose="确认文件存在"),
                PlanStep(command=f"sed -n '1,80p' {quoted}", purpose="预览文件内容"),
                PlanStep(
                    command=f"ai code comment {quoted} --start 1 --end {end_expression} --yes",
                    purpose="执行注释写入",
                ),
            ]
        if mode == "explain":
            return [
                PlanStep(command=f"test -f {quoted}", purpose="确认文件存在"),
                PlanStep(command=f"ai code explain {quoted} --start 1 --end {end_expression}", purpose="解释代码"),
            ]
        if mode == "optimize":
            return [
                PlanStep(command=f"test -f {quoted}", purpose="确认文件存在"),
                PlanStep(
                    command=f"ai code optimize {quoted} --start 1 --end {end_expression} --yes",
                    purpose="优化代码",
                ),
            ]
        if mode == "fix":
            return [
                PlanStep(command=f"test -f {quoted}", purpose="确认文件存在"),
                PlanStep(command=f"ai code check {quoted} --start 1 --end {end_expression}", purpose="检查问题"),
                PlanStep(
                    command=f"ai code optimize {quoted} --start 1 --end {end_expression} --yes",
                    purpose="尝试修复问题",
                ),
                PlanStep(command=f"ai code check {quoted} --start 1 --end {end_expression}", purpose="复检结果"),
            ]
        return []

    @staticmethod
    def _build_file_resolution_steps(file_name: str, mode: str) -> list[PlanStep]:
        quoted_name = shlex.quote(file_name)
        steps = [
            PlanStep(command=f"find . -type f -name {quoted_name}", purpose="定位目标文件"),
            PlanStep(command="sed -n '1,80p' <FILE_PATH>", purpose="预览文件内容"),
        ]
        if mode == "fix":
            steps.append(PlanStep(command="ai code check <FILE_PATH> --start 1 --end <END_LINE>", purpose="检查问题"))
            steps.append(
                PlanStep(
                    command="ai code optimize <FILE_PATH> --start 1 --end <END_LINE> --yes",
                    purpose="修复问题",
                )
            )
            steps.append(PlanStep(command="ai code check <FILE_PATH> --start 1 --end <END_LINE>", purpose="复检结果"))
            return steps
        command = f"ai code {mode} <FILE_PATH> --start 1 --end <END_LINE>"
        if mode in {"comment", "optimize"}:
            command = f"{command} --yes"
        steps.append(PlanStep(command=command, purpose="执行目标操作"))
        return steps

    def _resolve_single_file(self, target_file: str) -> tuple[str | None, str]:
        matches = self._find_file_matches(target_file)
        if len(matches) == 1:
            return str(matches[0]), ""
        if len(matches) > 1:
            listing = "\n".join(f"  - {item}" for item in matches)
            return None, f"匹配到多个同名文件，请先确认目标路径：\n{listing}"
        return None, "未找到目标文件，先定位路径后再继续"

    def build_initial_steps(self, task: TaskSpec) -> tuple[bool, list[PlanStep], str]:
        capability_id = task.capability_id
        if capability_id is None:
            return False, [], ""

        if capability_id == "workflow.ensure_directory":
            base_dir = task.parameters.get("base_dir", ".")
            dir_name = task.parameters.get("dir_name")
            if not dir_name:
                return False, [], "❌ 描述缺少目录名称，请补充例如：文件夹叫 AI"
            base_path = Path(base_dir).expanduser()
            if not base_path.is_absolute():
                base_path = (Path.cwd() / base_path).resolve()
            target_path = (base_path / dir_name).resolve()
            quoted_target = shlex.quote(str(target_path))
            command = (
                f"if [ -d {quoted_target} ]; then echo {quoted_target}; "
                f"else mkdir -p {quoted_target} && echo {quoted_target}; fi"
            )
            return True, [PlanStep(command=command, purpose="确保目录存在并输出路径")], task.note

        if capability_id == "workflow.code_fix":
            target_file = task.parameters.get("file", "")
            resolved_path, note = self._resolve_single_file(target_file)
            if resolved_path:
                return True, self._build_code_steps(resolved_path, "fix"), task.note
            return True, self._build_file_resolution_steps(Path(target_file).name, "fix"), note

        if capability_id == "backup.create":
            target_file = task.parameters.get("file", "")
            if not target_file:
                return False, [], "❌ 描述缺少目标文件名，请补充例如：test123.c"
            resolved_path, note = self._resolve_single_file(target_file)
            if resolved_path:
                quoted = shlex.quote(resolved_path)
                return True, [PlanStep(command=f"ai backup create {quoted} --keep 5", purpose="创建备份")], task.note
            return True, [PlanStep(command=f"find . -type f -name {shlex.quote(Path(target_file).name)}", purpose="定位目标文件"), PlanStep(command="ai backup create <FILE_PATH> --keep 5", purpose="创建备份")], note

        if capability_id.startswith("code."):
            mode = capability_id.split(".", 1)[1]
            target_file = task.parameters.get("file", "")
            resolved_path, note = self._resolve_single_file(target_file)
            if resolved_path:
                steps = self._build_code_steps(resolved_path, mode)
                if not steps:
                    return False, [], f"❌ 暂不支持的代码流程：{mode}"
                return True, steps, task.note
            return True, self._build_file_resolution_steps(Path(target_file).name, mode), note

        return False, [], ""

    def resolve_placeholders(self, command: str, transcript: list[dict[str, Any]], task: TaskSpec) -> tuple[bool, str, str]:
        if not self._contains_placeholder(command):
            return True, command, ""

        resolved = command
        if "<FILE_NAME>" in resolved:
            file_name = task.parameters.get("file", "")
            file_name = Path(file_name).name if file_name else ""
            if not file_name:
                return False, "", "需要目标文件名，但当前描述没有提供"
            resolved = resolved.replace("<FILE_NAME>", shlex.quote(file_name))

        if "<FILE_PATH>" in resolved:
            find_records = [
                item
                for item in transcript
                if "find " in str(item.get("command", "")) and "-name" in str(item.get("command", ""))
            ]
            if not find_records:
                return False, "", "需要目标文件路径，但还没有可用的定位结果"
            last_find = find_records[-1]
            matches = self._extract_find_matches(str(last_find.get("stdout", "")))
            if not matches:
                return False, "", "没有找到目标文件，请确认文件名或搜索目录"
            if len(matches) > 1:
                listing = "\n".join(f"- {item}" for item in matches)
                return False, "", f"找到多个候选文件，请先明确目标路径：\n{listing}"
            file_path = matches[0]
            resolved = resolved.replace("<FILE_PATH>", shlex.quote(file_path))

        if "<END_LINE>" in resolved:
            file_path_match = re.search(r"'([^']+\.[A-Za-z0-9_]+)'|(/[^ ]+\.[A-Za-z0-9_]+)", resolved)
            if not file_path_match:
                return False, "", "无法推导结束行号，请先提供目标文件"
            file_path = file_path_match.group(1) or file_path_match.group(2) or ""
            quoted = shlex.quote(file_path)
            resolved = resolved.replace("<END_LINE>", self._end_line_expression(quoted))

        if self._contains_placeholder(resolved):
            return False, "", "命令仍包含未解析占位符，请补充必要信息"
        return True, resolved, ""

    @staticmethod
    def validate_ai_code_command(command: str) -> tuple[bool, str]:
        tokens = command.strip().split()
        if len(tokens) < 3:
            return True, ""
        if tokens[0] != "ai" or tokens[1] != "code":
            return True, ""

        subcommand = tokens[2]
        if subcommand not in {"check", "comment", "explain", "optimize", "generate"}:
            return True, ""

        missing_flags: list[str] = []
        if "--start" not in tokens:
            missing_flags.append("--start")
        if "--end" not in tokens:
            missing_flags.append("--end")
        if subcommand == "generate" and "--desc" not in tokens:
            missing_flags.append("--desc")

        if not missing_flags:
            return True, ""
        missing_text = ", ".join(missing_flags)
        return False, (
            f"❌ 生成命令缺少必要参数：{missing_text}\n"
            "建议补充完整参数，或直接描述文件路径和范围，例如：\n"
            "  ai code comment test123.c --start 1 --end 200"
        )

    def derive_retry_decision(self, transcript: list[dict[str, Any]]) -> PlanDecision | None:
        if not transcript:
            return None
        last_record = transcript[-1]
        last_command = str(last_record.get("command", "")).strip()
        last_exit = int(last_record.get("exit_code", 1))
        last_stderr = str(last_record.get("stderr", ""))

        if last_exit == 124 and re.search(r"\bai\s+code\s+(comment|optimize|generate)\b", last_command) and "--yes" not in last_command:
            return PlanDecision(
                action="next",
                command=f"{last_command} --yes",
                message="检测到命令超时，补充 --yes 后重试该步骤",
            )
        if "命令超时" in last_stderr and not last_command.startswith("timeout "):
            return PlanDecision(
                action="next",
                command=f"timeout {self.step_timeout_seconds}s {last_command}",
                message="检测到命令超时，尝试使用 timeout 包装后重试",
            )
        return None

    def fallback_next_from_suggestions(
        self, suggested_steps: list[str], transcript: list[dict[str, Any]], task: TaskSpec
    ) -> PlanDecision | None:
        if not suggested_steps:
            return None
        raw_next = suggested_steps.pop(0)
        resolved_ok, resolved_command, resolved_error = self.resolve_placeholders(raw_next, transcript, task)
        if not resolved_ok:
            return PlanDecision(action="need_input", message=resolved_error)
        valid, validation_message = self.validate_ai_code_command(resolved_command)
        if not valid:
            return PlanDecision(action="abort", message=validation_message)
        return PlanDecision(action="next", command=resolved_command, message="")
