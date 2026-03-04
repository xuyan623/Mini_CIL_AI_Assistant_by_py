from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from ai_assistant.planner.types import ExecutionFacts, PlanDecision, PlanStep, StepRewriteResult, TaskSpec, WorkflowState


class PlanEngine:
    def __init__(self, step_timeout_seconds: int = 30) -> None:
        self.step_timeout_seconds = step_timeout_seconds

    @staticmethod
    def _contains_placeholder(command: str) -> bool:
        return bool(re.search(r"<[^<>]{1,120}>", command))

    @staticmethod
    def _normalize_command(command: str) -> str:
        return " ".join((command or "").strip().split())

    @classmethod
    def _command_seen(cls, command: str, transcript: list[dict[str, Any]]) -> bool:
        expected = cls._normalize_command(command)
        for record in transcript:
            candidate = cls._normalize_command(str(record.get("command", "")))
            if candidate == expected:
                return True
        return False

    @classmethod
    def _command_result(cls, command: str, transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
        expected = cls._normalize_command(command)
        for record in reversed(transcript):
            candidate = cls._normalize_command(str(record.get("command", "")))
            if candidate == expected:
                return record
        return None

    @staticmethod
    def _extract_find_matches(output_text: str) -> list[str]:
        matches: list[str] = []
        for raw_line in (output_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(".") or line.startswith("/") or line.startswith("\\") or re.match(r"^[A-Za-z]:\\", line):
                matches.append(line)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in matches:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

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
    def _build_discovery_step(file_name: str) -> PlanStep:
        quoted_name = shlex.quote(file_name)
        return PlanStep(command=f"find . -type f -name {quoted_name}", purpose="定位目标文件")

    @staticmethod
    def _mode_for_task(task: TaskSpec) -> str | None:
        capability_id = task.capability_id or ""
        if capability_id == "workflow.code_fix":
            return "fix"
        if capability_id == "backup.create":
            return "backup"
        if capability_id.startswith("code."):
            return capability_id.split(".", 1)[1]
        return None

    def _build_sequence_for_mode(self, mode: str, file_path: str) -> list[PlanStep]:
        if mode == "backup":
            quoted = shlex.quote(file_path)
            return [PlanStep(command=f"ai backup create {quoted} --keep 5", purpose="创建备份")]
        return self._build_code_steps(file_path, mode)

    @staticmethod
    def _resolve_existing_file(path_text: str) -> str:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file():
            return str(candidate)
        return ""

    @staticmethod
    def _normalize_path_token(path_text: str) -> str:
        candidate = str(path_text or "").strip().strip("\"' ")
        if not candidate:
            return ""
        if re.match(r"^[A-Za-z]:\\", candidate):
            return candidate
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    @staticmethod
    def _first_int_from_output(output_text: str) -> int | None:
        match = re.search(r"(-?\d+)", str(output_text or ""))
        if not match:
            return None
        try:
            value = int(match.group(1))
        except Exception:
            return None
        return value if value >= 0 else None

    @staticmethod
    def _extract_file_from_wc_command(command: str) -> str:
        match = re.search(r"wc\s+-l\s*<\s*([^\s;\)]+)", command)
        if not match:
            return ""
        return str(match.group(1)).strip().strip("\"'")

    @staticmethod
    def _extract_file_from_testf_command(command: str) -> str:
        match = re.search(r"^\s*test\s+-f\s+(.+)$", command)
        if not match:
            return ""
        return str(match.group(1)).strip().strip("\"'")

    @staticmethod
    def _extract_name_from_find_command(command: str) -> str:
        match = re.search(r"-name\s+([^\s]+)", command)
        if not match:
            return ""
        return str(match.group(1)).strip().strip("\"'")

    def extract_execution_facts(self, transcript: list[dict[str, Any]], task: TaskSpec) -> ExecutionFacts:
        facts = ExecutionFacts()
        target_file = str(task.parameters.get("file", "")).strip()
        if target_file:
            normalized_target = self._normalize_path_token(target_file)
            if normalized_target:
                facts.resolved_files["target"] = normalized_target
                facts.resolved_files[Path(normalized_target).name] = normalized_target

        for index, record in enumerate(transcript, 1):
            command = str(record.get("command", "")).strip()
            exit_code = int(record.get("exit_code", 0))
            stdout_text = str(record.get("stdout", ""))
            if not command:
                continue

            test_file = self._extract_file_from_testf_command(command)
            if test_file and exit_code == 0:
                normalized_test_file = self._normalize_path_token(test_file)
                if normalized_test_file:
                    facts.file_exists_ok.add(normalized_test_file)
                    facts.resolved_files.setdefault(Path(normalized_test_file).name, normalized_test_file)

            if "find " in command and "-name" in command and exit_code == 0:
                find_matches = self._extract_find_matches(stdout_text)
                if len(find_matches) == 1:
                    normalized_match = self._normalize_path_token(find_matches[0])
                    if normalized_match:
                        facts.file_exists_ok.add(normalized_match)
                        facts.resolved_files["target"] = normalized_match
                        facts.resolved_files[Path(normalized_match).name] = normalized_match

            wc_file = self._extract_file_from_wc_command(command)
            if wc_file and exit_code == 0:
                line_count = self._first_int_from_output(stdout_text)
                if line_count is not None:
                    normalized_wc_file = self._normalize_path_token(wc_file)
                    if normalized_wc_file:
                        facts.file_line_count[normalized_wc_file] = max(line_count, 1)

            write_match = re.search(r"\bai\s+code\s+(comment|optimize|generate)\s+([^\s]+)", command)
            if write_match and exit_code == 0:
                write_file = str(write_match.group(2)).strip().strip("\"'")
                normalized_write_file = self._normalize_path_token(write_file)
                if normalized_write_file:
                    next_version = int(facts.file_version_token.get(normalized_write_file, 0)) + 1
                    facts.file_version_token[normalized_write_file] = next_version
                    facts.last_mutation_step[normalized_write_file] = index
                    facts.file_line_count.pop(normalized_write_file, None)
                    facts.file_exists_ok.add(normalized_write_file)
                    facts.resolved_files["target"] = normalized_write_file
                    facts.resolved_files[Path(normalized_write_file).name] = normalized_write_file

        return facts

    @staticmethod
    def _replace_end_expression(command: str, raw_file: str, line_count: int) -> str:
        replacements = [
            (
                rf'"?\$\(lines=\$\(wc -l < {re.escape(raw_file)}\); '
                rf'\[ "\$lines" -gt 0 \] && echo "\$lines" \|\| echo 1\)"?'
            ),
            rf'"?\$\(wc -l < {re.escape(raw_file)}\)"?',
        ]
        rewritten = command
        for pattern in replacements:
            rewritten = re.sub(pattern, str(line_count), rewritten)
        return rewritten

    def rewrite_command_with_facts(self, command: str, facts: ExecutionFacts, task: TaskSpec) -> StepRewriteResult:
        if "wc -l <" not in command or "--end" not in command:
            return StepRewriteResult(command=command, rewritten=False, reason="")

        raw_file = self._extract_file_from_wc_command(command)
        if not raw_file:
            return StepRewriteResult(command=command, rewritten=False, reason="")

        normalized_file = self._normalize_path_token(raw_file)
        line_count = facts.file_line_count.get(normalized_file)
        if line_count is None:
            fallback_file = self._normalize_path_token(str(task.parameters.get("file", "")))
            if fallback_file and fallback_file == normalized_file:
                line_count = facts.file_line_count.get(fallback_file)
        if line_count is None:
            return StepRewriteResult(command=command, rewritten=False, reason="")

        rewritten_command = self._replace_end_expression(command, raw_file, line_count)
        if rewritten_command == command:
            return StepRewriteResult(command=command, rewritten=False, reason="")
        return StepRewriteResult(
            command=rewritten_command,
            rewritten=True,
            reason=f"已复用上一步行数: {line_count}",
        )

    def should_skip_redundant_step(self, command: str, facts: ExecutionFacts, task: TaskSpec) -> tuple[bool, str]:
        normalized_command = self._normalize_command(command)

        test_file = self._extract_file_from_testf_command(normalized_command)
        if test_file:
            normalized_test_file = self._normalize_path_token(test_file)
            if normalized_test_file and normalized_test_file in facts.file_exists_ok:
                return True, f"目标文件已确认存在：{normalized_test_file}"
            return False, ""

        if normalized_command.startswith("find ") and " -type f " in f" {normalized_command} " and " -name " in f" {normalized_command} ":
            find_name = self._extract_name_from_find_command(normalized_command)
            if not find_name:
                return False, ""
            normalized_target_file = self._normalize_path_token(str(task.parameters.get("file", "")))
            if normalized_target_file and Path(normalized_target_file).name == find_name and normalized_target_file in facts.file_exists_ok:
                return True, f"目标路径已确定：{normalized_target_file}"
            same_name_candidates = [item for item in facts.file_exists_ok if Path(item).name == find_name]
            if len(same_name_candidates) == 1:
                return True, f"已存在唯一候选路径：{same_name_candidates[0]}"
            return False, ""

        return False, ""

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

        mode = self._mode_for_task(task)
        if mode is None:
            return False, [], ""

        target_file = str(task.parameters.get("file", "")).strip()
        if not target_file:
            return False, [], "❌ 描述缺少目标文件名，请补充例如：test123.c"

        resolved_path, note = self._resolve_single_file(target_file)
        if resolved_path:
            steps = self._build_sequence_for_mode(mode, resolved_path)
            if not steps:
                return False, [], f"❌ 暂不支持的代码流程：{mode}"
            return True, steps, task.note

        discovery_step = self._build_discovery_step(Path(target_file).name)
        return True, [discovery_step], note

    def build_workflow_state(self, task: TaskSpec, transcript: list[dict[str, Any]]) -> WorkflowState:
        target_file = str(task.parameters.get("file", "")).strip()
        state = WorkflowState(
            target_file=target_file,
            resolution_status="unresolved",
            candidates=[],
            resolved_file="",
            last_step=str(transcript[-1].get("command", "")).strip() if transcript else "",
        )
        if not target_file:
            return state

        existing = self._resolve_existing_file(target_file)
        if existing:
            state.resolution_status = "resolved"
            state.resolved_file = existing
            return state

        find_records = [
            item
            for item in transcript
            if "find " in str(item.get("command", "")) and "-name" in str(item.get("command", ""))
        ]
        if not find_records:
            return state

        latest_find = find_records[-1]
        matches = self._extract_find_matches(str(latest_find.get("stdout", "")))
        if not matches:
            state.resolution_status = "missing"
            return state
        if len(matches) == 1:
            state.resolution_status = "resolved"
            state.resolved_file = matches[0]
            state.candidates = matches
            return state

        state.resolution_status = "ambiguous"
        state.candidates = matches
        return state

    def derive_workflow_decision(self, task: TaskSpec, transcript: list[dict[str, Any]]) -> PlanDecision | None:
        mode = self._mode_for_task(task)
        if mode is None:
            return None

        state = self.build_workflow_state(task, transcript)
        if not state.target_file:
            return PlanDecision(action="need_input", message="缺少目标文件，请补充完整路径")

        if state.resolution_status == "unresolved":
            return PlanDecision(action="next", command=self._build_discovery_step(Path(state.target_file).name).command)
        if state.resolution_status == "missing":
            return PlanDecision(
                action="need_input",
                message=f"没有找到目标文件：{Path(state.target_file).name}，请补充路径或确认搜索目录",
            )
        if state.resolution_status == "ambiguous":
            listing = "\n".join(f"- {item}" for item in state.candidates[:10])
            return PlanDecision(action="need_input", message=f"找到多个候选文件，请先明确目标路径：\n{listing}")

        resolved_file = state.resolved_file
        if not resolved_file:
            return PlanDecision(action="need_input", message="未能解析目标文件路径，请补充明确路径")
        task.parameters["file"] = resolved_file

        sequence = self._build_sequence_for_mode(mode, resolved_file)
        if not sequence:
            return PlanDecision(action="abort", message=f"暂不支持的流程：{mode}")

        if transcript:
            last_record = transcript[-1]
            last_command = str(last_record.get("command", "")).strip()
            last_exit = int(last_record.get("exit_code", 0))
            last_stderr = str(last_record.get("stderr", "")).strip()
            if last_exit != 0 and last_command:
                for step in sequence:
                    if self._normalize_command(step.command) == self._normalize_command(last_command):
                        if step.command.startswith("test -f "):
                            return PlanDecision(action="need_input", message=f"目标文件不存在：{resolved_file}")
                        return PlanDecision(action="abort", message=last_stderr or "上一步执行失败，流程已停止")

        for step in sequence:
            if not self._command_seen(step.command, transcript):
                return PlanDecision(action="next", command=step.command)
        return PlanDecision(action="done", message="执行完成")

    def resolve_placeholders(self, command: str, transcript: list[dict[str, Any]], task: TaskSpec) -> tuple[bool, str, str]:
        _ = transcript
        _ = task
        if self._contains_placeholder(command):
            return False, "", "命令包含未解析占位符，请先补充缺失信息"
        return True, command, ""

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
        _ = transcript
        _ = task
        if not suggested_steps:
            return None
        raw_next = suggested_steps.pop(0).strip()
        if not raw_next:
            return None
        if self._contains_placeholder(raw_next):
            return PlanDecision(action="need_input", message="步骤草案包含未解析信息，请先补充必要参数")
        valid, validation_message = self.validate_ai_code_command(raw_next)
        if not valid:
            return PlanDecision(action="abort", message=validation_message)
        return PlanDecision(action="next", command=raw_next, message="")
