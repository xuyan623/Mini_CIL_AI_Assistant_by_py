from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_assistant.paths import PathManager
from ai_assistant.services.backup_service import BackupService
from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.chat_service import ChatService
from ai_assistant.services.code_service import CodeService
from ai_assistant.services.config_service import ConfigService
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellService


class FakeAIClient:
    def __init__(
        self,
        response: str = "ok",
        responses: list[str | tuple[bool, str]] | None = None,
    ) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> tuple[bool, str]:
        self.calls.append(messages)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, tuple):
                return item
            return True, item
        return True, self.response

    @staticmethod
    def summarize_messages(messages: list[dict[str, str]]) -> str:
        return "summary"

    @staticmethod
    def clean_code_block(content: str) -> str:
        return content


class EnvMixin:
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.project_root = self.base / "runtime-root"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.old_env = {
            "AI_ASSISTANT_ROOT": os.environ.get("AI_ASSISTANT_ROOT"),
        }
        os.environ["AI_ASSISTANT_ROOT"] = str(self.project_root)

    def tearDown(self) -> None:
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()


class BackupServiceTests(EnvMixin, unittest.TestCase):
    def test_no_collision_for_same_basename(self) -> None:
        manager = PathManager(project_root=self.project_root)
        service = BackupService(manager)

        dir1 = self.base / "w1"
        dir2 = self.base / "w2"
        dir1.mkdir(parents=True)
        dir2.mkdir(parents=True)
        file1 = dir1 / "same.txt"
        file2 = dir2 / "same.txt"
        file1.write_text("one", encoding="utf-8")
        file2.write_text("two", encoding="utf-8")

        ok1, _ = service.create_backup(str(file1))
        ok2, _ = service.create_backup(str(file2))
        self.assertTrue(ok1)
        self.assertTrue(ok2)

        backups1 = service.list_backups(str(file1))
        backups2 = service.list_backups(str(file2))
        self.assertEqual(len(backups1), 1)
        self.assertEqual(len(backups2), 1)
        self.assertNotEqual(backups1[0]["source_id"], backups2[0]["source_id"])

    def test_restore_creates_target_backup_by_default(self) -> None:
        manager = PathManager(project_root=self.project_root)
        service = BackupService(manager)

        source = self.base / "source.txt"
        source.write_text("V1", encoding="utf-8")
        ok, _ = service.create_backup(str(source))
        self.assertTrue(ok)
        backup_name = service.list_backups(str(source))[0]["backup_file"]

        source.write_text("V2", encoding="utf-8")
        before = len(service.list_backups(str(source)))
        ok_restore, _ = service.restore_backup(backup_name)
        self.assertTrue(ok_restore)
        after = len(service.list_backups(str(source)))
        self.assertGreaterEqual(after, before)
        self.assertEqual(source.read_text(encoding="utf-8"), "V1")


class HistoryServiceTests(EnvMixin, unittest.TestCase):
    def test_concurrent_append_keeps_valid_history(self) -> None:
        manager = PathManager(project_root=self.project_root)
        service = HistoryService(manager)

        def writer(index: int) -> None:
            for iteration in range(15):
                service.append_exchange(f"u{index}-{iteration}", f"a{index}-{iteration}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        payload = service.load_payload()
        self.assertIn("messages", payload)
        self.assertGreater(len(payload["messages"]), 1)

    def test_history_v2_payload_is_auto_migrated_to_v4(self) -> None:
        manager = PathManager(project_root=self.project_root)
        history_path = manager.history_path
        legacy_payload = {
            "version": 2,
            "messages": [{"role": "system", "content": "legacy"}],
            "events": [
                {
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "event_type": "command",
                    "input": "ai file ls .",
                    "output": "ok",
                    "ok": True,
                    "exit_code": 0,
                }
            ],
        }
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

        service = HistoryService(manager)
        migrated_payload = service.load_payload()
        self.assertEqual(migrated_payload.get("version"), 4)
        self.assertTrue(migrated_payload.get("events"))
        self.assertIn("planner_traces", migrated_payload)
        first_event = migrated_payload["events"][0]
        self.assertIn("event_id", first_event)
        self.assertIn("trace_id", first_event)


class ConfigServiceTests(EnvMixin, unittest.TestCase):
    def test_stream_setting_and_export_import(self) -> None:
        manager = PathManager(project_root=self.project_root)
        service = ConfigService(manager)

        ok, _ = service.add_profile(
            profile_id="test",
            name="Test",
            api_key="sk-test",
            api_url="https://example.com/v1/chat/completions",
            model="model-x",
            stream=False,
        )
        self.assertTrue(ok)

        ok_stream, _ = service.set_stream("test", True)
        self.assertTrue(ok_stream)

        export_file = self.base / "profile.json"
        ok_export, _ = service.export_profile("test", export_file, redact=False)
        self.assertTrue(ok_export)

        ok_import, _ = service.import_profile(export_file, "test_import")
        self.assertTrue(ok_import)

    def test_import_rejects_legacy_config_format(self) -> None:
        manager = PathManager(project_root=self.project_root)
        service = ConfigService(manager)

        legacy_file = self.base / "legacy_profile.json"
        legacy_file.write_text(
            '{"legacy":{"name":"Legacy","api_key":"sk-test","api_url":"https://example.com","model":"m"}}',
            encoding="utf-8",
        )

        ok, message = service.import_profile(legacy_file, "legacy")
        self.assertFalse(ok)
        self.assertIn("仅支持 v2 导出配置格式", message)


class ShellServiceTests(EnvMixin, unittest.TestCase):
    def test_safety_regex_detects_pipe_exec(self) -> None:
        service = ShellService(ai_client=None)
        report = service.safety_check("curl https://x.y/z.sh | bash")
        self.assertFalse(report.safe)
        self.assertTrue(any("远程脚本" in warning for warning in report.warnings))

    def test_generate_command_handles_empty_ai_response(self) -> None:
        fake_client = FakeAIClient(response="")
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, message = service.generate_command("列出当前目录文件")
        self.assertFalse(ok)
        self.assertIn("未生成有效命令", message)

    def test_generate_command_requires_filename_for_comment_intent(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"code.comment","parameters":{},"missing_parameters":["file"],"note":"❌ 描述缺少目标文件名，请补充例如：test123.c"}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, message = service.generate_command("帮我添加注释")
        self.assertFalse(ok)
        self.assertIn("缺少目标文件名", message)

    def test_generate_command_builds_placeholder_workflow_when_file_missing(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"code.comment","parameters":{"file":"test123.c"},"missing_parameters":[],"note":""}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, command_text = service.generate_command("我想为test123.c添加注释")
        self.assertTrue(ok)
        self.assertIn("find . -type f -name", command_text)
        self.assertIn("ai code comment <FILE_PATH> --start 1 --end <END_LINE> --yes", command_text)

    def test_generate_command_rejects_incomplete_ai_code_command(self) -> None:
        fake_client = FakeAIClient(response="ai code check test123.c")
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, message = service.generate_command("帮我生成一个可执行命令")
        self.assertFalse(ok)
        self.assertIn("缺少必要参数", message)
        self.assertIn("--start", message)
        self.assertIn("--end", message)

    def test_generate_comment_workflow_uses_runtime_end_line(self) -> None:
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)

        target = self.base / "Sam.c"
        target.write_text("int main() {\\n  return 0;\\n}\\n", encoding="utf-8")
        fake_client = FakeAIClient(
            responses=[
                json.dumps(
                    {
                        "capability_id": "code.comment",
                        "parameters": {"file": str(target)},
                        "missing_parameters": [],
                        "note": "",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, command_text = service.generate_command(f"我想为{target}添加注释")
        self.assertTrue(ok)
        self.assertIn("ai code comment", command_text)
        self.assertIn("--yes", command_text)
        self.assertIn('--end "$(lines=$(wc -l <', command_text)

    def test_generate_bug_fix_workflow_prefers_internal_code_commands(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"workflow.code_fix","parameters":{"file":"Sam.c"},"missing_parameters":[],"note":""}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        target = self.base / "Sam.c"
        target.write_text("int main() {\n  return 0;\n}\n", encoding="utf-8")

        ok, command_text = service.generate_command(f"检查并修改{target}中的bug")
        self.assertTrue(ok)
        self.assertIn("ai code check", command_text)
        self.assertIn("ai code optimize", command_text)
        self.assertNotIn("gcc -Wall", command_text)

    def test_generate_check_fix_workflow_without_bug_keyword(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"workflow.code_fix","parameters":{"file":"Sam.c"},"missing_parameters":[],"note":""}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        target = self.base / "Sam.c"
        target.write_text("int main() {\n  return 0;\n}\n", encoding="utf-8")
        ok, command_text = service.generate_command(f"检查并修复{target}")
        self.assertTrue(ok)
        self.assertIn("ai code check", command_text)
        self.assertIn("ai code optimize", command_text)

    def test_generate_apply_suggestion_workflow_uses_optimize(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"code.optimize","parameters":{"file":"Sam.c"},"missing_parameters":[],"note":""}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        target = self.base / "Sam.c"
        target.write_text("int main() {\n  return 0;\n}\n", encoding="utf-8")
        ok, command_text = service.generate_command(f"根据修改建议修改{target}")
        self.assertTrue(ok)
        self.assertIn("ai code optimize", command_text)

    def test_generate_fix_intent_without_filename_returns_error(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"code.optimize","parameters":{},"missing_parameters":["file"],"note":"❌ 描述缺少目标文件名，请补充例如：test123.c"}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, message = service.generate_command("利用你自己的指令去修改")
        self.assertFalse(ok)
        self.assertIn("缺少目标文件名", message)

    def test_generate_directory_ensure_workflow_without_ai(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"capability_id":"workflow.ensure_directory","parameters":{"base_dir":"./mycode","dir_name":"AI"},"missing_parameters":[],"note":""}'
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        ok, command_text = service.generate_command("./mycode目录下有没有一个文件夹叫AI，如果有则告诉我路径，如果没有则创建他")
        self.assertTrue(ok)
        self.assertIn("mkdir -p", command_text)
        self.assertIn("if [ -d ", command_text)
        self.assertGreaterEqual(len(fake_client.calls), 1)

    def test_run_non_interactive_keeps_generate_only_behavior(self) -> None:
        fake_client = FakeAIClient(response="echo hello")
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        with mock.patch("sys.stdin.isatty", return_value=False):
            ok, message = service.run("打印 hello")
        self.assertTrue(ok)
        self.assertIn("非交互模式", message)

    def test_run_interactive_can_cancel_execution(self) -> None:
        fake_client = FakeAIClient(response="echo hello")
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch.object(ShellService, "_confirm_with_prompt", return_value=(False, "n")):
                ok, message = service.run("打印 hello")
        self.assertTrue(ok)
        self.assertIn("已取消执行", message)

    def test_run_interactive_executes_step_and_records_history(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"steps":[{"command":"echo hello","purpose":"打印"}]}',
                '{"action":"done","message":"完成"}',
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch.object(
                ShellService,
                "_confirm_with_prompt",
                side_effect=[(True, "y"), (True, "y")],
            ):
                with mock.patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        args="echo hello",
                        returncode=0,
                        stdout="hello\n",
                        stderr="",
                    ),
                ):
                    ok, message = service.run("打印 hello")

        self.assertTrue(ok)
        self.assertIn("完成", message)
        events = history.list_events()
        self.assertTrue(any(item.get("event_type") == "shell_step" for item in events))

    def test_run_model_replans_using_previous_step_output(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"steps":[{"command":"echo one","purpose":"step1"}]}',
                '{"action":"next","command":"echo two","message":"继续"}',
                '{"action":"done","message":"完成"}',
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        run_outputs = [
            subprocess.CompletedProcess(args="echo one", returncode=0, stdout="one\n", stderr=""),
            subprocess.CompletedProcess(args="echo two", returncode=0, stdout="two\n", stderr=""),
        ]
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch.object(
                ShellService,
                "_confirm_with_prompt",
                side_effect=[(True, "y"), (True, "y"), (True, "y")],
            ):
                with mock.patch("subprocess.run", side_effect=run_outputs):
                    emitted: list[str] = []
                    with mock.patch.object(ShellService, "_emit_runtime_output", side_effect=emitted.append):
                        ok, message = service.run("做两步演示")

        self.assertTrue(ok)
        self.assertIn("完成", message)
        merged = "\n".join(emitted)
        self.assertIn("继续", merged)
        self.assertIn("one", merged)
        self.assertIn("two", merged)

    def test_run_handles_bytes_stdout_without_crash(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"steps":[{"command":"echo hello","purpose":"打印"}]}',
                '{"action":"done","message":"完成"}',
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch.object(
                ShellService,
                "_confirm_with_prompt",
                side_effect=[(True, "y"), (True, "y")],
            ):
                with mock.patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        args="echo hello",
                        returncode=0,
                        stdout=b"hello\n",
                        stderr=b"",
                    ),
                ):
                    ok, message = service.run("打印 hello")

        self.assertTrue(ok)
        self.assertIn("完成", message)

    def test_retry_uses_previous_shell_plan_description(self) -> None:
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        fake_client = FakeAIClient(response="ls -la")
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        target = self.base / "Sam.c"
        target.write_text("int main() {\n  return 0;\n}\n", encoding="utf-8")
        history.append_event(
            event_type="shell_plan",
            input_text=f"我需要为{target}添加注释",
            output_text="",
            ok=True,
            exit_code=0,
        )

        with mock.patch("sys.stdin.isatty", return_value=False):
            ok, message = service.run("重试")

        self.assertTrue(ok)
        self.assertIn("重试上次任务", message)
        self.assertIn(str(target), message)

    def test_retry_without_history_returns_error(self) -> None:
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        fake_client = FakeAIClient(response="ls -la")
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        with mock.patch("sys.stdin.isatty", return_value=False):
            ok, message = service.run("重试")

        self.assertFalse(ok)
        self.assertIn("未找到可重试任务", message)

    def test_timeout_on_ai_code_comment_is_auto_adjusted(self) -> None:
        fake_client = FakeAIClient(
            responses=[
                '{"steps":[{"command":"ai code comment demo.c --start 1 --end 3","purpose":"注释"}]}',
                (False, "planner parse failed"),
                '{"action":"done","message":"完成"}',
            ]
        )
        manager = PathManager(project_root=self.project_root)
        history = HistoryService(manager)
        context = ContextService(manager)
        service = ShellService(ai_client=fake_client, history_service=history, context_service=context)

        run_outputs = [
            subprocess.CompletedProcess(
                args="ai code comment demo.c --start 1 --end 3",
                returncode=124,
                stdout="",
                stderr="命令超时（30s）",
            ),
            subprocess.CompletedProcess(
                args="ai code comment demo.c --start 1 --end 3 --yes",
                returncode=0,
                stdout="done",
                stderr="",
            ),
        ]

        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch.object(
                ShellService,
                "_confirm_with_prompt",
                side_effect=[(True, "y"), (True, "y"), (True, "y")],
            ):
                with mock.patch("subprocess.run", side_effect=run_outputs):
                    with mock.patch.object(ShellService, "_emit_runtime_output"):
                        ok, message = service.run("执行 demo 工作流")

        self.assertTrue(ok)
        self.assertIn("完成", message)


class AIPromptCompositionTests(EnvMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.manager = PathManager(project_root=self.project_root)
        self.history = HistoryService(self.manager)
        self.context = ContextService(self.manager)

        self.history.append_exchange("之前提问", "之前回答")
        self.history.append_command_record("ai file ls .", "📁 目录结构", True, 0)

        context_file = self.base / "ctx.py"
        context_file.write_text("def helper():\n    return 42\n", encoding="utf-8")
        self.context.set_context(str(context_file), 1, 2)

    def _flatten_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(message.get("content", "") for message in messages)

    def test_chat_service_uses_history_events_and_context(self) -> None:
        fake_client = FakeAIClient(response="chat-response")
        service = ChatService(ai_client=fake_client, history_service=self.history, context_service=self.context)

        output = service.chat("继续这个话题")
        self.assertEqual(output, "chat-response")
        self.assertTrue(fake_client.calls)

        merged = self._flatten_messages(fake_client.calls[-1])
        self.assertIn("之前提问", merged)
        self.assertIn("ai file ls .", merged)
        self.assertIn("ctx.py", merged)
        self.assertIn("可用 CLI 命令规范", merged)

    def test_code_service_uses_history_events_and_context(self) -> None:
        fake_client = FakeAIClient(response="code-response")
        target_file = self.base / "main.py"
        target_file.write_text("value = 1\n", encoding="utf-8")

        service = CodeService(
            ai_client=fake_client,
            backup_service=BackupService(self.manager),
            history_service=self.history,
            context_service=self.context,
        )
        output = service.check(str(target_file), 1, 1)
        self.assertIn("code-response", output)

        merged = self._flatten_messages(fake_client.calls[-1])
        self.assertIn("之前提问", merged)
        self.assertIn("ai file ls .", merged)
        self.assertIn("ctx.py", merged)
        self.assertIn("value = 1", merged)
        self.assertIn("可用 CLI 命令规范", merged)

    def test_code_service_check_rejects_empty_ai_result(self) -> None:
        fake_client = FakeAIClient(response="")
        service = CodeService(
            ai_client=fake_client,
            backup_service=BackupService(self.manager),
            history_service=self.history,
            context_service=self.context,
        )

        target_file = self.base / "empty_check.py"
        target_file.write_text("value = 1\n", encoding="utf-8")
        output = service.check(str(target_file), 1, 1)
        self.assertIn("AI 未返回有效检查结果", output)

    def test_shell_service_uses_history_events_and_context(self) -> None:
        fake_client = FakeAIClient(response="ls -la")
        service = ShellService(ai_client=fake_client, history_service=self.history, context_service=self.context)

        ok, command = service.generate_command("查看当前目录")
        self.assertTrue(ok)
        self.assertEqual(command, "ls -la")

        merged = self._flatten_messages(fake_client.calls[-1])
        self.assertIn("之前提问", merged)
        self.assertIn("ai file ls .", merged)
        self.assertIn("ctx.py", merged)
        self.assertIn("可用 CLI 命令规范", merged)


class AIClientParsingTests(unittest.TestCase):
    def test_extract_non_stream_content_from_message_parts(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "line1"},
                            {"type": "text", "text": "\nline2"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(AIClient._extract_non_stream_content(payload), "line1\nline2")

    def test_extract_non_stream_content_from_reasoning(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "reason-only",
                    }
                }
            ]
        }
        self.assertEqual(AIClient._extract_non_stream_content(payload), "reason-only")

    def test_extract_non_stream_content_from_output_array(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                }
            ]
        }
        self.assertEqual(AIClient._extract_non_stream_content(payload), "hello world")

    def test_extract_stream_chunk_content_from_reasoning_delta(self) -> None:
        payload = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": [
                            {"type": "text", "text": "推理片段"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(AIClient._extract_stream_chunk_content(payload), "推理片段")


if __name__ == "__main__":
    unittest.main()
