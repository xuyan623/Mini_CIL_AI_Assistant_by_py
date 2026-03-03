from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_assistant.paths import PathManager
from ai_assistant.services.backup_service import BackupService
from ai_assistant.services.chat_service import ChatService
from ai_assistant.services.code_service import CodeService
from ai_assistant.services.config_service import ConfigService
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellService


class FakeAIClient:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> tuple[bool, str]:
        self.calls.append(messages)
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


if __name__ == "__main__":
    unittest.main()
