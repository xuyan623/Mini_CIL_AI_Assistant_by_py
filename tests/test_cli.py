from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_ENTRY = REPO_ROOT / "ai.py"
ROOT_DIR = REPO_ROOT


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.runtime_root = self.base / "runtime-root"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.old_ai_assistant_root = os.environ.get("AI_ASSISTANT_ROOT")
        os.environ["AI_ASSISTANT_ROOT"] = str(self.runtime_root)
        self.env = os.environ.copy()
        self.env["AI_ASSISTANT_ROOT"] = str(self.runtime_root)

    def tearDown(self) -> None:
        if self.old_ai_assistant_root is None:
            os.environ.pop("AI_ASSISTANT_ROOT", None)
        else:
            os.environ["AI_ASSISTANT_ROOT"] = self.old_ai_assistant_root
        self.temp_dir.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AI_ENTRY), *args],
            cwd=str(ROOT_DIR),
            env=self.env,
            text=True,
            capture_output=True,
        )

    def load_history_payload(self) -> dict:
        history_path = self.runtime_root / "assistant-state" / "history.json"
        return json.loads(history_path.read_text(encoding="utf-8"))

    def test_legacy_command_migration_message(self) -> None:
        result = self.run_cli("/ls")
        self.assertEqual(result.returncode, 2)
        self.assertIn("旧命令语法已下线", result.stdout)

    def test_file_search_with_space_keyword(self) -> None:
        sample = self.base / "sample.txt"
        sample.write_text("hello world\nhello alpine\n", encoding="utf-8")

        result = self.run_cli("file", "search", str(sample), "hello world")
        self.assertEqual(result.returncode, 0)
        self.assertIn("匹配1处", result.stdout)

    def test_config_stream_command(self) -> None:
        add = self.run_cli(
            "config",
            "add",
            "--profile",
            "demo",
            "--name",
            "Demo",
            "--api-key",
            "sk-demo",
            "--api-url",
            "https://example.com/v1/chat/completions",
            "--model",
            "demo-model",
            "--stream",
            "off",
        )
        self.assertEqual(add.returncode, 0)

        stream = self.run_cli("config", "stream", "demo", "on")
        self.assertEqual(stream.returncode, 0)
        self.assertIn("启用流式", stream.stdout)

    def test_parse_error_is_recorded_in_history_events(self) -> None:
        result = self.run_cli("config", "stream", "demo")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stdout)

        payload = self.load_history_payload()
        self.assertIn("events", payload)
        self.assertGreaterEqual(len(payload["events"]), 1)
        latest = payload["events"][-1]
        self.assertEqual(latest.get("event_type"), "command")
        self.assertIn("ai config stream demo", latest.get("input", ""))
        self.assertIn("error:", latest.get("output", ""))
        self.assertEqual(latest.get("ok"), False)
        self.assertEqual(latest.get("exit_code"), 2)

    def test_service_error_is_recorded_in_history_events(self) -> None:
        result = self.run_cli("file", "read", str(self.base / "missing.txt"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("❌ 无效文件", result.stdout)

        payload = self.load_history_payload()
        latest = payload["events"][-1]
        self.assertIn("ai file read", latest.get("input", ""))
        self.assertIn("❌ 无效文件", latest.get("output", ""))
        self.assertEqual(latest.get("ok"), False)
        self.assertEqual(latest.get("exit_code"), 1)

    def test_config_help_works(self) -> None:
        result = self.run_cli("config", "-h")
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: ai config", result.stdout)

    def test_execute_flag_is_removed(self) -> None:
        result = self.run_cli("--execute")
        self.assertEqual(result.returncode, 2)
        self.assertIn("已下线", result.stdout)

    def test_shell_run_help_no_execute_flag(self) -> None:
        result = self.run_cli("shell", "run", "-h")
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: ai shell run", result.stdout)
        self.assertNotIn("--execute", result.stdout)

    def test_shell_run_rejects_removed_execute_flag(self) -> None:
        result = self.run_cli("shell", "run", "查找大文件", "--execute")
        self.assertEqual(result.returncode, 2)
        self.assertIn("已下线", result.stdout)

    def test_code_comment_help_has_yes_flag(self) -> None:
        result = self.run_cli("code", "comment", "-h")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--yes", result.stdout)

    def test_context_ask_includes_cli_command_rules_prompt(self) -> None:
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))

        import ai_assistant.cli as cli_module  # pylint: disable=import-outside-toplevel

        ctx = cli_module.AppContext()
        context_file = self.base / "ctx.py"
        context_file.write_text("def f():\n    return 1\n", encoding="utf-8")
        ctx.context_service.set_context(str(context_file), 1, 2)

        captured: dict[str, object] = {}

        def fake_chat(messages: list[dict[str, str]], **kwargs: object) -> tuple[bool, str]:
            captured["messages"] = messages
            return True, "ok"

        args = cli_module.argparse.Namespace(action="ask", question="这段代码做了什么")
        with mock.patch.object(ctx.ai_client, "chat", side_effect=fake_chat):
            result = cli_module._handle_context(args, ctx)

        self.assertTrue(result.ok)
        merged = "\n".join(message.get("content", "") for message in captured.get("messages", []))
        self.assertIn("可用 CLI 命令规范", merged)

    def test_retry_alias_routes_to_shell_run(self) -> None:
        result = self.run_cli("重试")
        self.assertEqual(result.returncode, 1)
        self.assertIn("未找到可重试任务", result.stdout)


if __name__ == "__main__":
    unittest.main()
