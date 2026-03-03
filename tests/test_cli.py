from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ENTRY = REPO_ROOT / "root" / "ai.py"


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.runtime_root = self.base / "runtime-root"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env["AI_ASSISTANT_ROOT"] = str(self.runtime_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AI_ENTRY), *args],
            cwd=str(REPO_ROOT / "root"),
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


if __name__ == "__main__":
    unittest.main()
