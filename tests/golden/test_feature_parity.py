from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ENTRY = PROJECT_ROOT / "ai.py"


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AI_ENTRY), *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )


def test_command_matrix_reachable(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AI_ASSISTANT_ROOT"] = str(runtime_root)
    sample = tmp_path / "a.txt"
    sample.write_text("hello\n", encoding="utf-8")

    cases = [
        ["-h"],
        ["file", "ls", "."],
        ["file", "read", str(sample)],
        ["code", "check", str(sample), "--start", "1", "--end", "1"],
        ["context", "set", str(sample), "--start", "1", "--end", "1"],
        ["context", "list"],
        ["backup", "create", str(sample)],
        ["backup", "status", str(sample)],
        ["config", "list"],
        ["shell", "run", "打印 hello"],
    ]

    for case in cases:
        result = _run(case, env)
        assert result.returncode in {0, 1, 2}
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr

