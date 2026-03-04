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


def test_failure_paths_keep_no_traceback(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AI_ASSISTANT_ROOT"] = str(runtime_root)

    parse_error = _run(["config", "stream", "demo"], env)
    assert parse_error.returncode == 2
    assert "error:" in parse_error.stdout
    assert "Traceback" not in parse_error.stdout
    assert "Traceback" not in parse_error.stderr

    missing_file = _run(["file", "read", str(tmp_path / "missing.txt")], env)
    assert missing_file.returncode == 1
    assert "无效文件" in missing_file.stdout
    assert "Traceback" not in missing_file.stdout
    assert "Traceback" not in missing_file.stderr

