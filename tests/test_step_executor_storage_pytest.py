from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ai_assistant.planner.step_executor import StepExecutor
from ai_assistant.storage import atomic_write_json, atomic_write_text, file_lock, safe_load_json


def test_step_executor_success_timeout_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = StepExecutor(timeout_seconds=1)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args="echo hi", returncode=0, stdout="hi\n", stderr=""),
    )
    success = executor.execute("echo hi")
    assert success.ok is True
    assert success.stdout.strip() == "hi"

    def _raise_timeout(*args, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd="cmd", timeout=1, output=b"partial")

    monkeypatch.setattr("subprocess.run", _raise_timeout)
    timeout = executor.execute("sleep 5")
    assert timeout.ok is False
    assert timeout.exit_code == 124
    assert "命令超时" in timeout.stderr

    def _raise_error(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr("subprocess.run", _raise_error)
    failed = executor.execute("bad")
    assert failed.ok is False
    assert failed.exit_code == 1
    assert "boom" in failed.stderr


def test_storage_helpers(tmp_path: Path) -> None:
    text_file = tmp_path / "a.txt"
    json_file = tmp_path / "a.json"
    lock_file = tmp_path / "a.lock"

    atomic_write_text(text_file, "hello")
    assert text_file.read_text(encoding="utf-8") == "hello"

    atomic_write_json(json_file, {"k": 1})
    assert json.loads(json_file.read_text(encoding="utf-8"))["k"] == 1

    loaded = safe_load_json(json_file, {})
    assert loaded["k"] == 1

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    fallback = safe_load_json(bad_json, {"x": 1})
    assert fallback["x"] == 1

    with file_lock(lock_file):
        assert lock_file.exists()
    assert not lock_file.exists()

