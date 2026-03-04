from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_assistant.paths import PathManager
from ai_assistant.services.code_service import CodeService
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService


class DummyBackupService:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def create_backup(self, _file: str) -> tuple[bool, str]:
        if self.ok:
            return True, "backup ok"
        return False, "backup failed"


class DummyAIClient:
    @staticmethod
    def clean_code_block(content: str) -> str:
        return content.strip()


def _make_service(tmp_path: Path, backup_ok: bool = True) -> CodeService:
    manager = PathManager(project_root=tmp_path)
    history = HistoryService(manager)
    context = ContextService(manager)
    return CodeService(
        ai_client=DummyAIClient(),  # type: ignore[arg-type]
        backup_service=DummyBackupService(ok=backup_ok),  # type: ignore[arg-type]
        history_service=history,
        context_service=context,
    )


def test_check_explain_summarize_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_service(tmp_path)
    target = tmp_path / "main.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.2: (True, "analysis ok"))
    check_result = service.check(str(target), 1, 1)
    assert "代码检查结果" in check_result
    assert "analysis ok" in check_result

    explain_result = service.explain(str(target), 1, 1)
    assert "代码解释结果" in explain_result

    summarize_result = service.summarize(str(target))
    assert "文件总结" in summarize_result


def test_check_handles_empty_or_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_service(tmp_path)
    target = tmp_path / "main.c"
    target.write_text("int main(){return 0;}\n", encoding="utf-8")

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.2: (True, ""))
    empty = service.check(str(target), 1, 1)
    assert "未返回有效检查结果" in empty

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.2: (False, "err"))
    failed = service.explain(str(target), 1, 1)
    assert failed == "err"


def test_comment_and_optimize_write_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_service(tmp_path, backup_ok=True)
    target = tmp_path / "main.c"
    target.write_text("int main(){\nreturn 0;\n}\n", encoding="utf-8")

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.2: (True, "int main(){\nreturn 1;\n}"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    comment_result = service.comment(str(target), 1, 3)
    assert "文件已更新" in comment_result
    assert "return 1;" in target.read_text(encoding="utf-8")

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.2: (True, "int main(){\nreturn 2;\n}"))
    optimize_result = service.optimize(str(target), 1, 3, yes=True)
    assert "文件已更新" in optimize_result
    assert "return 2;" in target.read_text(encoding="utf-8")


def test_modify_range_cancel_backup_failure_and_empty_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "main.c"
    target.write_text("int main(){\nreturn 0;\n}\n", encoding="utf-8")

    service_cancel = _make_service(tmp_path, backup_ok=True)
    monkeypatch.setattr(service_cancel, "_ask_ai", lambda _prompt, temperature=0.2: (True, "int main(){\nreturn 0;\n}"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    cancel_result = service_cancel.comment(str(target), 1, 3)
    assert "已取消" in cancel_result

    service_backup_fail = _make_service(tmp_path, backup_ok=False)
    monkeypatch.setattr(service_backup_fail, "_ask_ai", lambda _prompt, temperature=0.2: (True, "int main(){\nreturn 3;\n}"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    backup_failed = service_backup_fail.optimize(str(target), 1, 3)
    assert "backup failed" in backup_failed

    service_empty = _make_service(tmp_path, backup_ok=True)
    monkeypatch.setattr(service_empty, "_ask_ai", lambda _prompt, temperature=0.2: (True, ""))
    empty_code = service_empty.comment(str(target), 1, 3, yes=True)
    assert "未返回有效代码" in empty_code


def test_generate_insert_replace_and_invalid_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_service(tmp_path, backup_ok=True)
    target = tmp_path / "gen.c"
    target.write_text("line1\nline2\n", encoding="utf-8")

    monkeypatch.setattr(service, "_ask_ai", lambda _prompt, temperature=0.4: (True, "new_line"))
    generated = service.generate(str(target), 1, 1, "insert", yes=True)
    assert "写入完成" in generated
    assert "new_line" in target.read_text(encoding="utf-8")

    invalid = service.generate(str(target), 10, 12, "invalid", yes=True)
    assert "行号无效" in invalid


def test_ask_ai_empty_content_and_failures(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service.ai_gateway = SimpleNamespace(  # type: ignore[assignment]
        summarize_messages=lambda _messages: "",
        chat=lambda *args, **kwargs: SimpleNamespace(ok=False, content="", error_code="empty_content"),
    )
    ok, content = service._ask_ai("prompt")
    assert ok is True
    assert content == ""

    service.ai_gateway = SimpleNamespace(  # type: ignore[assignment]
        summarize_messages=lambda _messages: "",
        chat=lambda *args, **kwargs: SimpleNamespace(ok=False, content="boom", error_code="request_failed"),
    )
    ok2, content2 = service._ask_ai("prompt")
    assert ok2 is False
    assert content2 == "boom"


def test_code_service_file_and_slice_validation(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    ok, message, _path = service._read_text_file(str(tmp_path / "missing.py"))
    assert ok is False
    assert "无效文件" in message

    target = tmp_path / "slice.py"
    target.write_text("a\nb\n", encoding="utf-8")
    ok2, content, path = service._read_text_file(str(target))
    assert ok2 is True
    assert path.exists()

    valid, snippet, lines, total = service._slice_lines(content, 1, 2)
    assert valid is True
    assert snippet == "a\nb\n"
    assert total == 2
    assert len(lines) == 2

    invalid, error, _, _ = service._slice_lines(content, 2, 1)
    assert invalid is False
    assert "行号无效" in error


def test_preview_and_confirm_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert CodeService._preview_and_confirm("t", "c", auto_confirm=True) is True
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert CodeService._preview_and_confirm("t", "c", auto_confirm=False) is True
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert CodeService._preview_and_confirm("t", "c", auto_confirm=False) is False
