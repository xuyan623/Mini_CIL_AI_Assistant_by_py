from __future__ import annotations

import builtins
import os
import shutil
from pathlib import Path

import pytest

from ai_assistant.services.file_service import FileService


def test_sensitive_and_confirm_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = FileService()
    assert service._is_sensitive(Path("/etc")) is True

    monkeypatch.setattr("ai_assistant.services.file_service.os.name", "nt")
    monkeypatch.setattr("ai_assistant.services.file_service.os.path.splitdrive", lambda value: ("C:", value[2:]))
    assert service._is_sensitive(Path("C:/")) is True

    monkeypatch.setattr(builtins, "input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    assert service._confirm("x") is False


def test_read_search_remove_exception_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = FileService()
    target = tmp_path / "demo.txt"
    target.write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(service, "_is_text_file", lambda _path: True)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": (_ for _ in ()).throw(PermissionError()))
    read_perm = service.read_file(str(target))
    assert "无权限读取" in read_perm

    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": (_ for _ in ()).throw(RuntimeError("boom")))
    read_fail = service.read_file(str(target))
    assert "读取失败" in read_fail

    class _BrokenOpen:
        def __enter__(self):  # noqa: ANN001,D401
            raise RuntimeError("open failed")

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001,D401
            return False

    monkeypatch.setattr(Path, "open", lambda self, mode="r", encoding="utf-8": _BrokenOpen())
    search_fail = service.search_file(str(target), "hello")
    assert "搜索失败" in search_fail

    monkeypatch.setattr(service, "_confirm", lambda _prompt: True)
    monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(RuntimeError("unlink failed")))
    remove_fail = service.remove_file(str(target), force=True)
    assert "删除失败" in remove_fail

    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.setattr(shutil, "rmtree", lambda path: (_ for _ in ()).throw(RuntimeError("rmtree failed")))
    remove_dir_fail = service.remove_directory(str(folder), force=True)
    assert "删除失败" in remove_dir_fail


def test_find_files_empty_result(tmp_path: Path) -> None:
    service = FileService()
    directory = tmp_path / "search"
    directory.mkdir()
    (directory / "one.py").write_text("print(1)\n", encoding="utf-8")
    result = service.find_files("not-found", str(directory))
    assert "未找到包含" in result


def test_sensitive_prefix_with_os_name_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FileService()
    monkeypatch.setattr("ai_assistant.services.file_service.os.name", "posix")
    assert service._is_sensitive(Path("/var/log")) is True
