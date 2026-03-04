from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant.services.file_service import FileService


@pytest.fixture
def file_service() -> FileService:
    return FileService()


def test_list_directory_success_and_invalid(file_service: FileService, tmp_path: Path) -> None:
    (tmp_path / "dir_a").mkdir()
    (tmp_path / "dir_a" / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")

    result = file_service.list_directory(str(tmp_path))
    assert "目录结构" in result
    assert "dir_a/" in result
    assert "a.txt" in result

    missing = file_service.list_directory(str(tmp_path / "missing"))
    assert "目录不存在" in missing
    not_dir = file_service.list_directory(str(tmp_path / "z.txt"))
    assert "非目录" in not_dir


def test_read_file_truncated_and_invalid(file_service: FileService, tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("x" * 40, encoding="utf-8")
    ok_text = file_service.read_file(str(target), max_chars=20)
    assert "内容已截断" in ok_text

    invalid = file_service.read_file(str(tmp_path / "missing.txt"))
    assert "无效文件" in invalid


def test_search_file_and_find_files(file_service: FileService, tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("print('hello')\nprint('world')\n", encoding="utf-8")

    found = file_service.search_file(str(target), "hello")
    assert "匹配1处" in found
    not_found = file_service.search_file(str(target), "missing")
    assert "无匹配" in not_found
    empty_keyword = file_service.search_file(str(target), "")
    assert "关键词不能为空" in empty_keyword

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hello_data.csv").write_text("1,2\n", encoding="utf-8")
    find_result = file_service.find_files("hello", str(tmp_path))
    assert "找到" in find_result
    assert "hello_data.csv" in find_result
    invalid_dir = file_service.find_files("x", str(tmp_path / "missing"))
    assert "无效目录" in invalid_dir
    empty = file_service.find_files("", str(tmp_path))
    assert "关键词不能为空" in empty


def test_remove_file_and_directory_paths(file_service: FileService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "delete_me.txt"
    file_path.write_text("bye", encoding="utf-8")
    dir_path = tmp_path / "to_remove"
    dir_path.mkdir()
    (dir_path / "x.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    cancelled = file_service.remove_file(str(file_path), force=True)
    assert "已取消" in cancelled
    assert file_path.exists()

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    removed = file_service.remove_file(str(file_path), force=True)
    assert "已删除文件" in removed
    assert not file_path.exists()

    file_path.write_text("bye", encoding="utf-8")
    monkeypatch.setattr(file_service, "_is_sensitive", lambda _path: True)
    sensitive_file = file_service.remove_file(str(file_path), force=False)
    assert "敏感路径" in sensitive_file

    monkeypatch.setattr(file_service, "_is_sensitive", lambda _path: False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    file_service.remove_file(str(file_path), force=True)

    monkeypatch.setattr(file_service, "_is_sensitive", lambda _path: True)
    sensitive = file_service.remove_directory(str(dir_path), force=False)
    assert "敏感路径" in sensitive

    monkeypatch.setattr(file_service, "_is_sensitive", lambda _path: False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    dir_removed = file_service.remove_directory(str(dir_path), force=True)
    assert "已删除目录" in dir_removed
    assert not dir_path.exists()


def test_remove_invalid_targets_and_text_detection(file_service: FileService, tmp_path: Path) -> None:
    missing_file = file_service.remove_file(str(tmp_path / "none.txt"), force=True)
    assert "无效文件" in missing_file

    missing_dir = file_service.remove_directory(str(tmp_path / "none"), force=True)
    assert "无效目录" in missing_dir

    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"\x00\xff\x10")
    assert not file_service._is_text_file(binary)


def test_misc_file_service_branches(file_service: FileService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved_default = file_service._resolve(None)
    assert str(resolved_default)

    root_dir = tmp_path / "perm"
    root_dir.mkdir()
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda self: (_ for _ in ()).throw(PermissionError()),  # noqa: ARG005
    )
    perm_result = file_service.list_directory(str(root_dir))
    assert "Permission Denied" in perm_result

    target = tmp_path / "x.dat"
    target.write_text("text", encoding="utf-8")
    monkeypatch.setattr(file_service, "_is_text_file", lambda _p: False)
    non_text = file_service.read_file(str(target))
    assert "非文本文件" in non_text
