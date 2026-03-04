from __future__ import annotations

from pathlib import Path

import pytest

from ai_assistant.paths import PathManager
from ai_assistant.services.context_service import ContextService


def test_set_add_list_clear_and_get_files(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ContextService(manager)
    target = tmp_path / "a.py"
    target.write_text("a=1\nb=2\n", encoding="utf-8")

    set_result = service.set_context(str(target), 1, 2)
    assert "已设置上下文" in set_result

    duplicate_add = service.add_context(str(target), 1, 2)
    assert "已在上下文中" in duplicate_add

    second = tmp_path / "b.py"
    second.write_text("x=1\n", encoding="utf-8")
    add_result = service.add_context(str(second), None, None)
    assert "已追加上下文" in add_result

    listed = service.list_context()
    assert "当前代码上下文" in listed
    assert "a.py" in listed and "b.py" in listed

    files = service.get_context_files()
    assert len(files) == 2

    clear_result = service.clear_context()
    assert "已清除代码上下文" in clear_result
    assert service.get_context_files() == []


def test_read_file_content_and_invalid_ranges(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ContextService(manager)
    target = tmp_path / "a.py"
    target.write_text("a\nb\n", encoding="utf-8")

    full, full_range = service._read_file_content(target, None, None)
    assert full_range == "全文"
    assert full == "a\nb\n"

    snippet, range_info = service._read_file_content(target, 1, 2)
    assert range_info == "行 1-2"
    assert snippet == "a\nb\n"

    with pytest.raises(ValueError):
        service._read_file_content(target, 2, 1)
    with pytest.raises(FileNotFoundError):
        service._read_file_content(tmp_path / "missing.py", 1, 1)


def test_render_context_block_and_build_prompt(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ContextService(manager)
    target = tmp_path / "ctx.py"
    target.write_text("print('x')\n", encoding="utf-8")
    service.set_context(str(target), 1, 1)

    block = service.render_context_block()
    assert "代码上下文" in block
    assert "ctx.py" in block

    truncated = service.render_context_block(max_chars=20)
    assert "上下文已截断" in truncated

    prompt = service.build_prompt("解释", recent_messages=[{"role": "user", "content": "hi"}])
    assert "最近对话" in prompt
    assert "解释" in prompt

    service.clear_context()
    with pytest.raises(ValueError):
        service.build_prompt("x")

