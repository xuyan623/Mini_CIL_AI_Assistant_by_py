from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.backup_service import BackupService


def test_create_list_status_restore_and_clean(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = BackupService(manager)
    source = tmp_path / "demo.txt"
    source.write_text("v1", encoding="utf-8")

    ok_create, msg_create = service.create_backup(str(source), keep=2)
    assert ok_create is True
    assert "备份完成" in msg_create

    items = service.list_backups(str(source))
    assert len(items) >= 1
    status = service.backup_status(str(source))
    assert "备份统计" in status

    backup_file = items[0]["backup_file"]
    source.write_text("v2", encoding="utf-8")
    ok_restore, msg_restore = service.restore_backup(backup_file)
    assert ok_restore is True
    assert "已恢复备份" in msg_restore
    assert source.read_text(encoding="utf-8") == "v1"

    ok_clean, msg_clean = service.clean_backups(str(source), keep=1)
    assert ok_clean is True
    assert "清理完成" in msg_clean


def test_backup_error_paths_and_filename_parse(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = BackupService(manager)

    ok_missing, msg_missing = service.create_backup(str(tmp_path / "missing.txt"))
    assert ok_missing is False
    assert "无效文件" in msg_missing

    parse_new = service._parse_backup_filename("abc123__file.py__20260304_120000_000001.bak")
    assert parse_new["source_id"] == "abc123"
    assert parse_new["source_path"] == "file.py"

    parse_legacy = service._parse_backup_filename("file.py.20260304_120000.bak")
    assert parse_legacy["source_id"].startswith("legacy-")

    parse_fallback = service._parse_backup_filename("unknown-format")
    assert parse_fallback["source_id"] == "legacy"

    missing_restore_ok, missing_restore_msg = service.restore_backup("missing.bak")
    assert missing_restore_ok is False
    assert "备份文件不存在" in missing_restore_msg

    no_target_backup = tmp_path / "orphan.bak"
    no_target_backup.write_text("x", encoding="utf-8")
    ok_no_target, msg_no_target = service.restore_backup(str(no_target_backup))
    assert ok_no_target is False
    assert "无法推断恢复目标路径" in msg_no_target

