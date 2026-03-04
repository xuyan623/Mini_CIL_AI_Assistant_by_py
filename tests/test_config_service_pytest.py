from __future__ import annotations

import json
from pathlib import Path

from ai_assistant.paths import PathManager
from ai_assistant.services.config_service import ConfigService


def test_load_and_profile_lifecycle(tmp_path: Path, monkeypatch) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ConfigService(manager)

    payload = service.load_payload()
    assert payload["version"] == 2
    assert "deepseek" in payload["profiles"]

    ok_add, _ = service.add_profile(
        profile_id="demo",
        name="Demo",
        api_key="sk-demo",
        api_url="https://example.com/v1/chat/completions",
        model="demo-model",
        stream=False,
    )
    assert ok_add is True

    ids = service.list_profile_ids()
    assert "demo" in ids
    assert service.get_profile("demo") is not None
    assert service.get_profile("missing") is None

    ok_switch, _ = service.switch_profile("demo")
    assert ok_switch is True
    assert service.get_active_profile_id() == "demo"

    ok_stream, _ = service.set_stream("demo", True)
    assert ok_stream is True
    assert service.get_profile("demo").stream is True  # type: ignore[union-attr]

    ok_delete, _ = service.delete_profile("demo")
    assert ok_delete is True
    assert service.get_profile("demo") is None


def test_config_error_paths_and_import_export(tmp_path: Path) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ConfigService(manager)

    # duplicate add
    service.add_profile(
        profile_id="demo",
        name="Demo",
        api_key="sk-demo",
        api_url="https://example.com/v1/chat/completions",
        model="demo-model",
        stream=False,
    )
    duplicate_ok, duplicate_msg = service.add_profile(
        profile_id="demo",
        name="Demo",
        api_key="sk-demo",
        api_url="https://example.com/v1/chat/completions",
        model="demo-model",
        stream=False,
    )
    assert duplicate_ok is False
    assert "已存在" in duplicate_msg

    missing_switch_ok, _ = service.switch_profile("missing")
    assert missing_switch_ok is False

    missing_stream_ok, _ = service.set_stream("missing", True)
    assert missing_stream_ok is False

    export_file = tmp_path / "demo.profile.json"
    ok_export, _ = service.export_profile("demo", export_file, redact=True)
    assert ok_export is True
    exported = json.loads(export_file.read_text(encoding="utf-8"))
    assert exported["version"] == 2
    assert exported["profile"]["api_key"] == ""

    import_ok, import_msg = service.import_profile(export_file, "imported")
    assert import_ok is False
    assert "缺少 api_key" in import_msg

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not json", encoding="utf-8")
    bad_ok, bad_msg = service.import_profile(bad_file, "bad")
    assert bad_ok is False
    assert "无法解析配置文件" in bad_msg

    legacy_file = tmp_path / "legacy.json"
    legacy_file.write_text(json.dumps({"version": 1}), encoding="utf-8")
    legacy_ok, legacy_msg = service.import_profile(legacy_file, "legacy")
    assert legacy_ok is False
    assert "仅支持 v2 导出配置格式" in legacy_msg


def test_plaintext_warning_control(tmp_path: Path, monkeypatch, capsys) -> None:
    manager = PathManager(project_root=tmp_path)
    service = ConfigService(manager)
    payload = service._default_payload()
    payload["profiles"]["deepseek"]["api_key"] = "sk-abc"

    ConfigService._warned_plaintext = False
    monkeypatch.delenv("AI_SUPPRESS_PLAINTEXT_WARN", raising=False)
    service._warn_if_plaintext(payload)
    captured = capsys.readouterr()
    assert "明文 API Key" in captured.err

    ConfigService._warned_plaintext = False
    monkeypatch.setenv("AI_SUPPRESS_PLAINTEXT_WARN", "1")
    service._warn_if_plaintext(payload)
    captured2 = capsys.readouterr()
    assert "明文 API Key" not in captured2.err

