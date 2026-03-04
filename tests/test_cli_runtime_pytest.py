from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pytest

import ai_assistant.cli_runtime as cli_runtime


class _FakeContext:
    def __init__(self) -> None:
        self.file_service = SimpleNamespace(
            list_directory=lambda path=".": f"ls:{path}",
            read_file=lambda path: f"read:{path}",
            search_file=lambda path, keyword: f"search:{path}:{keyword}",
            find_files=lambda keyword, search_dir=".": f"find:{keyword}:{search_dir}",
            remove_file=lambda path, force=False: f"rm:{path}:{force}",
            remove_directory=lambda path, force=False: f"rmdir:{path}:{force}",
        )
        self.code_service = SimpleNamespace(
            check=lambda file, start, end: f"check:{file}:{start}:{end}",
            comment=lambda file, start, end, yes=False: f"comment:{file}:{yes}",
            explain=lambda file, start, end: f"explain:{file}",
            optimize=lambda file, start, end, yes=False: f"optimize:{file}:{yes}",
            generate=lambda file, start, end, desc, yes=False: f"generate:{file}:{desc}:{yes}",
            summarize=lambda file: f"summarize:{file}",
        )
        self.context_service = SimpleNamespace(
            set_context=lambda file, start, end: f"set:{file}:{start}:{end}",
            add_context=lambda file, start, end: f"add:{file}:{start}:{end}",
            list_context=lambda: "context:list",
            clear_context=lambda: "context:clear",
            build_prompt=lambda question: f"prompt:{question}",
        )
        self.backup_service = SimpleNamespace(
            create_backup=lambda file, keep=5: (True, f"backup:create:{file}:{keep}"),
            backup_status=lambda file=None: f"backup:status:{file or ''}",
            list_backups=lambda file: [{"backup_file": "a.bak", "ts": "now"}],
            restore_backup=lambda backup_file, target_path=None, backup_target_before_restore=True: (
                True,
                f"backup:restore:{backup_file}:{target_path}:{backup_target_before_restore}",
            ),
            clean_backups=lambda file, keep: (True, f"backup:clean:{file}:{keep}"),
        )
        self.config_service = SimpleNamespace(
            add_profile=lambda **kwargs: (True, f"config:add:{kwargs['profile_id']}"),
            switch_profile=lambda profile: (True, f"config:switch:{profile}"),
            list_profiles=lambda: [
                {"id": "deepseek", "name": "DeepSeek", "model": "m", "api_url": "u", "stream": False, "current": True}
            ],
            get_active_profile=lambda: SimpleNamespace(profile_id="deepseek", name="DeepSeek", model="m", api_url="u", stream=False),
            delete_profile=lambda profile: (True, f"config:delete:{profile}"),
            set_stream=lambda profile, enabled: (True, f"config:stream:{profile}:{enabled}"),
            export_profile=lambda profile, output, redact=False: (True, f"config:export:{profile}:{output}:{redact}"),
            import_profile=lambda input_file, profile=None: (True, f"config:import:{input_file}:{profile}"),
            list_profile_ids=lambda: ["deepseek"],
        )
        self.shell_service = SimpleNamespace(run=lambda description: (True, f"shell:{description}"))
        self.chat_service = SimpleNamespace(chat=lambda message, use_history=True: f"chat:{message}:{use_history}", clear_history=lambda: "cleared")
        self.ai_gateway = SimpleNamespace(
            summarize_messages=lambda _messages: "",
            chat=lambda *args, **kwargs: SimpleNamespace(ok=True, content="context answer"),  # noqa: ARG005
        )
        self.history_service = SimpleNamespace(
            trim_and_summarize=lambda callback: None,
            build_messages_for_request=lambda **kwargs: [{"role": "user", "content": kwargs["user_prompt"]}],
            append_exchange=lambda user, assistant: None,
            append_command_record=lambda **kwargs: None,
        )


def test_parse_on_off_valid_and_invalid() -> None:
    assert cli_runtime._parse_on_off("on") is True
    assert cli_runtime._parse_on_off("off") is False
    with pytest.raises(argparse.ArgumentTypeError):
        cli_runtime._parse_on_off("x")


def test_dispatch_chat_and_execute_guard() -> None:
    ctx = _FakeContext()
    chat_result = cli_runtime._dispatch(["hello", "world"], ctx)  # type: ignore[arg-type]
    assert chat_result.ok is True
    assert "chat:hello world" in chat_result.message

    execute_result = cli_runtime._dispatch(["shell", "run", "x", "--execute"], ctx)  # type: ignore[arg-type]
    assert execute_result.ok is False
    assert "已下线" in execute_result.message


def test_dispatch_file_and_code_commands() -> None:
    ctx = _FakeContext()
    assert cli_runtime._dispatch(["file", "ls", "."], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["file", "read", "a.txt"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["file", "search", "a.txt", "k"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["file", "find", "Sam.c"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["file", "rm", "a.txt"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["file", "rmdir", "d"], ctx).ok  # type: ignore[arg-type]

    assert cli_runtime._dispatch(["code", "check", "a.py", "--start", "1", "--end", "1"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["code", "comment", "a.py", "--start", "1", "--end", "1", "--yes"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["code", "explain", "a.py", "--start", "1", "--end", "1"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["code", "optimize", "a.py", "--start", "1", "--end", "1"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["code", "generate", "a.py", "--start", "1", "--end", "1", "--desc", "x"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["code", "summarize", "a.py"], ctx).ok  # type: ignore[arg-type]


def test_dispatch_context_backup_shell_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _FakeContext()
    assert cli_runtime._dispatch(["context", "set", "a.py", "--start", "1", "--end", "2"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["context", "add", "a.py", "--start", "1", "--end", "2"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["context", "list"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["context", "clear"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["context", "ask", "what"], ctx).ok  # type: ignore[arg-type]

    assert cli_runtime._dispatch(["backup", "create", "a.py"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["backup", "status", "a.py"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["backup", "list", "a.py"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["backup", "restore", "a.bak"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["backup", "clean", "a.py", "--keep", "1"], ctx).ok  # type: ignore[arg-type]

    assert cli_runtime._dispatch(["shell", "run", "do"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["--clear"], ctx).ok  # type: ignore[arg-type]


def test_dispatch_config_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _FakeContext()
    assert cli_runtime._dispatch(
        ["config", "add", "--profile", "p", "--name", "n", "--api-key", "k", "--api-url", "u", "--model", "m"],
        ctx,  # type: ignore[arg-type]
    ).ok
    assert cli_runtime._dispatch(["config", "switch", "deepseek"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "list"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "current"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "delete", "deepseek"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "stream", "deepseek", "on"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "export", "deepseek", "out.json"], ctx).ok  # type: ignore[arg-type]
    assert cli_runtime._dispatch(["config", "import", "in.json"], ctx).ok  # type: ignore[arg-type]

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    result = cli_runtime._dispatch(["config", "switch"], ctx)  # type: ignore[arg-type]
    assert result.ok is False
    assert "请指定 profile" in result.message


def test_run_catches_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AI_ASSISTANT_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(cli_runtime, "_dispatch", lambda argv, ctx: (_ for _ in ()).throw(KeyboardInterrupt()))  # type: ignore[arg-type]
    code = cli_runtime.run(["chat", "hello"])
    assert code == 130


def test_migration_message_and_format_command_line() -> None:
    msg = cli_runtime._migration_message("/ls")
    assert "旧命令语法已下线" in msg
    formatted = cli_runtime._format_command_line(["chat", "hello world"])
    assert formatted.startswith("ai ")
    assert "'hello world'" in formatted


def test_dispatch_no_module_returns_help() -> None:
    ctx = _FakeContext()
    result = cli_runtime._dispatch([], ctx)  # type: ignore[arg-type]
    assert result.ok is True
    assert "usage: ai" in result.message
