from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import ai_assistant.cli_runtime as cli_runtime
from ai_assistant.models import CommandResult


class _ContextStub:
    def __init__(self) -> None:
        self.file_service = SimpleNamespace(
            list_directory=lambda path=".": "ok",
            read_file=lambda path: "ok",
            search_file=lambda path, keyword: "ok",
            find_files=lambda keyword, search_dir=".": "ok",
            remove_file=lambda path, force=False: "ok",
            remove_directory=lambda path, force=False: "ok",
        )
        self.code_service = SimpleNamespace(
            check=lambda file, start, end: "ok",
            comment=lambda file, start, end, yes=False: "ok",
            explain=lambda file, start, end: "ok",
            optimize=lambda file, start, end, yes=False: "ok",
            generate=lambda file, start, end, desc, yes=False: "ok",
            summarize=lambda file: "ok",
        )
        self.context_service = SimpleNamespace(
            set_context=lambda file, start, end: "ok",
            add_context=lambda file, start, end: "ok",
            list_context=lambda: "ok",
            clear_context=lambda: "ok",
            build_prompt=lambda question: f"prompt:{question}",
        )
        self.backup_service = SimpleNamespace(
            create_backup=lambda file, keep=5: (True, "ok"),
            backup_status=lambda file=None: "ok",
            list_backups=lambda file: [],
            restore_backup=lambda backup_file, target_path=None, backup_target_before_restore=True: (True, "ok"),
            clean_backups=lambda file, keep: (True, "ok"),
        )
        self.config_service = SimpleNamespace(
            add_profile=lambda **kwargs: (True, "ok"),
            switch_profile=lambda profile: (True, f"switched:{profile}"),
            list_profiles=lambda: [{"id": "a", "name": "A", "current": True, "model": "m", "stream": False}],
            get_active_profile=lambda: SimpleNamespace(profile_id="a", name="A", model="m", api_url="u", stream=False),
            delete_profile=lambda profile: (True, "ok"),
            set_stream=lambda profile, enabled: (True, "ok"),
            export_profile=lambda profile, output, redact=False: (True, "ok"),
            import_profile=lambda input_file, profile=None: (True, "ok"),
        )
        self.shell_service = SimpleNamespace(run=lambda description: (True, "ok"))
        self.chat_service = SimpleNamespace(chat=lambda message, use_history=True: "chat-ok", clear_history=lambda: "cleared")
        self.ai_gateway = SimpleNamespace(chat=lambda *args, **kwargs: SimpleNamespace(ok=False, content="boom"))
        self.history_service = SimpleNamespace(
            trim_and_summarize=lambda callback: None,
            build_messages_for_request=lambda **kwargs: [{"role": "user", "content": kwargs["user_prompt"]}],
            append_exchange=lambda user, assistant: None,
            append_command_record=lambda **kwargs: None,
        )


def test_recording_argument_parser_paths() -> None:
    parser = cli_runtime.RecordingArgumentParser(prog="ai")
    parser._print_message("hello")
    assert parser._consume_output() == "hello"

    parser._print_message("bye")
    with pytest.raises(cli_runtime.ArgumentParsingExit) as ok_exit:
        parser.exit(0)
    assert ok_exit.value.exit_code == 0
    assert ok_exit.value.message == "bye"

    with pytest.raises(cli_runtime.ArgumentParsingExit) as err_exit:
        parser.error("bad")
    assert err_exit.value.exit_code == 2
    assert "error:" in err_exit.value.message


def test_unknown_handlers_and_context_error_branches(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _ContextStub()

    file_unknown = cli_runtime._handle_file(argparse.Namespace(action="x"), ctx)  # type: ignore[arg-type]
    assert file_unknown.ok is False
    code_unknown = cli_runtime._handle_code(argparse.Namespace(action="x"), ctx)  # type: ignore[arg-type]
    assert code_unknown.ok is False
    backup_unknown = cli_runtime._handle_backup(argparse.Namespace(action="x"), ctx)  # type: ignore[arg-type]
    assert backup_unknown.ok is False
    config_unknown = cli_runtime._handle_config(argparse.Namespace(action="x"), ctx)  # type: ignore[arg-type]
    assert config_unknown.ok is False
    context_unknown = cli_runtime._handle_context(argparse.Namespace(action="x"), ctx)  # type: ignore[arg-type]
    assert context_unknown.ok is False

    ctx.context_service.set_context = lambda file, start, end: (_ for _ in ()).throw(FileNotFoundError("no file"))
    missing_context = cli_runtime._handle_context(
        argparse.Namespace(action="set", file="x.py", start=1, end=2),
        ctx,  # type: ignore[arg-type]
    )
    assert missing_context.ok is False
    assert "no file" in missing_context.message

    ctx.context_service.add_context = lambda file, start, end: (_ for _ in ()).throw(RuntimeError("explode"))
    failed_context = cli_runtime._handle_context(
        argparse.Namespace(action="add", file="x.py", start=1, end=2),
        ctx,  # type: ignore[arg-type]
    )
    assert failed_context.ok is False
    assert "设置上下文失败" in failed_context.message

    ctx.context_service.build_prompt = lambda question: (_ for _ in ()).throw(ValueError("empty"))
    ask_error = cli_runtime._handle_context(argparse.Namespace(action="ask", question="?"), ctx)  # type: ignore[arg-type]
    assert ask_error.ok is False
    assert "empty" in ask_error.message

    ctx.config_service.list_profiles = lambda: [{"id": "a", "name": "A", "current": False, "model": "m", "stream": False}]
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    result_empty = cli_runtime._handle_config(argparse.Namespace(action="switch", profile=None), ctx)  # type: ignore[arg-type]
    assert result_empty.ok is True
    assert "已取消切换" in result_empty.message
    assert "请输入有效序号或配置 ID" in capsys.readouterr().out


def test_config_switch_interactive_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ctx = _ContextStub()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    ctx.config_service.list_profiles = lambda: []
    no_profile = cli_runtime._handle_config(argparse.Namespace(action="switch", profile=None), ctx)  # type: ignore[arg-type]
    assert no_profile.ok is False
    assert "无可用配置" in no_profile.message

    ctx.config_service.list_profiles = lambda: [{"id": "a", "name": "A", "current": True, "model": "m", "stream": False}]
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    eof_cancel = cli_runtime._handle_config(argparse.Namespace(action="switch", profile=None), ctx)  # type: ignore[arg-type]
    assert eof_cancel.ok is True
    assert "已取消切换" in eof_cancel.message

    answers = iter(["2", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    switched = cli_runtime._handle_config(argparse.Namespace(action="switch", profile=None), ctx)  # type: ignore[arg-type]
    assert switched.ok is True
    assert "switched:a" in switched.message
    assert "序号无效，请重新输入" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    quit_switch = cli_runtime._handle_config(argparse.Namespace(action="switch", profile=None), ctx)  # type: ignore[arg-type]
    assert quit_switch.ok is True


def test_dispatch_extra_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ContextStub()

    retry_result = cli_runtime._dispatch(["retry"], ctx)  # type: ignore[arg-type]
    assert retry_result.ok is True
    assert retry_result.data.get("module_hint") == "shell"

    migrate_result = cli_runtime._dispatch(["/old"], ctx)  # type: ignore[arg-type]
    assert migrate_result.ok is False
    assert "旧命令语法已下线" in migrate_result.message

    parse_error = cli_runtime._dispatch(["config", "stream", "x"], ctx)  # type: ignore[arg-type]
    assert parse_error.ok is False
    assert parse_error.exit_code == 2

    fake_args = argparse.Namespace(module="unknown", clear=False)
    monkeypatch.setattr(cli_runtime, "_build_parser", lambda: SimpleNamespace(parse_args=lambda _argv: fake_args))
    unknown_module = cli_runtime._dispatch(["-x"], ctx)  # type: ignore[arg-type]
    assert unknown_module.ok is False
    assert "不支持的模块" in unknown_module.message

    assert cli_runtime._format_command_line([]) == "ai"


def test_run_exception_paths_and_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    records: list[dict[str, object]] = []

    class _History:
        @staticmethod
        def append_command_record(**kwargs: object) -> None:
            records.append(kwargs)

    fake_ctx = SimpleNamespace(history_service=_History())
    monkeypatch.setattr(cli_runtime, "AppContext", lambda: fake_ctx)
    monkeypatch.setattr(
        cli_runtime,
        "_dispatch",
        lambda _argv, _ctx: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    code = cli_runtime.run(["chat", "x"])
    assert code == 1
    assert "内部错误" in capsys.readouterr().out
    assert "RuntimeError: boom" in str(records[0]["output_text"])

    class _BrokenHistory:
        @staticmethod
        def append_command_record(**kwargs: object) -> None:
            raise RuntimeError("history broken")

    fake_ctx2 = SimpleNamespace(history_service=_BrokenHistory())
    monkeypatch.setattr(cli_runtime, "AppContext", lambda: fake_ctx2)
    monkeypatch.setattr(cli_runtime, "_dispatch", lambda _argv, _ctx: CommandResult(True, "ok", 0, data={}))
    code2 = cli_runtime.run(["chat", "x"])
    assert code2 == 0
    assert "ok" in capsys.readouterr().out

    monkeypatch.setattr(cli_runtime, "run", lambda argv=None: 7)  # type: ignore[assignment]
    with pytest.raises(SystemExit) as system_exit:
        cli_runtime.main()
    assert system_exit.value.code == 7
