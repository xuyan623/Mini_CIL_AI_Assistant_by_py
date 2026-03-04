from __future__ import annotations

import argparse
import os
import shlex
import traceback
import sys
from pathlib import Path

from ai_assistant.command_rules import build_cli_command_rules_prompt
from ai_assistant.models import CommandResult
from ai_assistant.paths import get_path_manager
from ai_assistant.state import JsonStateStore
from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.ai_gateway import AIGateway
from ai_assistant.services.backup_service import BackupService
from ai_assistant.services.chat_service import ChatService
from ai_assistant.services.code_service import CodeService
from ai_assistant.services.config_service import ConfigService
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.file_service import FileService
from ai_assistant.services.history_service import HistoryService
from ai_assistant.services.shell_service import ShellService
from ai_assistant.ui import RuntimeFeedback


class ArgumentParsingExit(Exception):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class RecordingArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._captured_output: list[str] = []

    def _print_message(self, message: str | None, file: object | None = None) -> None:
        if message:
            self._captured_output.append(message)

    def _consume_output(self) -> str:
        merged = "".join(self._captured_output).strip()
        self._captured_output.clear()
        return merged

    def exit(self, status: int = 0, message: str | None = None) -> None:
        text = self._consume_output()
        if message:
            text = f"{text}\n{message}".strip() if text else message.strip()
        if not text and status == 0:
            text = self.format_help().strip()
        raise ArgumentParsingExit(text, status)

    def error(self, message: str) -> None:
        usage = self.format_usage().strip()
        self.exit(2, f"{usage}\n{self.prog}: error: {message}")


def _parse_on_off(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"on", "true", "1", "yes", "y"}:
        return True
    if lowered in {"off", "false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("请使用 on/off")


class AppContext:
    def __init__(self) -> None:
        root_override = os.environ.get("AI_ASSISTANT_ROOT")
        if root_override:
            project_root = Path(root_override).expanduser().resolve()
        else:
            project_root = Path(__file__).resolve().parents[1]
        self.path_manager = get_path_manager(project_root)
        self.state_store = JsonStateStore()
        self.config_service = ConfigService(self.path_manager, state_store=self.state_store)
        self.history_service = HistoryService(self.path_manager, state_store=self.state_store)
        self.ai_client = AIClient(self.config_service)
        self.ai_gateway = AIGateway(self.ai_client)
        self.backup_service = BackupService(self.path_manager)
        self.file_service = FileService()
        self.context_service = ContextService(self.path_manager, state_store=self.state_store)
        self.code_service = CodeService(
            self.ai_client,
            self.backup_service,
            history_service=self.history_service,
            context_service=self.context_service,
        )
        self.chat_service = ChatService(
            self.ai_client,
            self.history_service,
            context_service=self.context_service,
        )
        self.shell_service = ShellService(
            self.ai_client,
            history_service=self.history_service,
            context_service=self.context_service,
        )


def _migration_message(old_command: str) -> str:
    return (
        f"❌ 旧命令语法已下线：{old_command}\n"
        "请使用新子命令语法，例如：\n"
        "  ai file ls .\n"
        "  ai code check app.py --start 1 --end 20\n"
        "  ai backup create app.py --keep 5\n"
        "  ai config stream deepseek on"
    )


def _build_parser() -> RecordingArgumentParser:
    parser = RecordingArgumentParser(prog="ai", description="AI Assistant v2 (Alpine-first)")
    parser.add_argument("--clear", action="store_true", help="清空历史")

    subparsers = parser.add_subparsers(dest="module")

    chat_parser = subparsers.add_parser("chat", help="普通对话")
    chat_parser.add_argument("message", nargs="*", help="消息文本")
    chat_parser.add_argument("--no-history", action="store_true", help="兼容参数（当前版本保留但不生效）")

    file_parser = subparsers.add_parser("file", help="文件与目录操作")
    file_sub = file_parser.add_subparsers(dest="action", required=True)
    ls_parser = file_sub.add_parser("ls", help="查看目录")
    ls_parser.add_argument("path", nargs="?", default=".")
    read_parser = file_sub.add_parser("read", help="读取文件")
    read_parser.add_argument("path")
    search_parser = file_sub.add_parser("search", help="搜索文件内容")
    search_parser.add_argument("path")
    search_parser.add_argument("keyword")
    find_parser = file_sub.add_parser("find", help="查找文件")
    find_parser.add_argument("keyword")
    find_parser.add_argument("--dir", default=".", dest="search_dir")
    rm_parser = file_sub.add_parser("rm", help="删除文件")
    rm_parser.add_argument("path")
    rm_parser.add_argument("--force", action="store_true")
    rmdir_parser = file_sub.add_parser("rmdir", help="删除目录")
    rmdir_parser.add_argument("path")
    rmdir_parser.add_argument("--force", action="store_true")

    code_parser = subparsers.add_parser("code", help="代码操作")
    code_sub = code_parser.add_subparsers(dest="action", required=True)
    for action in ["check", "comment", "explain", "optimize"]:
        action_parser = code_sub.add_parser(action, help=f"{action} 代码")
        action_parser.add_argument("file")
        action_parser.add_argument("--start", type=int, required=True)
        action_parser.add_argument("--end", type=int, required=True)
        if action in {"comment", "optimize"}:
            action_parser.add_argument("--yes", action="store_true", help="自动确认写入")

    generate_parser = code_sub.add_parser("generate", help="生成代码")
    generate_parser.add_argument("file")
    generate_parser.add_argument("--start", type=int, required=True)
    generate_parser.add_argument("--end", type=int, required=True)
    generate_parser.add_argument("--desc", required=True)
    generate_parser.add_argument("--yes", action="store_true", help="自动确认写入")

    summarize_parser = code_sub.add_parser("summarize", help="总结文件")
    summarize_parser.add_argument("file")

    context_parser = subparsers.add_parser("context", help="代码上下文")
    context_sub = context_parser.add_subparsers(dest="action", required=True)
    for action in ["set", "add"]:
        action_parser = context_sub.add_parser(action)
        action_parser.add_argument("file")
        action_parser.add_argument("--start", type=int)
        action_parser.add_argument("--end", type=int)

    context_sub.add_parser("list")
    ask_parser = context_sub.add_parser("ask")
    ask_parser.add_argument("question")
    context_sub.add_parser("clear")

    backup_parser = subparsers.add_parser("backup", help="备份管理")
    backup_sub = backup_parser.add_subparsers(dest="action", required=True)
    backup_create = backup_sub.add_parser("create")
    backup_create.add_argument("file")
    backup_create.add_argument("--keep", type=int, default=5)
    backup_status = backup_sub.add_parser("status")
    backup_status.add_argument("file", nargs="?")
    backup_list = backup_sub.add_parser("list")
    backup_list.add_argument("file")
    backup_restore = backup_sub.add_parser("restore")
    backup_restore.add_argument("backup_file")
    backup_restore.add_argument("--target")
    backup_restore.add_argument("--no-backup-target-before-restore", action="store_true")
    backup_clean = backup_sub.add_parser("clean")
    backup_clean.add_argument("file")
    backup_clean.add_argument("--keep", type=int, required=True)

    config_parser = subparsers.add_parser("config", help="配置管理")
    config_sub = config_parser.add_subparsers(dest="action", required=True)
    config_add = config_sub.add_parser("add")
    config_add.add_argument("--profile", required=True)
    config_add.add_argument("--name", required=True)
    config_add.add_argument("--api-key", required=True)
    config_add.add_argument("--api-url", required=True)
    config_add.add_argument("--model", required=True)
    config_add.add_argument("--stream", type=_parse_on_off, default=False)
    config_switch = config_sub.add_parser("switch")
    config_switch.add_argument("profile", nargs="?")
    config_sub.add_parser("list")
    config_sub.add_parser("current")
    config_delete = config_sub.add_parser("delete")
    config_delete.add_argument("profile")
    config_stream = config_sub.add_parser("stream")
    config_stream.add_argument("profile")
    config_stream.add_argument("enabled", type=_parse_on_off)
    config_export = config_sub.add_parser("export")
    config_export.add_argument("profile")
    config_export.add_argument("output")
    config_export.add_argument("--redact", action="store_true")
    config_import = config_sub.add_parser("import")
    config_import.add_argument("input")
    config_import.add_argument("--profile")

    shell_parser = subparsers.add_parser("shell", help="shell 命令生成")
    shell_sub = shell_parser.add_subparsers(dest="action", required=True)
    shell_run = shell_sub.add_parser("run")
    shell_run.add_argument("description")

    return parser


def _result_from_message(
    message: str,
    error_code: int = 1,
    data: dict[str, object] | None = None,
) -> CommandResult:
    ok = not message.startswith("❌")
    return CommandResult(ok, message, 0 if ok else error_code, data=data or {})


def _handle_file(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    if args.action == "ls":
        return _result_from_message(ctx.file_service.list_directory(args.path))
    if args.action == "read":
        return _result_from_message(ctx.file_service.read_file(args.path))
    if args.action == "search":
        return _result_from_message(ctx.file_service.search_file(args.path, args.keyword))
    if args.action == "find":
        return _result_from_message(ctx.file_service.find_files(args.keyword, args.search_dir))
    if args.action == "rm":
        return _result_from_message(ctx.file_service.remove_file(args.path, args.force))
    if args.action == "rmdir":
        return _result_from_message(ctx.file_service.remove_directory(args.path, args.force))
    return CommandResult(False, "❌ 未知 file 子命令", 2)


def _handle_code(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    if args.action == "check":
        return _result_from_message(ctx.code_service.check(args.file, args.start, args.end))
    if args.action == "comment":
        return _result_from_message(ctx.code_service.comment(args.file, args.start, args.end, yes=getattr(args, "yes", False)))
    if args.action == "explain":
        return _result_from_message(ctx.code_service.explain(args.file, args.start, args.end))
    if args.action == "optimize":
        return _result_from_message(ctx.code_service.optimize(args.file, args.start, args.end, yes=getattr(args, "yes", False)))
    if args.action == "generate":
        return _result_from_message(
            ctx.code_service.generate(args.file, args.start, args.end, args.desc, yes=getattr(args, "yes", False))
        )
    if args.action == "summarize":
        return _result_from_message(ctx.code_service.summarize(args.file))
    return CommandResult(False, "❌ 未知 code 子命令", 2)


def _handle_context(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    if args.action in {"set", "add"}:
        try:
            if args.action == "set":
                return _result_from_message(ctx.context_service.set_context(args.file, args.start, args.end))
            return _result_from_message(ctx.context_service.add_context(args.file, args.start, args.end))
        except (FileNotFoundError, ValueError) as exc:
            return CommandResult(False, f"❌ {exc}", 1)
        except Exception as exc:
            return CommandResult(False, f"❌ 设置上下文失败：{exc}", 1)
    if args.action == "list":
        return _result_from_message(ctx.context_service.list_context())
    if args.action == "clear":
        return _result_from_message(ctx.context_service.clear_context())
    if args.action == "ask":
        try:
            prompt = ctx.context_service.build_prompt(args.question)
        except ValueError as exc:
            return CommandResult(False, f"❌ {exc}", 1)

        ctx.history_service.trim_and_summarize(ctx.ai_gateway.summarize_messages)
        messages = ctx.history_service.build_messages_for_request(
            user_prompt=prompt,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=[build_cli_command_rules_prompt()],
        )
        response = ctx.ai_gateway.chat(
            messages,
            stream_override=None,
            print_stream=True,
            timeout=90,
            attempt_callback=RuntimeFeedback(enabled=True).as_attempt_callback(),
        )
        stream_enabled = ctx.config_service.get_active_profile().stream
        if response.ok:
            ctx.history_service.append_exchange(f"context ask: {args.question}", response.content)
            return CommandResult(
                True,
                response.content,
                data={"streamed_output": bool(stream_enabled)},
            )

        ctx.history_service.append_exchange(f"context ask: {args.question}", response.content)
        return CommandResult(False, response.content, 1, data={"streamed_output": False})

    return CommandResult(False, "❌ 未知 context 子命令", 2)


def _handle_backup(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    if args.action == "create":
        ok, message = ctx.backup_service.create_backup(args.file, args.keep)
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "status":
        return CommandResult(True, ctx.backup_service.backup_status(args.file))
    if args.action == "list":
        backups = ctx.backup_service.list_backups(args.file)
        if not backups:
            return CommandResult(True, f"📝 文件没有备份：{Path(args.file).expanduser().resolve()}")
        lines = [f"📝 备份列表：共 {len(backups)} 个"]
        for index, item in enumerate(backups, 1):
            lines.append(f"  {index}. {item.get('backup_file')} ({item.get('ts', '')})")
        return CommandResult(True, "\n".join(lines))
    if args.action == "restore":
        ok, message = ctx.backup_service.restore_backup(
            args.backup_file,
            target_path=args.target,
            backup_target_before_restore=not args.no_backup_target_before_restore,
        )
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "clean":
        ok, message = ctx.backup_service.clean_backups(args.file, args.keep)
        return CommandResult(ok, message, 0 if ok else 1)
    return CommandResult(False, "❌ 未知 backup 子命令", 2)


def _handle_config(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    service = ctx.config_service
    if args.action == "add":
        ok, message = service.add_profile(
            profile_id=args.profile,
            name=args.name,
            api_key=args.api_key,
            api_url=args.api_url,
            model=args.model,
            stream=args.stream,
            overwrite=False,
        )
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "switch":
        if args.profile:
            ok, message = service.switch_profile(args.profile)
            return CommandResult(ok, message, 0 if ok else 1)

        profiles = service.list_profiles()
        if not profiles:
            return CommandResult(False, "❌ 无可用配置", 1)

        stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if not (stdin_is_tty and stdout_is_tty):
            lines = ["❌ 请指定 profile。可用配置："]
            for profile in profiles:
                lines.append(f"  - {profile['id']}")
            lines.append("示例：ai config switch <profile>")
            return CommandResult(False, "\n".join(lines), 2)

        print("📋 可用配置：")
        for index, profile in enumerate(profiles, 1):
            current_mark = "⭐" if profile["current"] else "  "
            print(f"  {index}. {current_mark} {profile['id']} ({profile['name']})")

        while True:
            try:
                choice = input("请输入序号或配置 ID（q 取消）: ").strip()
            except EOFError:
                return CommandResult(True, "✅ 已取消切换", 0)
            if not choice:
                print("请输入有效序号或配置 ID")
                continue
            lowered = choice.lower()
            if lowered in {"q", "quit", "exit"}:
                return CommandResult(True, "✅ 已取消切换", 0)

            target = choice
            if choice.isdigit():
                index = int(choice) - 1
                if index < 0 or index >= len(profiles):
                    print("序号无效，请重新输入")
                    continue
                target = profiles[index]["id"]

            ok, message = service.switch_profile(target)
            return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "list":
        profiles = service.list_profiles()
        lines = ["📋 配置列表："]
        for profile in profiles:
            current_mark = "⭐" if profile["current"] else "  "
            stream_mark = "on" if profile["stream"] else "off"
            lines.append(f"  {current_mark} {profile['id']} ({profile['name']}) model={profile['model']} stream={stream_mark}")
        return CommandResult(True, "\n".join(lines))
    if args.action == "current":
        profile = service.get_active_profile()
        stream_mark = "on" if profile.stream else "off"
        message = (
            f"⭐ 当前配置：{profile.profile_id}\n"
            f"名称：{profile.name}\n模型：{profile.model}\nURL：{profile.api_url}\n流式：{stream_mark}"
        )
        return CommandResult(True, message)
    if args.action == "delete":
        ok, message = service.delete_profile(args.profile)
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "stream":
        ok, message = service.set_stream(args.profile, args.enabled)
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "export":
        ok, message = service.export_profile(args.profile, Path(args.output), args.redact)
        return CommandResult(ok, message, 0 if ok else 1)
    if args.action == "import":
        ok, message = service.import_profile(Path(args.input), args.profile)
        return CommandResult(ok, message, 0 if ok else 1)
    return CommandResult(False, "❌ 未知 config 子命令", 2)


def _handle_shell(args: argparse.Namespace, ctx: AppContext) -> CommandResult:
    ok, message = ctx.shell_service.run(args.description)
    return CommandResult(ok, message, 0 if ok else 1)


def _format_command_line(argv: list[str]) -> str:
    if not argv:
        return "ai"
    return f"ai {' '.join(shlex.quote(item) for item in argv)}"


def _dispatch(argv: list[str], ctx: AppContext) -> CommandResult:
    retry_aliases = {"重试", "再试", "再试一次", "retry", "try again", "继续"}
    if len(argv) == 1 and argv[0].strip().lower() in retry_aliases:
        ok, message = ctx.shell_service.run(argv[0].strip())
        return CommandResult(ok, message, 0 if ok else 1, data={"module_hint": "shell"})

    if "--execute" in argv:
        return CommandResult(
            False,
            "❌ --execute 已下线，shell run 改为交互式分步执行：ai shell run \"查找大文件\"",
            2,
        )

    if argv and argv[0].startswith("/"):
        return CommandResult(False, _migration_message(argv[0]), 2)

    known_modules = {"chat", "file", "code", "context", "backup", "config", "shell"}
    if argv and argv[0] not in known_modules and not argv[0].startswith("-"):
        message = " ".join(argv).strip()
        output = ctx.chat_service.chat(message, use_history=True)
        stream_enabled = ctx.config_service.get_active_profile().stream
        suppress_print = bool(stream_enabled) and not output.startswith("❌")
        return _result_from_message(
            output,
            error_code=1,
            data={"streamed_output": suppress_print, "module_hint": "chat"},
        )

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except ArgumentParsingExit as exc:
        output = exc.message.strip() if exc.message else parser.format_help().strip()
        return CommandResult(exc.exit_code == 0, output, exc.exit_code)

    if args.clear:
        return _result_from_message(ctx.chat_service.clear_history())

    if args.module is None:
        return CommandResult(True, parser.format_help().strip(), 0)

    if args.module == "chat":
        message = " ".join(args.message).strip()
        if not message:
            return CommandResult(False, "❌ chat 需要消息文本，例如：ai chat 你好", 2)
        output = ctx.chat_service.chat(message, use_history=not args.no_history)
        stream_enabled = ctx.config_service.get_active_profile().stream
        suppress_print = bool(stream_enabled) and not output.startswith("❌")
        return _result_from_message(
            output,
            error_code=1,
            data={"streamed_output": suppress_print},
        )

    handlers = {
        "file": _handle_file,
        "code": _handle_code,
        "context": _handle_context,
        "backup": _handle_backup,
        "config": _handle_config,
        "shell": _handle_shell,
    }

    handler = handlers.get(args.module)
    if handler is None:
        return CommandResult(False, f"❌ 不支持的模块：{args.module}", 2)
    return handler(args, ctx)


def run(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    ctx = AppContext()
    command_line = _format_command_line(argv)

    try:
        result = _dispatch(argv, ctx)
    except KeyboardInterrupt:
        result = CommandResult(
            False,
            "❌ 操作已取消",
            130,
            data={"interrupted": True, "module_hint": (argv[0] if argv else "none")},
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        result = CommandResult(False, f"❌ 内部错误：{exc}", 1, data={"traceback": traceback_text})

    suppress_print = bool(result.data and result.data.get("streamed_output"))
    if result.message and not suppress_print:
        print(result.message)

    history_output = result.message
    traceback_text = result.data.get("traceback") if result.data else None
    if traceback_text:
        history_output = f"{history_output}\n{traceback_text}".strip()
    try:
        ctx.history_service.append_command_record(
            command_line=command_line,
            output_text=history_output,
            ok=result.ok,
            exit_code=result.exit_code,
            metadata={
                "module": (result.data or {}).get("module_hint") or (argv[0] if argv else "none"),
                "interrupted": bool((result.data or {}).get("interrupted", False)),
            },
        )
    except Exception:
        pass

    return result.exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
