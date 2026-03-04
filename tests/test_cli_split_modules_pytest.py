from __future__ import annotations

import ai_assistant.cli_handlers as cli_handlers
import ai_assistant.cli_parser as cli_parser


def test_cli_parser_exports() -> None:
    parser = cli_parser._build_parser()
    assert parser.prog == "ai"
    assert callable(cli_parser._parse_on_off)
    assert cli_parser.ArgumentParsingExit is not None


def test_cli_handlers_exports() -> None:
    assert cli_handlers.AppContext is not None
    assert callable(cli_handlers._result_from_message)
    assert callable(cli_handlers._migration_message)
    assert callable(cli_handlers._handle_file)
    assert callable(cli_handlers._handle_code)
    assert callable(cli_handlers._handle_context)
    assert callable(cli_handlers._handle_backup)
    assert callable(cli_handlers._handle_config)
    assert callable(cli_handlers._handle_shell)

