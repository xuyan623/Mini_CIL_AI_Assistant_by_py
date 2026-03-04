from __future__ import annotations

from ai_assistant.shell.command_validator import ShellCommandValidator


def test_rejects_natural_language_and_placeholders() -> None:
    validator = ShellCommandValidator()
    ok_text, msg_text = validator.validate("首先，用户描述是检查文件")
    assert ok_text is False
    assert "不是可执行命令" in msg_text

    ok_placeholder, msg_placeholder = validator.validate("ai code check <FILE_PATH> --start 1 --end 10")
    assert ok_placeholder is False
    assert "占位符" in msg_placeholder


def test_accepts_executable_shell_command() -> None:
    validator = ShellCommandValidator()
    ok, message = validator.validate("test -f /home/mycode/Sam.c")
    assert ok is True
    assert message == ""


def test_rejects_empty_and_overlong_natural_language() -> None:
    validator = ShellCommandValidator()
    ok_empty, message_empty = validator.validate("")
    assert ok_empty is False
    assert "命令为空" in message_empty

    long_text = "这是一段很长的自然语言描述" * 20
    ok_long, message_long = validator.validate(long_text)
    assert ok_long is False
    assert "不是可执行命令" in message_long


def test_contains_placeholder_token_patterns() -> None:
    validator = ShellCommandValidator()
    assert validator.contains_placeholder_token("<FILE_PATH>") is True
    assert validator.contains_placeholder_token("<MY_VAR>") is True
    assert validator.contains_placeholder_token("<文件路径>") is True
    assert validator.contains_placeholder_token("<desc here>") is True
    assert validator.contains_placeholder_token("echo '<not_a_placeholder>'") is True


def test_chinese_command_text_with_shell_token_not_rejected() -> None:
    validator = ShellCommandValidator()
    candidate = "请帮我执行 find . -type f -name Sam.c 并返回结果"
    assert validator.is_natural_language_line(candidate) is False
