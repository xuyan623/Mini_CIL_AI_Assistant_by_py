from __future__ import annotations

from ai_assistant.planner import capabilities
from ai_assistant.planner.task_interpreter import TaskInterpreter
from ai_assistant.planner.types import CapabilityParameter, CommandCapability


def test_capabilities_helpers_cover_remaining_branches(monkeypatch) -> None:
    assert capabilities.get_capability("not-exists") is None

    aliases = list(capabilities.iter_capability_aliases())
    assert aliases
    assert all(len(item) == 2 for item in aliases)

    registry = (
        CommandCapability(
            capability_id="chat.chat",
            module="chat",
            action="chat",
            summary="普通对话",
            command_template="ai chat <message>",
            aliases=("聊天",),
        ),
    )
    monkeypatch.setattr(capabilities, "CAPABILITY_REGISTRY", registry)
    reference = capabilities.build_capability_cli_reference()
    assert "chat.chat" in reference
    assert "规则：" in reference


def test_task_interpreter_json_and_parse_edge_branches() -> None:
    interpreter = TaskInterpreter()
    assert interpreter._load_json_object("") is None
    assert interpreter._load_json_object("x {\"capability_id\":\"code.check\"} y") == {"capability_id": "code.check"}
    assert interpreter._load_json_object("x {bad} y") is None

    assert interpreter.should_try_ai_language_parse(
        interpreter.interpret("只是闲聊", []),
    ) is False
    assert interpreter.should_try_ai_language_parse(
        interpreter.interpret("请帮我检查代码", []),
    ) is True

    unknown_capability = interpreter.parse_ai_task(
        raw_description="r",
        normalized_description="n",
        retry_note="",
        raw_response='{"capability_id":"x.unknown","parameters":{}}',
    )
    assert unknown_capability is None

    missing_file = interpreter.parse_ai_task(
        raw_description="r",
        normalized_description="请注释代码",
        retry_note="",
        raw_response='{"capability_id":"code.comment","parameters":{},"missing_parameters":"x"}',
    )
    assert missing_file is not None
    assert "file" in missing_file.missing_parameters
    assert "缺少目标文件名" in missing_file.note

    auto_fill_file = interpreter.parse_ai_task(
        raw_description="r",
        normalized_description="./src/main.c 请检查",
        retry_note="",
        raw_response='{"capability_id":"code.check","parameters":{"file":"main.c","start":1,"end":20},"missing_parameters":[]}',
    )
    assert auto_fill_file is not None
    assert auto_fill_file.parameters["file"].endswith("main.c")

    invalid_retry = interpreter.interpret(
        "重试",
        [{"event_type": "shell_plan", "input": "重试"}],
    )
    assert invalid_retry.capability_id == "__invalid__"
    assert "未找到可重试任务" in invalid_retry.note
