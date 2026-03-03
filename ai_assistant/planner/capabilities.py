from __future__ import annotations

from collections.abc import Iterable

from ai_assistant.planner.types import CapabilityParameter, CommandCapability


CAPABILITY_REGISTRY: tuple[CommandCapability, ...] = (
    CommandCapability(
        capability_id="chat.chat",
        module="chat",
        action="chat",
        summary="普通对话",
        command_template="ai chat <message>",
    ),
    CommandCapability(
        capability_id="file.ls",
        module="file",
        action="ls",
        summary="查看目录",
        command_template="ai file ls <path>",
    ),
    CommandCapability(
        capability_id="file.read",
        module="file",
        action="read",
        summary="读取文件",
        command_template="ai file read <path>",
    ),
    CommandCapability(
        capability_id="file.search",
        module="file",
        action="search",
        summary="搜索文件内容",
        command_template="ai file search <path> <keyword>",
    ),
    CommandCapability(
        capability_id="file.find",
        module="file",
        action="find",
        summary="按关键字查找文件",
        command_template="ai file find <keyword> --dir .",
    ),
    CommandCapability(
        capability_id="file.rm",
        module="file",
        action="rm",
        summary="删除文件",
        command_template="ai file rm <path> --force",
    ),
    CommandCapability(
        capability_id="file.rmdir",
        module="file",
        action="rmdir",
        summary="删除目录",
        command_template="ai file rmdir <path> --force",
    ),
    CommandCapability(
        capability_id="context.ask",
        module="context",
        action="ask",
        summary="基于上下文问答",
        command_template="ai context ask <question>",
    ),
    CommandCapability(
        capability_id="context.set",
        module="context",
        action="set",
        summary="设置代码上下文",
        command_template="ai context set <file> --start <N> --end <N>",
    ),
    CommandCapability(
        capability_id="context.add",
        module="context",
        action="add",
        summary="追加代码上下文",
        command_template="ai context add <file> --start <N> --end <N>",
    ),
    CommandCapability(
        capability_id="context.list",
        module="context",
        action="list",
        summary="查看上下文",
        command_template="ai context list",
    ),
    CommandCapability(
        capability_id="context.clear",
        module="context",
        action="clear",
        summary="清空上下文",
        command_template="ai context clear",
    ),
    CommandCapability(
        capability_id="backup.create",
        module="backup",
        action="create",
        summary="创建备份",
        command_template="ai backup create <file> --keep 5",
    ),
    CommandCapability(
        capability_id="backup.status",
        module="backup",
        action="status",
        summary="查看备份状态",
        command_template="ai backup status [file]",
    ),
    CommandCapability(
        capability_id="backup.list",
        module="backup",
        action="list",
        summary="查看备份列表",
        command_template="ai backup list <file>",
    ),
    CommandCapability(
        capability_id="backup.restore",
        module="backup",
        action="restore",
        summary="恢复备份",
        command_template="ai backup restore <backup_file> --target <path>",
    ),
    CommandCapability(
        capability_id="backup.clean",
        module="backup",
        action="clean",
        summary="清理历史备份",
        command_template="ai backup clean <file> --keep <N>",
    ),
    CommandCapability(
        capability_id="config.current",
        module="config",
        action="current",
        summary="查看当前配置",
        command_template="ai config current",
    ),
    CommandCapability(
        capability_id="config.add",
        module="config",
        action="add",
        summary="添加配置",
        command_template="ai config add --profile <id> --name <name> --api-key <key> --api-url <url> --model <model>",
    ),
    CommandCapability(
        capability_id="config.switch",
        module="config",
        action="switch",
        summary="切换配置",
        command_template="ai config switch <profile>",
    ),
    CommandCapability(
        capability_id="config.list",
        module="config",
        action="list",
        summary="查看配置列表",
        command_template="ai config list",
    ),
    CommandCapability(
        capability_id="config.delete",
        module="config",
        action="delete",
        summary="删除配置",
        command_template="ai config delete <profile>",
    ),
    CommandCapability(
        capability_id="config.stream",
        module="config",
        action="stream",
        summary="设置流式开关",
        command_template="ai config stream <profile> on|off",
    ),
    CommandCapability(
        capability_id="config.export",
        module="config",
        action="export",
        summary="导出配置",
        command_template="ai config export <profile> <output> --redact",
    ),
    CommandCapability(
        capability_id="config.import",
        module="config",
        action="import",
        summary="导入配置",
        command_template="ai config import <input> --profile <id>",
    ),
    CommandCapability(
        capability_id="shell.run",
        module="shell",
        action="run",
        summary="自然语言分步执行",
        command_template="ai shell run <description>",
    ),
    CommandCapability(
        capability_id="code.check",
        module="code",
        action="check",
        summary="代码检查",
        command_template="ai code check {file} --start {start} --end {end}",
        aliases=("检查", "check", "bug", "问题", "排查"),
        required_parameters=(
            CapabilityParameter("file", True, "目标文件路径", "main.c"),
            CapabilityParameter("start", True, "起始行号", "1"),
            CapabilityParameter("end", True, "结束行号", "120"),
        ),
    ),
    CommandCapability(
        capability_id="code.comment",
        module="code",
        action="comment",
        summary="代码注释",
        command_template="ai code comment {file} --start {start} --end {end} --yes",
        aliases=("注释", "comment"),
        required_parameters=(
            CapabilityParameter("file", True, "目标文件路径", "main.c"),
            CapabilityParameter("start", True, "起始行号", "1"),
            CapabilityParameter("end", True, "结束行号", "120"),
        ),
        interactive=True,
    ),
    CommandCapability(
        capability_id="code.explain",
        module="code",
        action="explain",
        summary="代码解释",
        command_template="ai code explain {file} --start {start} --end {end}",
        aliases=("解释", "explain"),
        required_parameters=(
            CapabilityParameter("file", True, "目标文件路径", "main.c"),
            CapabilityParameter("start", True, "起始行号", "1"),
            CapabilityParameter("end", True, "结束行号", "120"),
        ),
    ),
    CommandCapability(
        capability_id="code.optimize",
        module="code",
        action="optimize",
        summary="代码优化",
        command_template="ai code optimize {file} --start {start} --end {end} --yes",
        aliases=("优化", "optimize", "重构", "修复"),
        required_parameters=(
            CapabilityParameter("file", True, "目标文件路径", "main.c"),
            CapabilityParameter("start", True, "起始行号", "1"),
            CapabilityParameter("end", True, "结束行号", "120"),
        ),
        interactive=True,
    ),
    CommandCapability(
        capability_id="code.generate",
        module="code",
        action="generate",
        summary="生成代码",
        command_template="ai code generate <file> --start <N> --end <N> --desc <描述>",
    ),
    CommandCapability(
        capability_id="code.summarize",
        module="code",
        action="summarize",
        summary="总结代码文件",
        command_template="ai code summarize <file>",
    ),
    CommandCapability(
        capability_id="workflow.code_fix",
        module="shell",
        action="run",
        summary="检查并修复代码",
        command_template="workflow",
        aliases=("修复bug", "检查并修改", "修改bug", "fix bug", "bug fix"),
        required_parameters=(CapabilityParameter("file", True, "目标文件路径", "main.c"),),
    ),
    CommandCapability(
        capability_id="workflow.ensure_directory",
        module="shell",
        action="run",
        summary="目录存在性检查并按需创建",
        command_template="if [ -d {target_dir} ]; then echo {target_dir}; else mkdir -p {target_dir} && echo {target_dir}; fi",
        aliases=("目录", "文件夹", "存在", "创建", "directory", "folder", "create if missing"),
        required_parameters=(
            CapabilityParameter("base_dir", True, "父目录路径", "."),
            CapabilityParameter("dir_name", True, "目录名称", "AI"),
        ),
    ),
    CommandCapability(
        capability_id="workflow.find_file",
        module="shell",
        action="run",
        summary="按名称查找文件",
        command_template="find {search_dir} -type f -name {file_name}",
        aliases=("查找文件", "find file", "找文件"),
        required_parameters=(
            CapabilityParameter("search_dir", True, "搜索根目录", "."),
            CapabilityParameter("file_name", True, "文件名", "main.c"),
        ),
    ),
)


def list_capabilities() -> tuple[CommandCapability, ...]:
    return CAPABILITY_REGISTRY


def get_capability(capability_id: str) -> CommandCapability | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.capability_id == capability_id:
            return capability
    return None


def iter_capability_aliases() -> Iterable[tuple[str, str]]:
    for capability in CAPABILITY_REGISTRY:
        for alias in capability.aliases:
            yield alias, capability.capability_id


def build_capability_cli_reference() -> str:
    lines = ["=== 可用 CLI 命令规范（由能力注册表生成） ===", "命令入口：ai"]
    grouped: dict[str, list[CommandCapability]] = {}
    for capability in CAPABILITY_REGISTRY:
        grouped.setdefault(capability.module, []).append(capability)

    ordered_modules = ["chat", "file", "code", "context", "backup", "config", "shell"]
    for module_name in ordered_modules:
        module_capabilities = grouped.get(module_name, [])
        if not module_capabilities:
            continue
        module_capabilities.sort(key=lambda item: item.action)
        for item in module_capabilities:
            if item.capability_id.startswith("workflow."):
                continue
            lines.append(f"- {item.module}.{item.action}：{item.summary}")

    lines.extend(
        [
            "规则：",
            "- 不可使用不存在的子命令。",
            "- code 的 check/comment/explain/optimize 必须同时带 --start 和 --end。",
            "- code generate 必须带 --start、--end、--desc。",
            "- code comment/optimize/generate 支持 --yes 跳过写入确认（适合自动执行流程）。",
            "- --execute 已下线，禁止生成或建议该参数。",
            "- shell run 为交互式分步执行：先给步骤，再逐步确认执行。",
            "- 回答时不要先输出“ok/收到”等无意义前缀，直接给出最终内容。",
            "- 优先输出可直接执行的命令；信息不足时先输出定位信息的命令。",
        ]
    )
    return "\n".join(lines)
