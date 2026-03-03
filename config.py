#!/usr/bin/env python3
"""Interactive config helper script for ai profile management."""

from __future__ import annotations

import argparse
import sys

from ai_assistant.services.config_service import ConfigService


def ask(question: str, default: str | None = None) -> str:
    while True:
        prompt = f"{question}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        try:
            answer = input(prompt).strip()
        except EOFError:
            print("\n👋 输入中断")
            raise SystemExit(1)

        if not answer and default is not None:
            return default
        if answer:
            return answer
        print("⚠️ 该项不能为空")


def add_interactive(service: ConfigService) -> int:
    print("\n== 添加配置 ==")
    profile = ask("配置 ID（如 deepseek/openai）")
    name = ask("显示名称", default=profile)
    api_key = ask("API Key")
    api_url = ask("API URL")
    model = ask("模型名称")
    stream_raw = ask("启用流式？(y/n)", default="n").lower()
    stream = stream_raw in {"y", "yes"}

    ok, message = service.add_profile(profile, name, api_key, api_url, model, stream=stream, overwrite=False)
    print(message)
    if ok:
        switch = ask("立即切换到该配置？(y/n)", default="y").lower()
        if switch in {"y", "yes"}:
            ok2, message2 = service.switch_profile(profile)
            print(message2)
            return 0 if ok2 else 1
    return 0 if ok else 1


def switch_interactive(service: ConfigService) -> int:
    profiles = service.list_profiles()
    if not profiles:
        print("❌ 无可用配置")
        return 1

    print("\n== 可用配置 ==")
    for index, profile in enumerate(profiles, 1):
        current = "⭐" if profile["current"] else "  "
        print(f"{index}. {current} {profile['id']} ({profile['model']})")

    choice = ask("输入序号或配置 ID")
    target = choice
    if choice.isdigit():
        index = int(choice) - 1
        if index < 0 or index >= len(profiles):
            print("❌ 序号无效")
            return 1
        target = profiles[index]["id"]

    ok, message = service.switch_profile(target)
    print(message)
    return 0 if ok else 1


def show_current(service: ConfigService) -> int:
    profile = service.get_active_profile()
    print(f"\n⭐ 当前配置：{profile.profile_id}")
    print(f"名称：{profile.name}")
    print(f"模型：{profile.model}")
    print(f"URL：{profile.api_url}")
    print(f"流式：{'on' if profile.stream else 'off'}")
    return 0


def list_profiles(service: ConfigService) -> int:
    profiles = service.list_profiles()
    if not profiles:
        print("📝 暂无配置")
        return 0

    print("\n📋 配置列表")
    for profile in profiles:
        current = "⭐" if profile["current"] else "  "
        stream = "on" if profile["stream"] else "off"
        print(f"{current} {profile['id']} ({profile['name']}) model={profile['model']} stream={stream}")
    return 0


def menu(service: ConfigService) -> int:
    while True:
        print("\n=== 配置管理 ===")
        print("1. 添加配置")
        print("2. 切换配置")
        print("3. 列出配置")
        print("4. 查看当前配置")
        print("5. 删除配置")
        print("6. 设置流式")
        print("0. 退出")

        choice = ask("请选择", default="0")
        if choice == "1":
            add_interactive(service)
        elif choice == "2":
            switch_interactive(service)
        elif choice == "3":
            list_profiles(service)
        elif choice == "4":
            show_current(service)
        elif choice == "5":
            profile = ask("输入要删除的配置 ID")
            ok, message = service.delete_profile(profile)
            print(message)
        elif choice == "6":
            profile = ask("配置 ID")
            value = ask("流式 on/off", default="off").lower()
            enabled = value in {"on", "true", "1", "y", "yes"}
            ok, message = service.set_stream(profile, enabled)
            print(message)
        elif choice == "0":
            print("👋 已退出")
            return 0
        else:
            print("❌ 无效选择")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config helper")
    parser.add_argument("--add", action="store_true", help="交互式添加配置")
    parser.add_argument("--switch", nargs="?", const="", metavar="PROFILE", help="切换配置")
    parser.add_argument("--list", action="store_true", help="列出配置")
    parser.add_argument("--current", action="store_true", help="查看当前配置")
    parser.add_argument("--delete", metavar="PROFILE", help="删除配置")
    parser.add_argument("--stream", nargs=2, metavar=("PROFILE", "ON_OFF"), help="设置流式开关")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = ConfigService()

    if args.add:
        return add_interactive(service)

    if args.switch is not None:
        if args.switch == "":
            return switch_interactive(service)
        ok, message = service.switch_profile(args.switch)
        print(message)
        return 0 if ok else 1

    if args.list:
        return list_profiles(service)

    if args.current:
        return show_current(service)

    if args.delete:
        ok, message = service.delete_profile(args.delete)
        print(message)
        return 0 if ok else 1

    if args.stream:
        profile, value = args.stream
        enabled = value.lower() in {"on", "true", "1", "y", "yes"}
        ok, message = service.set_stream(profile, enabled)
        print(message)
        return 0 if ok else 1

    return menu(service)


if __name__ == "__main__":
    sys.exit(main())
