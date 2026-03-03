from __future__ import annotations

from ai_assistant.command_rules import build_cli_command_rules_prompt
from ai_assistant.services.ai_client import AIClient
from ai_assistant.services.ai_gateway import AIGateway
from ai_assistant.services.context_service import ContextService
from ai_assistant.services.history_service import HistoryService


class ChatService:
    def __init__(
        self,
        ai_client: AIClient | None = None,
        history_service: HistoryService | None = None,
        context_service: ContextService | None = None,
    ) -> None:
        self.ai_client = ai_client or AIClient()
        self.ai_gateway = AIGateway(self.ai_client)
        self.history_service = history_service or HistoryService()
        self.context_service = context_service or ContextService()

    def clear_history(self) -> str:
        self.history_service.clear()
        return "✅ 历史已清空"

    def _build_extra_system_messages(self) -> list[str]:
        messages = [build_cli_command_rules_prompt()]
        context_block = self.context_service.render_context_block(max_chars=12000)
        if context_block:
            messages.append(f"当前会话激活了代码上下文，回答相关问题时必须优先结合该上下文：\n{context_block}")
        return messages

    def chat(self, message: str, use_history: bool = True) -> str:
        if not message:
            return "❌ 请输入消息"

        _ = use_history
        # 为保持兼容仍保留 use_history 参数，但默认会始终参考历史与上下文。
        self.history_service.trim_and_summarize(self.ai_gateway.summarize_messages)
        history_messages = self.history_service.build_messages_for_request(
            user_prompt=message,
            include_recent_history=True,
            include_recent_events=True,
            extra_system_messages=self._build_extra_system_messages(),
        )

        response = self.ai_gateway.chat(
            history_messages,
            stream_override=None,
            print_stream=True,
        )
        self.history_service.append_exchange(message, response.content)
        return response.content
