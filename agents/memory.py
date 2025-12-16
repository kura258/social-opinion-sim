import random
from typing import List

class Memory:
    def __init__(self, agent_name: str, buffer_limit: int = 5):
        self.agent_name = agent_name
        self.long_term_summary: str = "暂无过往经历。"
        self.short_term_buffer: List[str] = []
        # 随机化阈值（3-7之间），防止所有 Agent 同时触发整理导致流量尖峰
        self.buffer_limit: int = max(3, buffer_limit + random.randint(-1, 2))

    def add(self, text: str):
        self.short_term_buffer.append(text)

    def get_context_for_prompt(self) -> str:
        """为 Prompt 准备的记忆上下文"""
        if not self.short_term_buffer:
            recent = "（无近期动态）"
        else:
            recent = "\n".join([f"- {m}" for m in self.short_term_buffer])
        
        return (
            f"【长期记忆摘要】：{self.long_term_summary}\n"
            f"【近期短期记忆】：\n{recent}"
        )

    def needs_consolidation(self) -> bool:
        return len(self.short_term_buffer) >= self.buffer_limit

    async def consolidate(self, llm_client):
        """调用 LLM 压缩记忆"""
        if not self.short_term_buffer:
            return

        buffer_text = "\n".join(self.short_term_buffer)
        system_prompt = "你是一个记忆整理助手。请将用户的近期经历整合进长期记忆摘要中。"
        user_prompt = (
            f"当前长期记忆：{self.long_term_summary}\n\n"
            f"新增近期经历：\n{buffer_text}\n\n"
            f"任务：更新长期记忆。保留关键观点、情感变化和重要事件，去除无关细节。使用第三人称描述 {self.agent_name}。"
            f"输出控制在 100 字以内。"
        )

        try:
            # 注意：此处假设 llm_client 实现了 chat_async，如未实现请看第四步
            new_summary = await llm_client.chat_async(system_prompt, user_prompt)
            self.long_term_summary = new_summary
            # 保留最后一条以保持连贯性
            last_item = self.short_term_buffer[-1]
            self.short_term_buffer = [last_item] 
        except Exception as e:
            print(f"[Memory] Consolidation failed for {self.agent_name}: {e}")