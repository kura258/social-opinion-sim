from typing import List, Optional, Dict
from .memory import Memory

class Agent:
    def __init__(self, name: str, role: str, profile: str, llm_client, topics: Optional[List[str]] = None):
        self.name = name
        self.role = role
        self.profile = profile
        self.llm_client = llm_client
        self.topics = topics or []
        self.memory = Memory(name)
        self.last_act_time = 0
        # 兼容旧代码的属性
        self.weight_ratio = 1.0 

    def get_batch_context(self, observed_posts: List[dict], env_context: dict) -> dict:
        """打包 Agent 数据供批处理使用"""
        obs_summary = self._summarize_observation(observed_posts)
        return {
            "agent_id": self.name,
            "role": self.role,
            "unique_profile": self.profile, 
            "memory_context": self.memory.get_context_for_prompt(),
            "observation": obs_summary
        }

    def apply_batch_result(self, action_result: dict, time_step: int) -> dict:
        """接收批处理结果 -> 更新记忆 -> 返回 Action 供环境执行"""
        action_type = action_result.get("action", "silent").lower()
        content = action_result.get("content", "")

        # 兜底：如果说是 post 但没内容，转为 silent
        if action_type in ["post", "retweet"] and not content.strip():
            action_type = "silent"

        if action_type != "silent":
            # 记录到记忆
            self.memory.add(f"在 t={time_step} {action_type}: {content}")
            self.last_act_time = time_step
        
        return {
            "agent_id": self.name,
            "action_type": action_type, # 统一字段名
            "content": content,
            "sentiment": action_result.get("sentiment", "NEUTRAL"),
            "topic": action_result.get("topic", self.topics[0] if self.topics else "未标注"),
            "target_post_id": action_result.get("target_post_id")
        }

    def _summarize_observation(self, posts: List[dict]) -> str:
        if not posts: return "当前无新消息。"
        # 只取最近 5 条，避免上下文过长
        recent = posts[-5:]
        return "; ".join([f"{p['author']}: {p['text'][:30]}..." for p in recent])

    async def check_memory_maintenance(self):
        """维护任务：检查是否需要压缩记忆"""
        if self.memory.needs_consolidation():
            await self.memory.consolidate(self.llm_client)