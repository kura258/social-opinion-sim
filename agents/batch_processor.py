import json
import asyncio
from typing import List, Dict, Any

# 6个分组映射：请根据实际 Agent name/role 调整
GROUP_MAPPING = {
    # 官方/媒体组
    "Official_Media": "Official", "Local_News": "Official", "BrandOfficial": "Official",
    # KOL组
    "KOL_Tech": "KOL", "KOL_Social": "KOL", "KOL_Finance": "KOL", "KOL": "KOL",
    # 杠精/反对方
    "Troll_Hater": "Troll", "Troll_Skeptic": "Troll", "Troll": "Troll",
    # 粉丝/支持方
    "Defender_Fan": "Defender", "Defender_Patriot": "Defender",
    # 吃瓜群众 - 身份组
    "Crowd_Student": "Crowd_Identity", "Crowd_Worker": "Crowd_Identity",
    "Crowd_Mom": "Crowd_Identity", "Crowd_Uncle": "Crowd_Identity",
    # 吃瓜群众 - 行为组
    "Crowd_Gossip": "Crowd_Behavior", "Crowd_Emotional": "Crowd_Behavior",
    "Crowd_Silent": "Crowd_Behavior", "Crowd_Expert": "Crowd_Behavior",
}


class BatchActionProcessor:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def run_batch(self, active_agents: List[Any], env_context: dict, observed_common: List[dict]) -> Dict[str, dict]:
        if not active_agents:
            return {}

        groups: Dict[str, List[Any]] = {}
        for ag in active_agents:
            gid = GROUP_MAPPING.get(ag.name, GROUP_MAPPING.get(ag.role, "Default"))
            groups.setdefault(gid, []).append(ag)

        tasks = [
            self._process_group(gid, agents, env_context, observed_common)
            for gid, agents in groups.items()
        ]
        results = await asyncio.gather(*tasks)

        final_map: Dict[str, dict] = {}
        for res in results:
            final_map.update(res)
        return final_map

    async def _process_group(self, group_id: str, agents: List[Any], env_context: dict, observed_common: List[dict]) -> Dict[str, dict]:
        agents_data = [ag.get_batch_context(observed_common, env_context) for ag in agents]
        system_prompt = self._build_system_prompt(group_id)
        user_prompt = self._build_user_prompt(agents_data, env_context)

        temp = 0.1 if group_id == "Official" else 0.9

        try:
            response = await self.llm.chat_async(system_prompt, user_prompt, temperature=temp)
            return self._parse_json(response, [ag.name for ag in agents])
        except Exception as e:
            print(f"[Batch] Group {group_id} failed: {e}")
            return {}

    def _build_system_prompt(self, group_id: str) -> str:
        base = (
            f"你是一个多智能体社交模拟器。当前你代表【{group_id}】用户组。\n"
            "原则：\n"
            "1. 严格角色扮演：必须基于 unique_profile 和 memory_context 发言。\n"
            "2. 格式：仅输出标准 JSON List，不要包含 markdown 代码块标记。\n"
            "3. 【行为阈值】：并不是每个人每个时刻都要发言！\n"
            "   - 如果话题与我不相关或没意思，必须选择 action: 'silent'。\n"
            "   - 只有表达欲强烈时才 'post'。\n"
            "4. 风格指令："
        )
        if group_id == "Official":
            base += "严肃、客观、权威。使用“通报”、“据悉”，不带个人情绪。"
        elif group_id == "KOL":
            base += "观点鲜明。区分“理中客”（逻辑分析）与“营销号”（情绪煽动）。"
        elif group_id == "Troll":
            base += "【攻击性强】。喜欢抬杠、阴阳怪气、嘲讽（“笑死”、“不会吧”），散布阴谋论。"
        elif group_id == "Defender":
            base += "【防御性强】。立场坚定，护短，反击负面言论（“抱走不约”、“不信谣”）。"
        elif group_id == "Crowd_Identity":
            base += "【生活化】。强调身份视角（打工人/宝妈），关注切身利益，语气接地气。"
        elif group_id == "Crowd_Behavior":
            base += "【网络化】。吃瓜乐子人，使用流行语（emo/破防/666），或纯围观。"
        else:
            base += "口语化，像真实用户。"
        return base

    def _build_user_prompt(self, agents_data: List[dict], env_context: dict) -> str:
        agents_json = json.dumps(agents_data, ensure_ascii=False, indent=2)
        return (
            f"【环境信息】\n"
            f"舆论阶段：{env_context.get('phase')}\n"
            f"环境紧张度(0-1)：{env_context.get('global_tension', 0.0):.2f} (越高越敏感)\n"
            f"话题热度参考：{env_context.get('topic_heats')}\n\n"
            f"【任务】\n"
            f"作为以下 {len(agents_data)} 位用户，决定是否发声。大多数用户在非高潮期应选择 silent。\n"
            f"{agents_json}\n\n"
            "【输出格式】\n"
            "[\n"
            "  {\"agent_id\": \"ID\", \"action\": \"post/retweet/silent\", \"content\": \"...\", \"sentiment\": \"NEUTRAL\", \"topic\": \"...\"}, ...\n"
            "]"
        )

    def _parse_json(self, text: str, expected_ids: List[str]) -> Dict[str, dict]:
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            res: Dict[str, dict] = {}
            if isinstance(data, list):
                for item in data:
                    aid = item.get("agent_id")
                    if aid in expected_ids:
                        res[aid] = item
            return res
        except Exception:
            print(f"[Batch] JSON Parse Error. Raw: {text[:50]}...")
            return {}
