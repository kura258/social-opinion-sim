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
        user_prompt = self._build_user_prompt(group_id, agents_data, env_context)

        temp = 0.1 if group_id == "Official" else 0.9

        try:
            response = await self.llm.chat_async(system_prompt, user_prompt, temperature=temp)
            parsed = self._parse_json(response, [ag.name for ag in agents])
            if not agents:
                return parsed
            if parsed:
                return parsed

            # 兜底：避免全员沉默/解析失败导致长期死火
            topic = self._pick_topic(env_context)
            fallback = agents[0]
            return {
                fallback.name: {
                    "agent_id": fallback.name,
                    "action": "post",
                    "content": self._fallback_content(group_id),
                    "sentiment": "NEUTRAL",
                    "topic": topic,
                }
            }
        except Exception as e:
            print(f"[Batch] Group {group_id} failed: {e}")
            return {}

    def _build_system_prompt(self, group_id: str) -> str:
        base = (
            f"你是一个多智能体社交模拟器。当前你代表【{group_id}】用户组。\n"
            "原则：\n"
            "1. 严格角色扮演：必须基于 unique_profile 和 memory_context 发言。\n"
            "2. 格式：仅输出标准 JSON List，不要包含 markdown 代码块标记。\n"
            "3. 【行为阈值】：不是每个人每个时刻都要发言，但也不要全员长期沉默。\n"
            "   - 若环境可见度/紧张度上升，请提高发声比例。\n"
            "   - action 只能是 post / retweet / silent。\n"
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

    def _build_user_prompt(self, group_id: str, agents_data: List[dict], env_context: dict) -> str:
        agents_json = json.dumps(agents_data, ensure_ascii=False, indent=2)
        visibility = float(env_context.get("visibility", 0.0) or 0.0)
        global_scale = float(env_context.get("global_scale", 0.0) or 0.0)
        phase = env_context.get("phase")
        target_n = self._desired_active_count(group_id, len(agents_data), visibility, global_scale, phase)
        topic_heats = env_context.get("topic_heats")
        return (
            f"【环境信息】\n"
            f"舆论阶段：{phase}\n"
            f"环境紧张度(0-1)：{env_context.get('global_tension', 0.0):.2f} (越高越敏感)\n"
            f"可见度(0-1)：{visibility:.2f}\n"
            f"全局缩放参考(0-∞)：{global_scale:.3f}\n"
            f"话题热度参考：{topic_heats}\n\n"
            f"【任务】\n"
            f"作为以下 {len(agents_data)} 位用户，决定是否发声。\n"
            f"- 本组本轮建议发声人数：约 {target_n} 人（post/retweet），其余为 silent。\n"
            f"- 如果话题没意思可以 silent，但不要所有人都 silent。\n"
            f"{agents_json}\n\n"
            "【输出格式】\n"
            "[\n"
            "  {\"agent_id\": \"ID\", \"action\": \"post/retweet/silent\", \"content\": \"...\", \"sentiment\": \"NEUTRAL\", \"topic\": \"...\"}, ...\n"
            "]"
        )

    def _desired_active_count(
        self,
        group_id: str,
        group_size: int,
        visibility: float,
        global_scale: float,
        phase: Any,
    ) -> int:
        vis = max(0.0, min(1.0, float(visibility)))
        gs = max(0.0, float(global_scale))
        base = 0.03 + 0.25 * vis
        if group_id in ("Troll", "Defender"):
            base = 0.10 + 0.40 * vis
        elif group_id == "KOL":
            base = 0.08 + 0.35 * vis
        elif group_id == "Official":
            base = 0.03 + 0.15 * vis
        elif group_id in ("Crowd_Identity", "Crowd_Behavior"):
            base = 0.05 + 0.25 * vis

        # 若全局缩放很高或已进入高潮阶段，允许更多人发声
        if gs > 1.0 or str(phase).lower() in ("fermentation", "climax"):
            base = max(base, 0.20 + 0.30 * vis)

        target = int(round(group_size * base))
        if group_size > 0:
            target = max(1, min(group_size, target))
        return target

    def _pick_topic(self, env_context: dict) -> str:
        topic_heats = env_context.get("topic_heats") or {}
        if isinstance(topic_heats, dict) and topic_heats:
            try:
                return max(topic_heats.items(), key=lambda kv: float(kv[1]))[0]
            except Exception:
                return next(iter(topic_heats.keys()))
        return "General"

    def _fallback_content(self, group_id: str) -> str:
        if group_id == "Official":
            return "关注到相关讨论，已在核实信息，请以权威通报为准。"
        if group_id == "KOL":
            return "这事热度又上来了，我先捋一捋逻辑，大家别急着站队。"
        if group_id == "Troll":
            return "不会吧，这也有人信？你们真就随便带节奏？"
        if group_id == "Defender":
            return "别带节奏了，等事实出来再说，别造谣。"
        return "围观一下，大家怎么看？"

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
