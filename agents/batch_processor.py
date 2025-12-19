import json
import asyncio
from typing import List, Dict, Any

# Group mapping for batch prompts
GROUP_MAPPING = {
    "Official_Media": "Official", "Local_News": "Official", "BrandOfficial": "Official",
    "KOL_Tech": "KOL", "KOL_Social": "KOL", "KOL_Finance": "KOL", "KOL": "KOL",
    "Troll_Hater": "Troll", "Troll_Skeptic": "Troll", "Troll": "Troll",
    "Defender_Fan": "Defender", "Defender_Patriot": "Defender",
    "Crowd_Student": "Crowd_Identity", "Crowd_Worker": "Crowd_Identity",
    "Crowd_Mom": "Crowd_Identity", "Crowd_Uncle": "Crowd_Identity",
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
        base = f"""????????????????????{group_id}?????
???
1. ??????????? unique_profile ? memory_context ???
2. ????? JSON List????? markdown ??????
3. ??????????????????????????????
   - ??????/??????????????
   - action ??? post / retweet / silent?
   - ? action ? post/retweet????? topic?????? candidate_topics ?????
   - ??????????????????????
4. ?????"""
        if group_id == "Official":
            base += "???????????????????????????"
        elif group_id == "KOL":
            base += "????????????????"
        elif group_id == "Troll":
            base += "????????????????????????"
        elif group_id == "Defender":
            base += "????????????????????"
        elif group_id == "Crowd_Identity":
            base += "?????????????????????????"
        elif group_id == "Crowd_Behavior":
            base += "??/???????????????"
        else:
            base += "??????????"
        return base

    def _build_user_prompt(self, group_id: str, agents_data: List[dict], env_context: dict) -> str:
        agents_json = json.dumps(agents_data, ensure_ascii=False, indent=2)
        visibility = float(env_context.get("visibility", 0.0) or 0.0)
        global_scale = float(env_context.get("global_scale", 0.0) or 0.0)
        phase = env_context.get("phase")
        global_mood = float(env_context.get("global_mood", 0.0) or 0.0)
        target_n = self._desired_active_count(group_id, len(agents_data), visibility, global_scale, phase)
        topic_heats = env_context.get("topic_heats")
        diversity_hint = ""
        if isinstance(topic_heats, dict) and len(topic_heats) >= 2 and visibility >= 0.2:
            diversity_hint = "?????? 2 ??? topic????????????"
        return f"""??????
?????{phase}
?????(0-1)?{env_context.get('global_tension', 0.0):.2f} (?????)
???(0-1)?{visibility:.2f}
??????(0-?)?{global_scale:.3f}
?????global_mood 0-1??{global_mood:.2f}
???????{topic_heats}

??LLM????????
???? {len(agents_data)} ??????????->??->???
- Phase 1 (Sensory): ?? observation??? global_mood ??? emotion ?????????
- Phase 2 (Decision): ?? unique_profile ? social_confidence ???????confidence ?? emotion ????(?0.5)??? silent/browsing?confidence ?? emotion ??(??0/1)??? post/retweet?
- Phase 3 (Reflection): ????????????????????? emotion ? social_confidence?
- ???????????? {target_n} ??post/retweet???? silent?
- ?????? silent???????? silent?
- {diversity_hint}

{agents_json}

??????JSON List????? JSON??? markdown ????? item ???
[
  {
    "agent_id": "string",
    "thought": "brief reasoning",
    "emotion_change": float (-0.2 to 0.2),
    "confidence_change": float (-0.2 to 0.2),
    "action": "post" | "retweet" | "silent",
    "content": "string (content if action is post)"
  }, ...
]"""

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
            return "????????????????????????"
        if group_id == "KOL":
            return "?????????????????????????"
        if group_id == "Troll":
            return "????????????????????"
        if group_id == "Defender":
            return "??????????????????"
        return "???????????"

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
