import random
from typing import Any, Dict, List, Optional

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
        self.weight_ratio = 1.0
        self.emotion = 0.5
        self.social_confidence = 0.5

    def _suggest_topic(self, env_context: dict) -> str:
        if not self.topics:
            return "General"

        topic_heats = (env_context or {}).get("topic_heats") or {}
        if isinstance(topic_heats, dict) and topic_heats:
            candidates = [t for t in self.topics if t in topic_heats]
            if candidates:
                weights = [float(topic_heats.get(t, 0.0)) + 1e-6 for t in candidates]
                try:
                    return random.choices(candidates, weights=weights, k=1)[0]
                except Exception:
                    return random.choice(candidates)

        return random.choice(self.topics)

    def get_batch_context(self, observed_posts: List[dict], env_context: dict) -> dict:
        obs_summary = self._summarize_observation(observed_posts)
        suggested_topic = self._suggest_topic(env_context)
        return {
            "agent_id": self.name,
            "role": self.role,
            "unique_profile": self.profile,
            "memory_context": self.memory.get_context_for_prompt(),
            "observation": obs_summary,
            "observation_items": [
                {"id": p.get("id"), "author": p.get("author"), "topic": p.get("topic"), "text": (p.get("text") or "")[:120]}
                for p in (observed_posts or [])[-10:]
            ],
            "candidate_topics": list(self.topics),
            "suggested_topic": suggested_topic,
            "current_state": {"emotion": self.emotion, "confidence": self.social_confidence},
            "topic_backgrounds": env_context.get("topic_backgrounds", {}),
            "cold_topics": env_context.get("cold_topics", []),
        }

    def apply_batch_result(
        self,
        action_result: dict,
        time_step: int,
        observed_posts: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        if action_result is None:
            action_result = {}

        emotion_change = action_result.get("emotion_change")
        if emotion_change is not None:
            try:
                self.emotion = max(0.0, min(1.0, self.emotion + float(emotion_change)))
            except (TypeError, ValueError):
                pass

        confidence_change = action_result.get("confidence_change")
        if confidence_change is not None:
            try:
                self.social_confidence = max(0.0, min(1.0, self.social_confidence + float(confidence_change)))
            except (TypeError, ValueError):
                pass

        raw_action = (action_result.get("action", "silent") or "silent").lower()
        content = action_result.get("content", "") or ""
        topic = action_result.get("topic")
        original_topic = topic
        suggested_topic = action_result.get("suggested_topic")

        final_action_type = "silent"
        target_post_id = None

        if raw_action in ["like", "comment", "retweet"]:
            if observed_posts:
                target_post_id = action_result.get("target_post_id") or action_result.get("target_postId")
                target_hint = action_result.get("target_user", "") or ""
                preferred_topic = (
                    topic
                    if topic
                    else (suggested_topic if isinstance(suggested_topic, str) and suggested_topic.strip() else None)
                ) or (self.topics[0] if self.topics else None)

                target = None
                if target_post_id is not None:
                    try:
                        target_post_id = int(target_post_id)
                    except Exception:
                        target_post_id = None
                if target_post_id is not None:
                    for p in observed_posts:
                        if p.get("id") == target_post_id:
                            target = p
                            break

                if target is None:
                    candidates = [
                        p
                        for p in observed_posts
                        if (target_hint and target_hint in str(p.get("author", ""))) and (p.get("topic") == preferred_topic)
                    ]
                    if not candidates and target_hint:
                        candidates = [p for p in observed_posts if target_hint in str(p.get("author", ""))]
                    if not candidates and preferred_topic:
                        candidates = [p for p in observed_posts if p.get("topic") == preferred_topic]
                    if not candidates:
                        candidates = observed_posts
                    target = random.choice(candidates)

                target_post_id = target.get("id")
                topic = target.get("topic", topic)
                final_action_type = raw_action
            else:
                final_action_type = "silent"
        elif raw_action == "post":
            final_action_type = "post"
        else:
            final_action_type = "silent"

        if final_action_type in ["post", "retweet", "comment"] and not content.strip():
            final_action_type = "silent"
            content = ""

        if final_action_type == "like":
            content = "[Like]"
        elif final_action_type == "silent":
            content = ""

        # 强制 Post 使用 suggested_topic，避免 topic/text 失配（LLM 未给 topic 时也能锁定）
        if final_action_type == "post":
            if isinstance(suggested_topic, str) and suggested_topic.strip():
                topic = suggested_topic.strip()

        if final_action_type != "silent":
            if (not topic) or (isinstance(topic, str) and topic.strip().lower() in ("null", "none", "")):
                topic = original_topic or (random.choice(self.topics) if self.topics else "未标注")

            self.memory.add(f"在 t={time_step} {final_action_type}: {content}")
            self.last_act_time = time_step

        return {
            "agent_id": self.name,
            "action_type": final_action_type,
            "content": content,
            "sentiment": action_result.get("sentiment", "NEUTRAL"),
            "topic": topic if topic else (self.topics[0] if self.topics else "未标注"),
            "target_post_id": target_post_id,
        }

    def _summarize_observation(self, posts: List[dict]) -> str:
        if not posts:
            return "当前无新消息。"
        recent = posts[-5:]
        return "; ".join([f"[{p.get('topic','未标注')}] {p['author']}: {p['text'][:30]}..." for p in recent])

    async def check_memory_maintenance(self):
        if self.memory.needs_consolidation():
            await self.memory.consolidate(self.llm_client)
