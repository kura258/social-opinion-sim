from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import time
import random

import networkx as nx
import numpy as np
import pandas as pd

from config.settings import DEFAULT_REAL_DATA_PATH
from config.personas import PERSONAS
from utils.topic_helper import generate_topic_background

@dataclass
class Post:
    id: int
    author: str
    text: str
    sentiment: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    tag: str        # "rumor", "official", "user", ...
    time_step: int
    target_post_id: Optional[int] = None
    topic: Optional[str] = None


class TopicManager:
    """
    维护话题热度。
    【重要优化】移除基础统计干扰，严格遵循训练出的纯 Hawkes 范式。
    """

    def __init__(
        self,
        topics: Sequence[str],
        hawkes_params: Optional[Dict[str, float]] = None,
        initial_heats: Optional[Dict[str, float]] = None,
    ):
        params = hawkes_params or {}
        # 强制将基础统计项设为 0，消除模型结构偏差
        self.alpha_v = 0.0
        self.beta_c = 0.0
        self.gamma_r = 0.0
        # 训练集热度量级，严格用于缩放
        self.heat_scale = params.get("heat_scale", 1.0)
        # 双衰减核参数（与训练保持一致）
        self.H_base = params.get("H_base", 0.0)
        self.mu_fast = params.get("mu_fast", params.get("mu", 0.5))
        self.mu_slow = params.get("mu_slow", 0.2)
        self.lambda_fast = params.get("lambda_fast", params.get("lambda", 1.0))
        self.lambda_slow = params.get("lambda_slow", 0.3)
        # 保证衰减正且 fast > slow
        self.lambda_fast = max(self.lambda_fast, 1e-6)
        self.lambda_slow = max(self.lambda_slow, 1e-6)
        if self.lambda_slow >= self.lambda_fast:
            self.lambda_slow = max(self.lambda_fast * 0.8, 1e-6)

        init_map = initial_heats or {}
        self.topics: Dict[str, Dict[str, Any]] = {}
        for topic in topics:
            start_heat = init_map.get(topic, self.H_base * self.heat_scale)
            norm_heat = start_heat / max(self.heat_scale, 1.0)
            approx_mem = max(0.0, (norm_heat - self.H_base) / (self.mu_fast + self.mu_slow + 1e-6))
            self.topics[topic] = {
                "heat": start_heat,
                "posts": [],
                "events": [],
                "per_step": {},
                "mem_fast": approx_mem,
                "mem_slow": approx_mem,
                "last_time": None,
            }

    def _compute_base(self, topic: str, current_time: int) -> float:
        return 0.0

    def _compute_heat(self, topic: str, current_time: int) -> float:
        tdata = self.topics[topic]
        base = self._compute_base(topic, current_time)
        hawkes = self.H_base + self.mu_fast * tdata["mem_fast"] + self.mu_slow * tdata["mem_slow"]
        return (base + hawkes) * self.heat_scale

    def decay_to(self, topic: str, current_time: int) -> None:
        """
        将记忆衰减到当前时间步（即便本步没有新事件），保证 Hawkes 递推的时间一致性。
        """
        if topic not in self.topics:
            return
        tdata = self.topics[topic]
        last_time = tdata["last_time"]
        dt = (current_time - last_time) if last_time is not None else 0
        if dt <= 0:
            return
        decay_fast = math.exp(-self.lambda_fast * dt)
        decay_slow = math.exp(-self.lambda_slow * dt)
        tdata["mem_fast"] = tdata["mem_fast"] * decay_fast
        tdata["mem_slow"] = tdata["mem_slow"] * decay_slow
        tdata["last_time"] = current_time

    def add_post(
        self,
        topic: str,
        post_content: str,
        current_time: int = 0,
        reach: float = 0.0,
        count: float = 1.0,
    ) -> None:
        """
        记录新增帖子并刷新热度；reach 表示影响范围，可简单累加；count 表示该帖代表的事件权重。
        Hawkes 记忆项使用归一化计数，防止数值爆炸。
        """
        if topic not in self.topics:
            return
        tdata = self.topics[topic]
        tdata["posts"].append(post_content)
        tdata["events"].append(current_time)
        step_stats = tdata["per_step"].setdefault(current_time, {"V": 0, "C": 0, "R": 0})
        step_stats["V"] += count
        step_stats["R"] += reach * count
        # 双衰减记忆更新（使用归一化 count）
        norm_count = count / max(self.heat_scale, 1.0)
        last_time = tdata["last_time"]
        dt = (current_time - last_time) if last_time is not None else 0
        decay_fast = math.exp(-self.lambda_fast * dt) if dt > 0 else 1.0
        decay_slow = math.exp(-self.lambda_slow * dt) if dt > 0 else 1.0
        tdata["mem_fast"] = tdata["mem_fast"] * decay_fast + float(norm_count)
        tdata["mem_slow"] = tdata["mem_slow"] * decay_slow + float(norm_count)
        tdata["last_time"] = current_time

        tdata["heat"] = self._compute_heat(topic, current_time)

    def get_heat(self, topic: str, current_time: Optional[int] = None) -> float:
        """
        获取当前热度；若提供 current_time，则先衰减至当前时间步后返回。
        """
        if topic not in self.topics:
            return 0.0
        if current_time is not None:
            self.decay_to(topic, current_time)
            self.topics[topic]["heat"] = self._compute_heat(topic, current_time)
        return self.topics.get(topic, {}).get("heat", 0.0)


class SocialEnv:
    """
    社交环境：
    - G：关注关系图
    - agents：name -> Agent
    - posts：历史帖子列表
    - topic_manager：记录各话题热度并驱动双衰减 Hawkes 记忆
    - 每步计算 lifecycle phase / global_tension，并注入 env_context 影响 Agent 决策
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        graph: nx.DiGraph,
        topics: Optional[Sequence[str]] = None,
        hawkes_params: Optional[Dict[str, float]] = None,
        llm_client=None,
        fixed_heat_scale: Optional[float] = None,
        initial_topic_heats: Optional[Dict[str, float]] = None,
    ):
        self.llm_client = llm_client
        self.agents = agents
        self.G = graph
        self.posts: List[Post] = []
        self.t = 0
        self._next_post_id = 1
        self._topics = list(topics) if topics else []
        # 统一处理参数，确保 heat_scale 与真实数据量级对齐
        params = dict(hawkes_params or {})
        inferred_scale = fixed_heat_scale if fixed_heat_scale and fixed_heat_scale > 0 else params.get("heat_scale")
        if not inferred_scale or inferred_scale <= 1.0:
            inferred_scale = self._infer_data_scale(self._topics)
        params["heat_scale"] = inferred_scale
        self.data_scale = inferred_scale
        self._hawkes_params = params
        self.topic_manager: Optional[TopicManager] = (
            TopicManager(self._topics, params, initial_topic_heats) if self._topics else None
        )
        # 话题背景
        self.topic_backgrounds: Dict[str, str] = {}
        for t in self._topics:
            self.topic_backgrounds[t] = generate_topic_background(t, self.llm_client) if llm_client else f"关于 {t} 的讨论"
        # 角色分配占比（用于每个话题的配额拆分）
        self.role_distribution = {
            "Crowd": 0.85,
            "Defender": 0.08,
            "Troll": 0.04,
            "KOL": 0.02,
            "BrandOfficial": 0.01,
        }
        self._agent_last_action: Dict[str, int] = {name: 0 for name in agents}
        # 记录上一轮真实总量（未归一化）
        self._last_step_real_volume: float = 0.0
        hp = hawkes_params or {}
        self.params = {
            "mu_fast": hp.get("mu_fast", 0.5),
            "mu_slow": hp.get("mu_slow", 0.2),
            "H_base": hp.get("H_base", 0.0),
            "lambda_fast": hp.get("lambda_fast", 1.0),
            "lambda_slow": hp.get("lambda_slow", 0.3),
        }
        self.M_fast = 0.0
        self.M_slow = 0.0
        self.current_intensity = 0.0
        self.phase = "Incubation"
        self.official_has_spoken = False
    def reset(self):
        self.posts = []
        self.t = 0
        self._next_post_id = 1
        if self._topics:
            self.topic_manager = TopicManager(self._topics, self._hawkes_params)
        self._agent_last_action = {name: 0 for name in self.agents}
        self._last_step_real_volume = 0.0
        self.M_fast = 0.0
        self.M_slow = 0.0
        self.current_intensity = 0.0
        self.phase = "Incubation"
        self.official_has_spoken = False

    def _compute_reach(self, author: str) -> int:
        """简单地以关注入度作为传播影响力近似。"""
        try:
            return int(self.G.in_degree(author))
        except Exception:
            return 0

    def _update_hawkes_state(self, new_posts_count: int) -> None:
        """
        使用双衰减核递推更新 Hawkes 状态，无需回溯历史。
        """
        self.M_fast = self.M_fast * np.exp(-self.params["lambda_fast"]) + new_posts_count
        self.M_slow = self.M_slow * np.exp(-self.params["lambda_slow"]) + new_posts_count
        intensity = (
            self.params["H_base"]
            + self.params["mu_fast"] * self.M_fast
            + self.params["mu_slow"] * self.M_slow
        )
        self.current_intensity = intensity

    def _determine_phase(self, total_heat: float) -> str:
        """
        根据热度与官方发声情况判定生命周期阶段。
        """
        # 官方在上一轮是否发声
        recent_official = [
            p for p in self.posts if p.time_step == self.t - 1 and p.author == "BrandOfficial"
        ]
        if recent_official:
            self.official_has_spoken = True

        if self.official_has_spoken:
            return "Climax"
        if total_heat > 2000:
            return "Fermentation"
        if total_heat > 500:
            return "Diffusion"
        return "Incubation"

    def _add_post(
        self,
        author: str,
        text: str,
        sentiment: str,
        tag: str,
        target_post_id: Optional[int] = None,
        topic: Optional[str] = None,
        count: float = 1.0,
    ):
        post = Post(
            id=self._next_post_id,
            author=author,
            text=text,
            sentiment=sentiment,
            tag=tag,
            time_step=self.t,
            target_post_id=target_post_id,
            topic=topic,
        )
        self._next_post_id += 1
        self.posts.append(post)

        if self.topic_manager and topic:
            reach = self._compute_reach(author)
            self.topic_manager.add_post(topic, text, current_time=self.t, reach=reach, count=count)

    def step_legacy(self, pr_strategy=None, request_delay: float = 0.0):
        """（保留旧版）推进一个时间步：更新热度/强度/phase，构造 env_context，再按权重驱动 Agent 发声。"""
        self.t += 1
        new_posts: List[Post] = []
        observed = [
            {
                "id": p.id,
                "author": p.author,
                "text": p.text,
                "summary": p.text,
                "sentiment": p.sentiment,
                "tag": p.tag,
                "topic": p.topic,
            }
            for p in self.posts
            if p.time_step == self.t - 1
        ]
        posts_last_step = [p for p in self.posts if p.time_step == self.t - 1]

        # ---- 预计算环境信号 ----
        total_heat = 0.0
        if self.topic_manager and self._topics:
            total_heat = sum(self.topic_manager.get_heat(tp) for tp in self._topics)
        self.phase = self._determine_phase(total_heat)
        self._update_hawkes_state(len(posts_last_step))
        global_tension = float(np.tanh(self.current_intensity / 100.0))

        # ---- 环境上下文传递给 Agent ----
        env_context = {
            "phase": self.phase,
            "global_tension": global_tension,
            "is_official_intervened": self.official_has_spoken,
        }

        # ---- 基于 Hawkes 参数 + 角色画像的权重调度 ----
        mu_fast = self._hawkes_params.get("mu_fast", 0.5) if self._hawkes_params else 0.5
        mu_slow = self._hawkes_params.get("mu_slow", 0.2) if self._hawkes_params else 0.2
        lambda_fast = self._hawkes_params.get("lambda_fast", 1.0) if self._hawkes_params else 1.0
        lambda_slow = self._hawkes_params.get("lambda_slow", 0.3) if self._hawkes_params else 0.3
        base_map = {
            "BrandOfficial": 0.35 * mu_slow,
            "KOL": 0.30 * mu_fast,
            "Troll": 0.15 * mu_fast,
            "Defender": 0.10 * mu_fast,
            "Crowd": 0.35 * mu_slow,
        }
        fast_roles = {"KOL", "Troll", "Defender"}
        alpha_heat = 0.12  # 热度对出场概率的放大系数（更温和）

        weights = []
        names = []
        for name, agent in self.agents.items():
            role = getattr(agent, "role", "")
            base = base_map.get(role, 0.2 * mu_slow)
            lam = lambda_fast if role in fast_roles else lambda_slow
            last_t = self._agent_last_action.get(name, 0)
            dt = max(self.t - last_t, 0)
            decay = math.exp(-lam * dt) if dt > 0 else 1.0

            # 以 agent 关注的话题平均热度作为加权因子
            heat_boost = 1.0
            if self.topic_manager and getattr(agent, "topics", None):
                heats = [self.topic_manager.get_heat(tp) for tp in agent.topics]
                if heats:
                    avg_heat = sum(heats) / len(heats)
                    heat_boost = 1.0 + alpha_heat * math.log1p(avg_heat)

            w_eff = base * decay * heat_boost
            weights.append(max(w_eff, 0.0))
            names.append(name)

        active_agents: List[str] = []
        if weights and sum(weights) > 0:
            total_w = sum(weights)
            norm_w = [w / total_w for w in weights]
            # 使用 Hawkes intensity 动态控制参与人数，爆发期更多 Agent 入场
            max_intensity_ref = 100.0  # 依据训练热度量级设置的基准值
            intensity_ratio = 0.0
            if max_intensity_ref > 0:
                intensity_ratio = min(max(self.current_intensity / max_intensity_ref, 0.0), 1.0)
            base_activity = 0.2  # 潜伏期保证更高基线出场率
            dynamic_activity = base_activity + 0.8 * intensity_ratio  # 最高可达约 100%
            target_k = max(2, int(len(names) * dynamic_activity))
            pool = list(zip(names, norm_w))
            while pool and len(active_agents) < target_k:
                cand = random.choices(pool, weights=[w for _, w in pool], k=1)[0][0]
                active_agents.append(cand)
                pool = [(n, w) for n, w in pool if n != cand]
        else:
            active_agents = list(self.agents.keys())

        for name in active_agents:
            agent = self.agents[name]
            if request_delay > 0:
                time.sleep(request_delay)

            # 用环境信号驱动 agent 的混合决策
            signal_post = type("EnvSignal", (), {})()
            setattr(signal_post, "heat", total_heat)
            # æœªæœ‰å…¬å¼€å†³å®šçš„æ–‡æ¡£æ—¶ï¼Œé»˜è®¤ä»¥æ™®é€šç”¨æˆ·èº«ä»½è§†ä¸ºä¿¡æ¯æ¥æºï¼Œä¸è®¾ä¸º rumor
            setattr(signal_post, "author_role", "official" if self.official_has_spoken else "user")
            setattr(signal_post, "is_verified", self.official_has_spoken)
            if hasattr(agent, "driver_mode") and hasattr(agent, "decide_action"):
                try:
                    decision = agent.decide_action(signal_post, env_context)
                except Exception:
                    decision = None
            else:
                decision = None

            if pr_strategy and name == "BrandOfficial":
                action = pr_strategy.decide_brand_action(self.t, agent, observed)
            else:
                if decision and isinstance(decision, tuple):
                    if decision[0] == "SILENCE":
                        # Reflex 模式直接沉默；Brain 模式继续走完整社交决策
                        if getattr(agent, "driver_mode", "") == "reflex":
                            continue
                        action = agent.decide_social_action(
                            self.t, observed, environment=self, env_context=env_context
                        )
                    if decision[0] == "REPOST":
                        # Reflex 模式快速转发
                        action = {
                            "action": "retweet",
                            "post_text": "",
                            "sentiment": "NEUTRAL",
                            "target_post_id": None,
                            "topic": None,
                        }
                    else:
                        action = agent.decide_social_action(
                            self.t, observed, environment=self, env_context=env_context
                        )
                else:
                    action = agent.decide_social_action(
                        self.t, observed, environment=self, env_context=env_context
                    )

            if action is None:
                continue

            act_type = action.get("action") or action.get("type")
            if act_type == "post":
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag=action.get("tag", "user"),
                    topic=action.get("topic"),
                )
                self._agent_last_action[name] = self.t
                new_posts.append(self.posts[-1])
            elif act_type == "retweet":
                target_id = action.get("target_post_id")
                self._add_post(
                    author=name,
                    text=action.get("post_text", action.get("text", "")),
                    sentiment=action.get("sentiment", "NEUTRAL"),
                    tag="retweet",
                    target_post_id=target_id,
                    topic=action.get("topic"),
                )
                self._agent_last_action[name] = self.t
                new_posts.append(self.posts[-1])
            else:
                continue
        return new_posts

    def _infer_data_scale(self, topics: Sequence[str]) -> float:
        """
        根据默认真实数据计算所选话题的最大热度作为 scale，避免归一化过度。
        """
        if not topics or not DEFAULT_REAL_DATA_PATH.exists():
            return 1.0
        try:
            df = pd.read_csv(DEFAULT_REAL_DATA_PATH)
            df = df[df["topic"].isin(topics)]
            if df.empty or "heat" not in df.columns:
                return 1.0
            return float(df["heat"].max())
        except Exception:
            return 1.0

    def step(self, pr_strategy=None, request_delay: float = 0.0):
        """
        [Final Fix] 分话题确定性调度：逐话题计算预测热度并刚性派发给 Agent。
        """
        self.t += 1
        new_posts: List[Post] = []

        # --- 构造上下文 & 预测每个话题的目标热度 ---
        topic_heats: Dict[str, float] = {}
        if self.topic_manager and self._topics:
            for tp in self._topics:
                topic_heats[tp] = self.topic_manager.get_heat(tp, current_time=self.t)
        total_heat = sum(topic_heats.values())
        self.phase = self._determine_phase(total_heat)

        posts_last_step = [p for p in self.posts if p.time_step == self.t - 1]
        observed = [
            {"id": p.id, "author": p.author, "text": p.text, "tag": p.tag, "topic": p.topic}
            for p in posts_last_step
        ]

        # --- 核心：按话题拆分目标量并强制执行（精英制，限发言人数） ---
        total_weight_ratio = sum(getattr(a, "weight_ratio", 1.0) for a in self.agents.values()) or 1.0
        MAX_POSTS_PER_TOPIC = 3
        for topic, target_heat in topic_heats.items():
            if target_heat < (self.data_scale * 0.0001):
                continue
            num_speakers = min(MAX_POSTS_PER_TOPIC, len(self.agents))
            quota_per_speaker = target_heat / max(num_speakers, 1)
            candidates = list(self.agents.keys())
            weights = [getattr(self.agents[n], "weight_ratio", 1.0) for n in candidates]
            w_sum = sum(weights)
            probs = [w / w_sum for w in weights] if w_sum > 0 else None
            chosen_names = np.random.choice(candidates, size=num_speakers, replace=False, p=probs)

            for name in chosen_names:
                agent = self.agents[name]
                ctx = {
                    "time": self.t,
                    "phase": self.phase,
                    "topic_bg": self.topic_backgrounds.get(topic, f"关于 {topic} 的热议"),
                    "tension": min(target_heat / max(self.data_scale, 1.0), 1.0),
                }
                topic_related_posts = [
                    p for p in self.posts if p.topic == topic and p.time_step >= self.t - 5
                ]
                obs_for_topic = [
                    {"id": p.id, "author": p.author, "text": p.text, "topic": p.topic}
                    for p in topic_related_posts[-3:]
                ]

                action = None
                if hasattr(agent, "force_action_on_topic"):
                    try:
                        action = agent.force_action_on_topic(topic, ctx, obs_for_topic, quota_per_speaker)
                    except Exception:
                        action = None

                if action and action.get("action") in ["post", "retweet"]:
                    act_type = action.get("action")
                    sentiment = action.get("sentiment", "NEUTRAL")
                    tag = action.get("tag", "user" if act_type == "post" else "retweet")
                    post_text = action.get("post_text", action.get("text", ""))
                    target_id = action.get("target_post_id")
                    self._add_post(
                        author=name,
                        text=post_text,
                        sentiment=sentiment,
                        tag=tag,
                        target_post_id=target_id,
                        topic=topic,
                        count=quota_per_speaker,
                    )
                    self._agent_last_action[name] = self.t
                    new_posts.append(self.posts[-1])

        return new_posts
