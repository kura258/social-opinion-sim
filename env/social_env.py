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


@dataclass
class EnvFields:
    t: int
    visibility: float  # V(t) from Hawkes
    reward: float      # R(t) derived from V
    risk: float        # M(t) derived from V/Sentiment
    global_scale: float  # Normalization for 20-agent variance
    traces: Dict[str, Any]

    @property
    def V(self) -> float:
        return self.visibility

    @property
    def R(self) -> float:
        return self.reward

    @property
    def M(self) -> float:
        return self.risk


@dataclass
class AgentAction:
    agent_id: str
    action_type: str  # "post", "retweet", "silent"
    content: str
    topic: str
    timestamp: int


class FieldGenerator:
    """
    Maps Hawkes intensity and sentiment into broadcastable environment fields.
    """

    def __init__(self, heat_scale: float = 100.0):
        self.heat_scale = max(heat_scale, 1e-6)
        self.current_time = 0

    def compute_fields(
        self,
        hawkes_intensity: float,
        sentiment: float,
        current_traces: Optional[Dict[str, Any]] = None,
    ) -> EnvFields:
        # Visibility: normalize intensity and squeeze into [0,1] via sigmoid
        norm_intensity = (hawkes_intensity / self.heat_scale - 0.5) * 5.0
        visibility = 1.0 / (1.0 + math.exp(-norm_intensity))

        # Reward potential scales with visibility
        reward = visibility * 1.5

        # Moderation risk: high when visibility is high; sentiment can modulate if desired
        risk = 1.0 if visibility > 0.8 else 0.0

        global_scale = min(1.0, max(hawkes_intensity / self.heat_scale, 0.0))
        traces = current_traces or {}

        return EnvFields(
            t=self.current_time,
            visibility=visibility,
            reward=reward,
            risk=risk,
            global_scale=global_scale,
            traces=traces,
        )


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
        real_heat_trajectory: Optional[Dict[str, List[float]]] = None,
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
        # 真实热度轨迹，用于数据制导
        self.real_heat_trajectory = real_heat_trajectory or {}
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
        self.field_generator = FieldGenerator(heat_scale=params.get("heat_scale", 100.0))
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
        self.field_generator = FieldGenerator(heat_scale=self.data_scale or 100.0)

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

    def _compute_sentiment_score(self, posts: List[Post]) -> float:
        """
        Roughly aggregate sentiment into [-1, 1] based on last-step posts.
        """
        if not posts:
            return 0.0
        mapping = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
        scores = []
        for p in posts:
            sentiment = (p.sentiment or "").upper()
            scores.append(mapping.get(sentiment, 0.0))
        return sum(scores) / max(len(scores), 1)

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
        Field update with expected-matching normalization:
        Pass 1: compute total propensity; Pass 2: scale and sample actions.
        """
        self.t += 1
        posts_last_step = [p for p in self.posts if p.time_step == self.t - 1]
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
            for p in posts_last_step
        ]
        self._last_step_real_volume = float(len(posts_last_step))

        normalized_volume = len(posts_last_step) / max(self.data_scale, 1.0)
        self._update_hawkes_state(normalized_volume)

        topic_heats: Dict[str, float] = {}
        if self.topic_manager and self._topics:
            for tp in self._topics:
                topic_heats[tp] = self.topic_manager.get_heat(tp, current_time=self.t)
        hawkes_intensity = sum(topic_heats.values()) if topic_heats else self.current_intensity
        self.current_intensity = hawkes_intensity

        sentiment_score = self._compute_sentiment_score(posts_last_step)
        self.field_generator.current_time = self.t
        traces = {"trend_rank": 0, "velocity": 0.0}
        env_fields = self.field_generator.compute_fields(hawkes_intensity, sentiment_score, traces)
        self.phase = self._determine_phase(hawkes_intensity)

        # Pass 1: gather propensities
        agent_propensities: Dict[str, float] = {}
        for name, agent in self.agents.items():
            prop = 0.0
            if hasattr(agent, "calculate_raw_propensity"):
                try:
                    prop = float(agent.calculate_raw_propensity(env_fields))
                except Exception:
                    prop = 0.0
            agent_propensities[name] = max(prop, 0.0)

        total_propensity = sum(agent_propensities.values())
        delta_t = 1.0
        target_count = max(hawkes_intensity * delta_t, 0.0)
        global_scale = (target_count / total_propensity) if total_propensity > 0 else 0.0
        env_fields.global_scale = global_scale

        # Pass 2: sample actions probabilistically
        actions: List[AgentAction] = []
        for name, agent in self.agents.items():
            if request_delay > 0:
                time.sleep(request_delay)
            try:
                decision = agent.decide_action_probabilistic(
                    self.t, env_fields, observed_posts=observed, environment=self
                )
            except Exception:
                decision = None
            if not decision:
                continue
            act_type = decision.get("action") or decision.get("action_type") or "silent"
            actions.append(
                AgentAction(
                    agent_id=name,
                    action_type=act_type,
                    content=decision.get("post_text", decision.get("content", "")) or "",
                    topic=decision.get("topic") or "",
                    timestamp=self.t,
                )
            )

        return actions
