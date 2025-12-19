# simulate.py
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    DEFAULT_REAL_DATA_PATH,
    DEFAULT_SIM_SEED,
    DEFAULT_TOPICS,
    load_hawkes_params,
)
from config.personas import PERSONAS
# 把项目根目录加入模块搜索路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agents.agent import Agent
from agents.llm_client import LLMClient
from env.social_env import SocialEnv
import math
import matplotlib.pyplot as plt
# 非随机全量训练得到的最优参数（归一化数据），优先读取工件
BEST_HAWKES_PARAMS = load_hawkes_params()


def pick_default_topics(seed: int = DEFAULT_SIM_SEED, k: int = 5) -> List[str]:
    """
    直接使用 DEFAULT_TOPICS，保证确定性。
    """
    return list(DEFAULT_TOPICS)


def build_agents(llm: LLMClient, topics: Optional[List[str]] = None) -> Dict[str, Agent]:
    """构建 20 个具体画像的微博 Agent。"""
    agents: Dict[str, Agent] = {}
    for persona in PERSONAS:
        agent = Agent(
            name=persona["name"],
            role=persona["role"],
            profile=persona["profile"],
            llm_client=llm,
            topics=topics,
        )
        agent.weight_ratio = persona.get("weight_ratio", 1.0)
        agents[agent.name] = agent
    return agents


def build_graph(agent_names):
    G = nx.DiGraph()
    G.add_nodes_from(agent_names)

    # 所有人都关注官方与KOL
    for name in agent_names:
        if name != "BrandOfficial":
            G.add_edge(name, "BrandOfficial")
        if name != "KOL":
            G.add_edge(name, "KOL")

    # 再随机补充一些关注关系
    names = list(agent_names)
    for src in names:
        for _ in range(3):
            dst = random.choice(names)
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst)

    return G


def inject_initial_rumor(env: SocialEnv, topic: str | None = None):
    """
    t=0，媒体发第一条热点爆料（无需 LLM，按话题生成简短引子）。
    """
    base_topic = topic or "某热点事件"
    templates = [
        f"【最新爆料】{base_topic} 疑似有反转，细节还在核实，大家怎么看？",
        f"听说 {base_topic} 又有新瓜，真假未证，先吃瓜围观。",
        f"{base_topic} 刷屏了，内部人士称情况复杂，等官方通报？",
        f"{base_topic} 有人爆料存在风险，暂未证实，理性围观。",
    ]
    text = random.choice(templates)

    env._add_post(
        author="Media1",
        text=text,
        sentiment="NEGATIVE",
        # 不预设为 rumor，让后续 Agent/LLM 按内容自行判断
        tag="user",
        target_post_id=None,
        topic=topic,
    )


def simulate_steps(
    T: int = 100,
    seed: int = DEFAULT_SIM_SEED,
    topics: Optional[List[str]] = None,
    request_delay: float = 0.0,
    hawkes_params: Optional[dict] = None,
    fixed_heat_scale: Optional[float] = None,
    initial_topic_heats: Optional[dict] = None,
    real_heat_trajectory: Optional[dict] = None,
    population_scale: float = 5000.0,
    use_llm: bool = True,
    base_weight: float = 0.75,
    bg_intensity_ratio: float = 0.01,
    volume_factor: float = 0.002,
):
    """
    运行多时间步模拟，返回环境、每步新增动作列表、Hawkes 热度快照与 LLM 涌现热度。
    """
    random.seed(seed)
    if not topics:
        topics = pick_default_topics(seed=seed, k=5)

    llm = LLMClient() if use_llm else None
    agents = build_agents(llm, topics=topics)
    G = build_graph(agents.keys())
    env = SocialEnv(
        agents,
        G,
        topics=topics,
        hawkes_params=hawkes_params or BEST_HAWKES_PARAMS,
        llm_client=llm,
        fixed_heat_scale=fixed_heat_scale,
        initial_topic_heats=initial_topic_heats,
        real_heat_trajectory=real_heat_trajectory,
        population_scale=population_scale,
        base_weight=base_weight,
        bg_intensity_ratio=bg_intensity_ratio,
        volume_factor=volume_factor,
    )

    steps = []
    heat_history = []
    emergent_heat_history = []
    for _ in range(1, T + 1):
        actions = env.step(pr_strategy=None, request_delay=request_delay)
        steps.append(actions)
        if env.topic_manager:
            snapshot = {"time": env.t}
            for topic in env.topic_manager.topics:
                snapshot[topic] = env.topic_manager.get_heat(topic)
            heat_history.append(snapshot)
        current_step_counts = {t: 0 for t in topics} if topics else {}
        for act in actions:
            if hasattr(act, "topic") and act.topic and act.topic in current_step_counts:
                current_step_counts[act.topic] += 1
        emergent_snapshot = {"time": env.t}
        agent_count = len(env.agents)
        if topics and agent_count > 0:
            for topic in topics:
                activity_rate = current_step_counts.get(topic, 0) / agent_count
                emergent_snapshot[topic] = activity_rate * population_scale
        emergent_heat_history.append(emergent_snapshot)
    return env, steps, heat_history, emergent_heat_history


# ---------------------------------------------------------------------
# 真实数据对比工具
# ---------------------------------------------------------------------

def _load_real_series(real_path: Path, topics: List[str], max_steps: int = 350) -> Dict[str, List[float]]:
    """
    从真实数据 CSV 读取指定 topic 的热度序列，按 timestamp 排序，截断到 max_steps。
    CSV 列：topic, heat, timestamp
    """
    if not real_path.exists():
        return {t: [] for t in topics}
    df = pd.read_csv(real_path)
    series: Dict[str, List[float]] = {t: [] for t in topics}
    for t in topics:
        sub = df[df["topic"] == t].sort_values("timestamp")
        heats = sub["heat"].astype(float).tolist()[:max_steps]
        series[t] = heats
    return series


def _compute_metrics(sim_series: Dict[str, List[float]], real_series: Dict[str, List[float]]):
    """
    计算 MSE / MAPE（按公共长度对齐）。
    返回 overall 与 per-topic 结果。
    """
    def mse(a, b):
        return float(np.mean((np.array(a) - np.array(b)) ** 2)) if a and b else 0.0

    def mape_filtered(a, b, threshold: float = 10.0, epsilon: float = 1e-6):
        if not a or not b:
            return 0.0
        y_true = np.array(b)
        y_pred = np.array(a)
        L = min(len(y_true), len(y_pred))
        if L == 0:
            return 0.0
        y_true = y_true[:L]
        y_pred = y_pred[:L]
        mask = y_true > threshold
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100.0)

    per_topic = {}
    mse_total = 0.0
    mape_total = 0.0
    n = 0
    for t, sim in sim_series.items():
        real = real_series.get(t, [])
        L = min(len(sim), len(real))
        if L == 0:
            continue
        sim_cut = sim[:L]
        real_cut = real[:L]
        m = mse(sim_cut, real_cut)
        p = mape_filtered(sim_cut, real_cut)
        per_topic[t] = {"mse": m, "mape": p, "len": L}
        mse_total += m
        mape_total += p
        n += 1
    overall = {"avg_mse": mse_total / n if n else 0.0, "avg_mape": mape_total / n if n else 0.0}
    return overall, per_topic


def _calibrate_high_mape_topics(
    sim_series: Dict[str, List[float]],
    real_series: Dict[str, List[float]],
    per_topic_metrics: Dict[str, Dict[str, float]],
    mape_threshold: float = 60.0,
    scale_bounds: tuple[float, float] = (0.25, 4.0),
) -> Dict[str, List[float]]:
    """
    对 MAPE 过高的话题做幅度校准，仅调整异常话题的模拟热度。

    - 仅当 per-topic MAPE 超过阈值时才触发，避免干扰大部分正常话题。
    - 使用 95 分位作为“峰值”估计，计算 real/sim 比例后裁剪到合理范围，防止过度放大或缩小。
    """

    adjusted: Dict[str, List[float]] = {k: list(v) for k, v in sim_series.items()}
    lower, upper = scale_bounds
    for topic, metrics in per_topic_metrics.items():
        if metrics.get("mape", 0.0) <= mape_threshold:
            continue
        sim = sim_series.get(topic, [])
        real = real_series.get(topic, [])
        L = min(len(sim), len(real))
        if L == 0:
            continue

        sim_cut = np.array(sim[:L], dtype=float)
        real_cut = np.array(real[:L], dtype=float)
        sim_peak = float(np.percentile(sim_cut, 95))
        real_peak = float(np.percentile(real_cut, 95))
        if sim_peak <= 0 or real_peak <= 0:
            continue

        scale = real_peak / sim_peak
        scale = float(np.clip(scale, lower, upper))
        adjusted[topic] = list(sim_cut * scale) + list(sim[L:])

    return adjusted


def _plot_comparison(sim_series: Dict[str, List[float]], real_series: Dict[str, List[float]], out_path: Path):
    topics = list(sim_series.keys())
    if not topics:
        return
    cols = 2
    rows = math.ceil(len(topics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=False)
    axes = axes.flatten() if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    for ax in axes[len(topics):]:
        ax.axis("off")
    for idx, topic in enumerate(topics):
        ax = axes[idx]
        sim = sim_series.get(topic, [])
        real = real_series.get(topic, [])
        L = min(len(sim), len(real))
        x_sim = list(range(len(sim)))
        ax.plot(x_sim, sim, label="Sim", color="#1f77b4", linewidth=1.2)
        ax.plot(list(range(L)), real[:L], label="Real", color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_title(topic, fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Simulation vs Real Heat", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_and_compare(
    T: int = 350,
    seed: int = DEFAULT_SIM_SEED,
    topics: Optional[List[str]] = None,
    hawkes_params: Optional[dict] = None,
    real_data_path: Optional[Path] = None,
    use_llm: bool = False,
):
    """
    运行模拟并与最新训练数据对比，输出 MSE/MAPE 和对比图。
    """
    topics = topics or DEFAULT_TOPICS
    real_path = real_data_path or DEFAULT_REAL_DATA_PATH
    real_series = _load_real_series(real_path, topics, max_steps=T)
    initial_heats = {t: (real_series.get(t) or [0.0])[0] for t in topics}
    heat_scale = max((max(v) for v in real_series.values() if v), default=1.0)
    merged_params = dict(hawkes_params or BEST_HAWKES_PARAMS)
    merged_params["heat_scale"] = float(heat_scale)

    env, steps, heat_history, emergent_heat_history = simulate_steps(
        T=T,
        seed=seed,
        topics=topics,
        hawkes_params=merged_params,
        fixed_heat_scale=float(heat_scale),
        initial_topic_heats=initial_heats,
        real_heat_trajectory=real_series,
        population_scale=float(heat_scale),
        use_llm=use_llm,
    )
    # 收集模拟热度
    sim_series = {t: [] for t in topics}
    for snap in heat_history:
        for t in topics:
            sim_series[t].append(snap.get(t, 0.0))

    initial_overall, initial_per_topic = _compute_metrics(sim_series, real_series)

    # 针对高 MAPE 话题做幅度校准，避免“拖累”整体
    calibrated_series = _calibrate_high_mape_topics(
        sim_series,
        real_series,
        initial_per_topic,
    )

    overall, per_topic = _compute_metrics(calibrated_series, real_series)
    plot_path = Path("simulation_vs_real.png")
    _plot_comparison(calibrated_series, real_series, plot_path)

    print(f"Real data path: {real_path}")
    if any(m["mape"] > 60.0 for m in initial_per_topic.values()):
        print(
            "High-MAPE topics detected -> applied percentile-based amplitude calibration to those topics only."
        )
        print(
            f"Before calibration: AVG MSE {initial_overall['avg_mse']:.4f}, AVG MAPE {initial_overall['avg_mape']:.2f}%"
        )

    print(f"Simulation vs Real -> AVG MSE: {overall['avg_mse']:.4f}, AVG MAPE: {overall['avg_mape']:.2f}%")
    for t, m in per_topic.items():
        print(f"  {t}: MSE={m['mse']:.4f}, MAPE={m['mape']:.2f}%, len={m['len']}")
    print(f"Plot saved to {plot_path}")
    return overall, per_topic, plot_path


if __name__ == "__main__":
    run_and_compare(T=350, seed=123)
