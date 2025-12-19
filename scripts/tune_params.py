from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import DEFAULT_REAL_DATA_PATH, DEFAULT_SIM_SEED, DEFAULT_TOPICS
from simulate import _compute_metrics, _load_real_series, simulate_steps


@dataclass
class Params:
    base_weight: float = 0.75
    bg_intensity_ratio: float = 0.01
    volume_factor: float = 0.002


def _build_sim_series(heat_history: List[dict], topics: List[str]) -> Dict[str, List[float]]:
    sim_series = {t: [] for t in topics}
    for snap in heat_history:
        for t in topics:
            sim_series[t].append(float(snap.get(t, 0.0) or 0.0))
    return sim_series


def evaluate(
    params: Params,
    *,
    real_path: Path,
    topics: List[str],
    steps: int,
    seed: int,
) -> Tuple[float, dict]:
    real_series = _load_real_series(real_path, topics, max_steps=steps)
    initial_heats = {t: (real_series.get(t) or [0.0])[0] for t in topics}
    heat_scale = float(max((max(v) for v in real_series.values() if v), default=1.0))

    env, _, heat_history, _ = simulate_steps(
        T=steps,
        seed=seed,
        topics=topics,
        hawkes_params={"heat_scale": heat_scale},
        fixed_heat_scale=heat_scale,
        initial_topic_heats=initial_heats,
        real_heat_trajectory=real_series,
        population_scale=heat_scale,
        use_llm=False,  # 启发式 policy（可重复、便于调参）
        base_weight=float(params.base_weight),
        bg_intensity_ratio=float(params.bg_intensity_ratio),
        volume_factor=float(params.volume_factor),
    )
    sim_series = _build_sim_series(heat_history, topics)
    overall, per_topic = _compute_metrics(sim_series, real_series)
    loss = float(overall.get("avg_mape", 0.0))
    return loss, {"overall": overall, "per_topic": per_topic, "heat_scale": heat_scale}


def main():
    ap = argparse.ArgumentParser(description="Coordinate-descent style tuning for simulation parameters.")
    ap.add_argument("--real-data", type=Path, default=DEFAULT_REAL_DATA_PATH)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--seed", type=int, default=DEFAULT_SIM_SEED)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    topics = list(DEFAULT_TOPICS)

    # 搜索网格（可按需扩展）
    grid_base = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    grid_bg = [0.0, 0.001, 0.002, 0.005, 0.01]
    grid_vol = [0.0005, 0.001, 0.002, 0.005, 0.01]

    params = Params()
    best_loss, best_report = evaluate(params, real_path=args.real_data, topics=topics, steps=args.steps, seed=args.seed)
    print("init", asdict(params), "loss", round(best_loss, 3))

    for r in range(args.rounds):
        # 逐坐标更新（近似“梯度下降”：每轮只动一个维度）
        for name, grid in [("base_weight", grid_base), ("bg_intensity_ratio", grid_bg), ("volume_factor", grid_vol)]:
            current = getattr(params, name)
            candidates = grid
            if name != "base_weight":
                # 让搜索更贴近当前值（以乘法邻域为主）
                neigh = sorted(set([current] + [current * f for f in (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)] + candidates))
                candidates = [c for c in neigh if c >= 0.0]
            best_local = (best_loss, current)
            for c in candidates:
                trial = Params(**asdict(params))
                setattr(trial, name, float(c))
                loss, _ = evaluate(trial, real_path=args.real_data, topics=topics, steps=args.steps, seed=args.seed)
                if loss < best_local[0]:
                    best_local = (loss, float(c))
            setattr(params, name, best_local[1])
            best_loss = best_local[0]
            print(f"round={r+1} update {name}={getattr(params,name):.6g} loss={best_loss:.3f}")

        best_loss, best_report = evaluate(params, real_path=args.real_data, topics=topics, steps=args.steps, seed=args.seed)
        print("round", r + 1, "params", asdict(params), "loss", round(best_loss, 3))

    print("best", asdict(params), "loss", round(best_loss, 3))
    # 输出 top-10 最差话题，便于后续定向优化
    per_topic = best_report.get("per_topic", {})
    worst = sorted(per_topic.items(), key=lambda kv: float(kv[1].get("mape", 0.0)), reverse=True)[:10]
    print("worst_topics_top10:")
    for t, m in worst:
        print(f"  {t}: mape={float(m.get('mape',0.0)):.2f} mse={float(m.get('mse',0.0)):.4f} len={int(m.get('len',0))}")


if __name__ == "__main__":
    main()
