"""
核心训练脚本：全局拟合事件的共同参数。
默认从 datasets 目录读取数据，可通过 --data_dir 指定其他数据目录。
"""

import argparse
import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict
from pathlib import Path
from scipy.optimize import minimize

from utils.data_loader import load_all_datasets
from utils.spread_model import HawkesPredictor
from config.settings import DEFAULT_REAL_DATA_PATH, DEFAULT_TOPICS
from agents.agent import ROLE_PARAM_DISTRIBUTIONS
from env.social_env import SocialEnv
from simulate import (
    build_agents,
    build_graph,
    inject_initial_rumor,
    BEST_HAWKES_PARAMS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train global Hawkes parameters")
    parser.add_argument(
        "--data_dir",
        default="dataset_peak350",
        help="数据目录，包含若干 CSV（至少 heat 列），默认 datasets",
    )
    parser.add_argument(
        "--random_test",
        action="store_true",
        help="随机抽取 10% 时间点作为 test，其余按时间顺序 80/10 切分",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子（仅在 --random_test 时生效）",
    )
    parser.add_argument(
        "--use_global_init",
        action="store_true",
        help="使用 CMA-ES + 多起点 L-BFGS-B 进行全局搜索与精调",
    )
    parser.add_argument(
        "--global_n_starts",
        type=int,
        default=30,
        help="全局搜索后用于 L-BFGS-B 的起点数量",
    )
    parser.add_argument(
        "--cma_maxiter",
        type=int,
        default=40,
        help="CMA-ES 最大迭代轮数",
    )
    parser.add_argument(
        "--cma_popsize",
        type=int,
        default=16,
        help="CMA-ES 种群大小",
    )
    parser.add_argument(
        "--cma_sigma0",
        type=float,
        default=0.3,
        help="CMA-ES 初始步长标量",
    )
    parser.add_argument(
        "--perturb_scale",
        type=float,
        default=0.15,
        help="围绕 CMA-ES 最优解生成高斯扰动的尺度（相对 bounds 范围）",
    )
    parser.add_argument(
        "--lbfgs_maxiter",
        type=int,
        default=300,
        help="L-BFGS-B 最大迭代次数",
    )
    return parser.parse_args()


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.maximum(y_true, 1e-6)
    return float(np.mean(np.abs(y_pred - safe_true) / safe_true) * 100.0)


def rollout(params: np.ndarray, series: np.ndarray) -> np.ndarray:
    """
    双衰减核：pred = H_base + mu_fast * M_fast + mu_slow * M_slow
    M_fast = M_fast * exp(-lambda_fast) + y_{t-1}
    M_slow = M_slow * exp(-lambda_slow) + y_{t-1}
    """
    return HawkesPredictor.predict_dual_decay(params, series)


def global_loss(params: np.ndarray, datasets) -> float:
    if (params <= 0).any():
        return 1e12
    mu_fast, mu_slow, h_base, lam_fast, lam_slow = params
    if lam_fast <= lam_slow:
        return 1e12  # 违反双衰减约束，直接惩罚

    total = 0.0
    for ev in datasets:
        train_val = np.concatenate([ev["train"], ev["val"]])
        pred = rollout(params, train_val)
        total += compute_mse(train_val, pred)
    return total / len(datasets)


def evaluate_split(params: np.ndarray, datasets, split: str):
    """
    split: "train_val" 或 "test"
    返回 (avg_mse, avg_mape, per_dataset_list)
    """
    mu_fast, mu_slow, h_base, lam_fast, lam_slow = params
    if lam_fast <= lam_slow or (params <= 0).any():
        raise ValueError("参数不满足正值或 lambda_fast > lambda_slow 的约束")
    per_ds = []
    total = 0.0
    total_mape = 0.0
    for ev in datasets:
        series = (
            np.concatenate([ev["train"], ev["val"]]) if split == "train_val" else ev["test"]
        )
        pred = rollout(params, series)
        mse = compute_mse(series, pred)
        mape = compute_mape(series, pred)
        per_ds.append((ev["name"], mse, mape))
        total += mse
        total_mape += mape
    n = len(datasets)
    return total / n, total_mape / n, per_ds


def _project_to_bounds(x: Iterable[float], bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    arr = np.clip(arr, lower, upper)
    if arr[3] <= arr[4]:
        arr[3] = min(upper[3], max(arr[4] + 1e-3, arr[3] + 1e-3))
    return arr


def fit_with_init_guesses(
    datasets,
    bounds: Sequence[Tuple[float, float]],
    init_guesses: Sequence[np.ndarray],
    lbfgs_maxiter: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    原有多起点列表 L-BFGS-B 拟合逻辑。
    """
    best_params = None
    best_loss = float("inf")
    options = {"maxiter": lbfgs_maxiter} if lbfgs_maxiter is not None else None

    for idx, x0 in enumerate(init_guesses, start=1):
        print(f"\n[Start {idx}] init={x0.tolist()}")

        def wrapped(x):
            loss = global_loss(x, datasets)
            print(f"  params={x} loss={loss:.6f}")
            return loss

        res = minimize(
            wrapped,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=options,
        )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = np.array(res.x, dtype=float)
            print(f"  -> New best loss {best_loss:.6f}")

    if best_params is None:
        raise RuntimeError("未找到可行解")
    return best_params, best_loss


def global_search_cmaes(
    loss_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    x0: Iterable[float],
    sigma0: float = 0.3,
    popsize: int = 16,
    maxiter: int = 40,
    seed: Optional[int] = None,
):
    """
    使用 CMA-ES 进行全局粗搜索，返回 cma 的 result 与按 loss 排序的候选解。
    依赖: pip install cma
    """
    try:
        import cma
    except ImportError as e:  # pragma: no cover - 运行时安装依赖
        raise ImportError("需要安装 cma 库：pip install cma") from e

    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    opts = {
        "bounds": [lower, upper],
        "popsize": popsize,
        "maxiter": maxiter,
        "seed": seed,
        "verb_disp": 1,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    solutions: List[Tuple[float, np.ndarray]] = []
    while not es.stop():
        xs = es.ask()
        losses = [loss_fn(np.array(x, dtype=float)) for x in xs]
        es.tell(xs, losses)
        es.disp()
        solutions.extend((float(l), np.array(x, dtype=float)) for l, x in zip(losses, xs))

    res = es.result  # type: ignore[attr-defined]
    solutions.sort(key=lambda t: t[0])
    top_k = max(1, min(20, len(solutions)))
    top_solutions = solutions[:top_k]
    print(f"CMA-ES finished. Best loss={top_solutions[0][0]:.6f}, collected {len(top_solutions)} good samples.")
    return res, top_solutions


def generate_initial_points_from_cmaes(
    xbest: Iterable[float],
    candidate_pool: Sequence[Tuple[float, np.ndarray]],
    bounds: Sequence[Tuple[float, float]],
    n_starts: int = 30,
    perturb_scale: float = 0.15,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    根据 CMA-ES 最优解 + 优秀样本 + 扰动生成多起点。
    """
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    points: List[np.ndarray] = []

    def add_point(x):
        candidate = _project_to_bounds(x, bounds)
        if not any(np.allclose(p, candidate, atol=1e-6, rtol=1e-5) for p in points):
            points.append(candidate)

    add_point(xbest)
    for _, x in candidate_pool:
        if len(points) >= n_starts:
            break
        add_point(x)

    remaining = max(0, n_starts - len(points))
    for _ in range(remaining):
        noise = rng.normal(scale=perturb_scale * span)
        add_point(np.asarray(xbest, dtype=float) + noise)
        if len(points) >= n_starts:
            break
    print(f"Generated {len(points)} initial points for multi-start L-BFGS-B.")
    return points


def run_multi_start_lbfgsb(
    init_points: Sequence[np.ndarray],
    loss_fn: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    maxiter: int = 300,
):
    """
    多起点 L-BFGS-B 优化，返回所有 scipy OptimizeResult。
    """
    results = []
    options = {"maxiter": maxiter} if maxiter else None
    total = len(init_points)
    for i, x0 in enumerate(init_points, start=1):
        start_desc = np.round(x0, 4).tolist()
        print(f"[L-BFGS-B {i}/{total}] start={start_desc}")
        res = minimize(
            loss_fn,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=options,
        )
        print(f"  -> loss={res.fun:.6f} success={res.success} message={res.message}")
        results.append(res)
    return results


def fit_hawkes_params_global(
    datasets,
    bounds: Sequence[Tuple[float, float]],
    cma_x0: Iterable[float],
    sigma0: float = 0.3,
    popsize: int = 16,
    cma_maxiter: int = 40,
    n_starts: int = 30,
    perturb_scale: float = 0.15,
    lbfgs_maxiter: int = 300,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, dict]:
    """
    CMA-ES 全局搜索 + 多起点 L-BFGS-B 精调。
    返回 (best_params, best_loss, debug_info)。
    """
    loss_fn = lambda x: global_loss(x, datasets)
    print("\n== CMA-ES global search ==")
    cma_res, top_solutions = global_search_cmaes(
        loss_fn,
        bounds=bounds,
        x0=cma_x0,
        sigma0=sigma0,
        popsize=popsize,
        maxiter=cma_maxiter,
        seed=seed,
    )
    xbest = np.array(cma_res.xbest, dtype=float)
    init_points = generate_initial_points_from_cmaes(
        xbest,
        candidate_pool=top_solutions,
        bounds=bounds,
        n_starts=n_starts,
        perturb_scale=perturb_scale,
        seed=seed,
    )
    print("\n== Multi-start L-BFGS-B refinement ==")
    lbfgs_results = run_multi_start_lbfgsb(
        init_points,
        loss_fn=loss_fn,
        bounds=bounds,
        maxiter=lbfgs_maxiter,
    )
    lbfgs_results.sort(key=lambda r: float(r.fun))
    best = lbfgs_results[0]
    debug_info = {
        "cma_result": cma_res,
        "cma_top_solutions": top_solutions,
        "lbfgs_results": lbfgs_results,
        "init_points": init_points,
    }
    return np.array(best.x, dtype=float), float(best.fun), debug_info


# ---------------------------------------------------------------------------
# Agent-parameter calibration via CMA-ES (expected matching to real curves)
# ---------------------------------------------------------------------------

_CALIBRATION_ROLES = ["Crowd", "KOL", "Troll", "BrandOfficial"]
_CALIBRATION_KEYS = ["base_activity", "sensitivity_reward", "sensitivity_risk", "conformity"]


class FakeLLM:
    """Cheap stub to bypass real LLM calls during calibration."""

    def chat(self, system: str, user: str, **kwargs):
        import json

        return json.dumps(
            {
                "action": "post",
                "post_text": "占位自动生成",
                "sentiment": "NEUTRAL",
                "target_post_id": None,
                "topic": None,
            }
        )

    def chat_thinking(self, *args, **kwargs):
        return self.chat("", "")


def _update_role_params_from_vector(vec: np.ndarray) -> None:
    """Map flattened means/stds into ROLE_PARAM_DISTRIBUTIONS (with clipping)."""
    num_keys = len(_CALIBRATION_KEYS)
    num_roles = len(_CALIBRATION_ROLES)
    expected = num_keys * num_roles * 2
    if vec.shape[0] < expected:
        raise ValueError(f"params_vector length {vec.shape[0]} < expected {expected}")
    means = vec[: num_keys * num_roles]
    stds = vec[num_keys * num_roles : expected]
    for i, role in enumerate(_CALIBRATION_ROLES):
        dist: Dict[str, float] = ROLE_PARAM_DISTRIBUTIONS.get(role, {})
        for j, key in enumerate(_CALIBRATION_KEYS):
            idx = i * num_keys + j
            dist[key] = float(np.clip(means[idx], 0.0, 2.0))
            dist[f"{key}_std"] = float(np.clip(abs(stds[idx]), 0.05, 2.0))
        ROLE_PARAM_DISTRIBUTIONS[role] = dist


def _initial_param_vector() -> np.ndarray:
    means: List[float] = []
    stds: List[float] = []
    for role in _CALIBRATION_ROLES:
        dist = ROLE_PARAM_DISTRIBUTIONS.get(role, {})
        for key in _CALIBRATION_KEYS:
            means.append(float(dist.get(key, 0.5)))
            stds.append(float(dist.get(f"{key}_std", 0.1)))
    return np.array(means + stds, dtype=float)


def _load_real_series(path: Path, topics: List[str], max_steps: int) -> Dict[str, List[float]]:
    if not path.exists():
        return {t: [] for t in topics}
    df = pd.read_csv(path)
    out: Dict[str, List[float]] = {t: [] for t in topics}
    for t in topics:
        sub = df[df["topic"] == t].sort_values("timestamp")
        heats = sub["heat"].astype(float).tolist()[:max_steps]
        out[t] = heats
    return out


def _run_simulation(topics: List[str], params_vector: np.ndarray, steps: int = 350, seed: int = 123):
    np.random.seed(seed)
    _update_role_params_from_vector(params_vector)

    llm = FakeLLM()
    agents = build_agents(llm, topics=topics)
    G = build_graph(agents.keys())
    env = SocialEnv(
        agents,
        G,
        topics=topics,
        hawkes_params=BEST_HAWKES_PARAMS,
        llm_client=llm,
        fixed_heat_scale=None,
        initial_topic_heats=None,
        real_heat_trajectory=None,
    )

    # Seed a rumor per topic as in simulate.py
    seed_volume = 0.0
    for tp in topics:
        inject_initial_rumor(env, topic=tp)
        seed_volume += 1.0
    env._last_step_real_volume = seed_volume
    env._update_hawkes_state(seed_volume / max(getattr(env, "data_scale", 1.0), 1.0))

    heat_history: List[Dict[str, float]] = []
    for _ in range(1, steps + 1):
        env.step(pr_strategy=None, request_delay=0.0)
        if env.topic_manager:
            snap = {"time": env.t}
            for topic in env.topic_manager.topics:
                snap[topic] = env.topic_manager.get_heat(topic)
            heat_history.append(snap)

    sim_series: Dict[str, List[float]] = {t: [] for t in topics}
    for snap in heat_history:
        for t in topics:
            sim_series[t].append(snap.get(t, 0.0))
    return sim_series


def simulation_loss(
    params_vector: np.ndarray,
    topics: Optional[List[str]] = None,
    real_series: Optional[Dict[str, List[float]]] = None,
    steps: int = 350,
) -> float:
    topics = topics or list(DEFAULT_TOPICS)
    real_series = real_series or _load_real_series(Path(DEFAULT_REAL_DATA_PATH), topics, steps)
    sim_series = _run_simulation(topics, params_vector, steps=steps)

    losses = []
    for t in topics:
        sim = np.array(sim_series.get(t, []), dtype=float)
        real = np.array(real_series.get(t, []), dtype=float)
        L = min(len(sim), len(real))
        if L == 0:
            losses.append(1e3)
            continue
        sim = sim[:L]
        real = real[:L]
        mape = compute_mape(real, sim)
        penalty = 0.5 * abs(np.std(sim) - np.std(real))
        losses.append(mape + penalty)
    return float(np.mean(losses))


def calibrate(
    real_data_path: Path = DEFAULT_REAL_DATA_PATH,
    steps: int = 350,
    seed: Optional[int] = None,
    maxiter: int = 50,
):
    """
    Calibrate ROLE_PARAM_DISTRIBUTIONS with CMA-ES to match real heat curves.
    """
    try:
        import cma
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please install cma: pip install cma") from e

    topics = list(DEFAULT_TOPICS)
    real_series = _load_real_series(Path(real_data_path), topics, steps)
    x0 = _initial_param_vector()
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"maxiter": maxiter, "seed": seed})
    while not es.stop():
        xs = es.ask()
        losses = [simulation_loss(np.array(x), topics=topics, real_series=real_series, steps=steps) for x in xs]
        es.tell(xs, losses)
        es.disp()

    res = es.result  # type: ignore[attr-defined]
    print("\n[CMA-ES] Best loss:", res.fbest)
    print("[CMA-ES] Best parameters:", res.xbest)
    _update_role_params_from_vector(np.array(res.xbest, dtype=float))
    return res


def main():
    args = parse_args()
    datasets = load_all_datasets(
        args.data_dir,
        random_test=args.random_test,
        seed=args.seed,
        normalize=True,
    )
    print(f"共加载 {len(datasets)} 个事件，用于全局优化")
    scales = [ev.get("scale", 1.0) for ev in datasets]
    if any(s != 1.0 for s in scales):
        print(f"已按事件最大绝对值归一化，scale 范围: min={min(scales):.4f}, max={max(scales):.4f}")

    bounds = [
        (1e-6, 500.0),   # mu_fast
        (1e-6, 500.0),   # mu_slow
        (1e-6, 100.0),      # H_base
        (0.5, 5.0),      # lambda_fast
        (0.01, 2.0),     # lambda_slow
    ]

    init_guesses = [
        np.array([5.0, 2.0, 5.0, 3.5, 0.3]),
        np.array([3.0, 1.0, 10.0, 4.0, 0.5]),
        np.array([2.0, 2.0, 20.0, 2.5, 0.4]),
        np.array([1.0, 3.0, 30.0, 3.0, 0.2]),
        np.array([0.8, 0.8, 50.0, 2.0, 0.1]),
    ]

    if args.use_global_init:
        cma_x0 = init_guesses[0]
        best_params, best_loss, _ = fit_hawkes_params_global(
            datasets,
            bounds=bounds,
            cma_x0=cma_x0,
            sigma0=args.cma_sigma0,
            popsize=args.cma_popsize,
            cma_maxiter=args.cma_maxiter,
            n_starts=args.global_n_starts,
            perturb_scale=args.perturb_scale,
            lbfgs_maxiter=args.lbfgs_maxiter,
            seed=args.seed,
        )
    else:
        best_params, best_loss = fit_with_init_guesses(
            datasets,
            bounds=bounds,
            init_guesses=init_guesses,
            lbfgs_maxiter=args.lbfgs_maxiter,
        )

    print("\nGlobal Optimal Parameters (mu_fast, mu_slow, H_base, lambda_fast, lambda_slow):")
    print(best_params.tolist())
    print(f"Train+Val avg MSE (优化目标): {best_loss:.6f}")

    train_mse, train_mape, _ = evaluate_split(best_params, datasets, split="train_val")
    print(f"Train+Val avg MAPE: {train_mape:.4f}%")

    test_mse, test_mape, per_ds = evaluate_split(best_params, datasets, split="test")
    print(f"Test avg MSE: {test_mse:.6f}")
    print(f"Test avg MAPE: {test_mape:.4f}%")
    print("Per-dataset test MSE / MAPE:")
    for name, l, mape in sorted(per_ds):
        print(f"  {name}: MSE={l:.6f}, MAPE={mape:.4f}%")


if __name__ == "__main__":
    main()
