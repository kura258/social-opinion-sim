import numpy as np
import cma
import pandas as pd

from simulate import simulate_steps
from config.settings import DEFAULT_REAL_DATA_PATH


def simulation_loss(params):
    # params[0]: population_scale (潜在人口规模)
    scale = params[0]

    # 运行模拟（无延迟，加速计算）
    _, _, _, emergent_heat = simulate_steps(T=100, population_scale=scale, request_delay=0)

    # 提取第一条话题的热度曲线
    sim_curve = []
    topics = list(emergent_heat[0].keys())
    topics.remove("time")
    target_topic = topics[0]
    for item in emergent_heat:
        sim_curve.append(item[target_topic])

    # TODO: 使用真实数据计算 MAPE，这里用峰值对齐示例
    peak = max(sim_curve) if sim_curve else 0.0
    loss = abs(peak - 3000)
    return loss


def calibrate():
    print("开始 CMA-ES 校准...")
    x0 = [5000.0]  # 初始人口规模
    sigma0 = 1000.0
    es = cma.CMAEvolutionStrategy(x0, sigma0)
    es.optimize(simulation_loss, iterations=10)
    print(f"校准完成。最佳人口规模: {es.result.xbest[0]}")


if __name__ == "__main__":
    calibrate()
