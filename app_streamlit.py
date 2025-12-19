import time
import os
from typing import List, Optional, Dict
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config.settings import DEFAULT_SIM_SEED
from config.personas import PERSONAS
from agents.batch_processor import GROUP_MAPPING
from simulate import simulate_steps, pick_default_topics, DEFAULT_REAL_DATA_PATH
from env.social_env import SocialEnv


def collect_heat_history_df(heat_history: List[dict]) -> pd.DataFrame:
    if not heat_history:
        return pd.DataFrame(columns=["time"])
    return pd.DataFrame(heat_history)


def collect_posts_df(env: SocialEnv) -> pd.DataFrame:
    rows = []
    for p in env.posts:
        rows.append({
            "time": p.time_step,
            "author": p.author,
            "sentiment": p.sentiment,
            "topic": p.topic or "未标注",
            "tag": p.tag,
            "text": p.text,
        })
    return pd.DataFrame(rows)


def collect_agent_timeline(steps: List[List], agents) -> pd.DataFrame:
    """
    汇总每个 Agent 在每个时间步的行为（post/retweet/silent），并附带情绪与话题。
    """
    rows = []
    agent_names = list(agents.keys())
    for t_idx, posts in enumerate(steps, start=1):
        for name in agent_names:
            agent_posts = [p for p in posts if getattr(p, "agent_id", None) == name]
            if agent_posts:
                for p in agent_posts:
                    rows.append({
                        "time": t_idx,
                        "agent": name,
                        "action": getattr(p, "action_type", "unknown"),
                        "sentiment": "N/A",
                        "topic": getattr(p, "topic", "未标注"),
                        "text": getattr(p, "content", ""),
                    })
            else:
                rows.append({
                    "time": t_idx,
                    "agent": name,
                    "action": "silent",
                    "sentiment": "N/A",
                    "topic": "无",
                    "text": "",
                })
    return pd.DataFrame(rows)


def compute_pred_df(heat_history: List[dict]) -> pd.DataFrame:
    if not heat_history:
        return pd.DataFrame(columns=["time", "topic", "heat_pred"])
    wide = pd.DataFrame(heat_history)
    long = wide.melt(id_vars=["time"], var_name="topic", value_name="heat_pred")
    return long


def normalize_real_df(raw_df: pd.DataFrame, topics: List[str]) -> Optional[pd.DataFrame]:
    """
    将真实数据规范化为列：time, topic, heat_real
    - 若有 time 列直接使用
    - 若有 timestamp 列则按时间排序并为每个 topic 生成递增步数
    """
    df = raw_df.copy()
    if {"topic", "heat", "time"} <= set(df.columns):
        df = df.rename(columns={"heat": "heat_real"})
    elif {"topic", "heat", "timestamp"} <= set(df.columns):
        df = df.rename(columns={"heat": "heat_real"})
        df = df.sort_values(["topic", "timestamp"])
        df["time"] = df.groupby("topic").cumcount() + 1
    else:
        return None
    if topics:
        df = df[df["topic"].isin(topics)]
    return df[["time", "topic", "heat_real"]]


def build_topic_feed(steps: List[List], env: SocialEnv, heat_history: List[dict]):
    """
    构造话题热榜及互动流水：按 topic 归类每步的动作，并给出排序分值。
    """
    feed: Dict[str, Dict[str, List[dict]]] = {}
    post_lookup = {p.id: p for p in env.posts}
    # t=0 种子贴（系统注入）：展示为 seed 动作
    for p in env.posts:
        if getattr(p, "time_step", None) == 0 and getattr(p, "tag", "") == "official" and getattr(p, "topic", None):
            tp = p.topic
            entry = {
                "time": 0,
                "agent": getattr(p, "author", "System_Seed"),
                "action": "seed",
                "content": getattr(p, "text", ""),
                "target": "",
                "count": None,
            }
            feed.setdefault(tp, {}).setdefault("seed", []).append(entry)
    for t_idx, acts in enumerate(steps, start=1):
        for act in acts:
            topic = getattr(act, "topic", None)
            if not topic:
                continue
            entry = {
                "time": t_idx,
                "agent": getattr(act, "agent_id", ""),
                "action": getattr(act, "action_type", ""),
                "content": getattr(act, "content", ""),
                "target": "",
                "count": getattr(act, "count", None),
            }
            target_post_id = getattr(act, "target_post_id", None)
            if target_post_id and target_post_id in post_lookup:
                tp = post_lookup[target_post_id]
                entry["target"] = f"@{tp.author}: {tp.text[:40]}..."
            feed.setdefault(topic, {}).setdefault(entry["action"], []).append(entry)

    if heat_history:
        last = dict(heat_history[-1])
        last.pop("time", None)
        ranking = sorted(last.items(), key=lambda kv: kv[1], reverse=True)
    else:
        ranking = sorted(((tp, sum(len(v) for v in acts.values())) for tp, acts in feed.items()), key=lambda kv: kv[1], reverse=True)

    ordered_topics = [tp for tp, _ in ranking]
    return feed, ranking, ordered_topics


def compute_metrics(pred_df: pd.DataFrame, real_df: pd.DataFrame, topic_max: pd.Series | None = None):
    merged = pred_df.merge(real_df, on=["time", "topic"], how="inner")
    if merged.empty:
        return None, None, None, None
    epsilon = 1e-6
    # 如果做了按 topic 峰值归一化，MAPE 的分母会变得很小（长尾/早期容易“误差爆炸”）
    # 因此采用“过滤版 MAPE”：仅在 heat_real 大于阈值时计入
    mape_threshold = 0.01 if topic_max is not None else 10.0
    if topic_max is not None:
        merged = merged.join(topic_max.rename("heat_max"), on="topic")
        merged["heat_max"] = merged["heat_max"].replace(0, epsilon)
        merged["heat_real"] = merged["heat_real"] / merged["heat_max"]
        merged["heat_pred"] = merged["heat_pred"] / merged["heat_max"]

    merged["mse"] = (merged["heat_pred"] - merged["heat_real"]) ** 2
    abs_err = (merged["heat_pred"] - merged["heat_real"]).abs()
    denom = merged["heat_real"].abs() + epsilon
    merged["ape"] = abs_err / denom
    merged["mask"] = merged["heat_real"].abs() > float(mape_threshold)
    overall_mse = float(merged["mse"].mean())
    if merged["mask"].any():
        overall_mape = float(merged.loc[merged["mask"], "ape"].mean() * 100)
        overall_wmape = float(abs_err[merged["mask"]].sum() / (merged.loc[merged["mask"], "heat_real"].abs().sum() + epsilon) * 100)
    else:
        overall_mape = float(merged["ape"].mean() * 100)
        overall_wmape = float(abs_err.sum() / (merged["heat_real"].abs().sum() + epsilon) * 100)

    def _mape_filtered(group: pd.DataFrame) -> float:
        m = group["mask"]
        if m.any():
            return float(group.loc[m, "ape"].mean() * 100)
        return float(group["ape"].mean() * 100)

    def _wmape(group: pd.DataFrame) -> float:
        m = group["mask"]
        if m.any():
            return float(group.loc[m, "ape"].mul(group.loc[m, "heat_real"].abs()).sum() / (group.loc[m, "heat_real"].abs().sum() + epsilon) * 100)
        return float(group["ape"].mul(group["heat_real"].abs()).sum() / (group["heat_real"].abs().sum() + epsilon) * 100)

    per_topic = (
        merged.groupby("topic")
        .apply(lambda g: pd.Series({"mape": _mape_filtered(g), "wmape": _wmape(g), "mse": float(g["mse"].mean())}))
        .reset_index()
    )
    return overall_mse, overall_mape, overall_wmape, per_topic


def plot_metrics(per_topic: pd.DataFrame):
    if per_topic is None or per_topic.empty:
        return None
    per_topic = per_topic.sort_values("mape", ascending=False)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(per_topic["topic"], per_topic["mape"], color="#4C72B0")
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title("Per-topic MAPE")
    axes[0].invert_yaxis()

    axes[1].barh(per_topic["topic"], per_topic["mse"], color="#55A868")
    axes[1].set_xlabel("MSE")
    axes[1].set_title("Per-topic MSE")
    axes[1].invert_yaxis()

    plt.tight_layout()
    return fig


def estimate_llm_calls_per_step() -> int:
    """
    粗略估算：每个时间步会对每个 group 发起 1 次 LLM 请求。
    这里用 PERSONAS + GROUP_MAPPING 估算 group 数量。
    """
    groups = set()
    for persona in PERSONAS:
        name = persona.get("name", "")
        role = persona.get("role", "")
        gid = GROUP_MAPPING.get(name, GROUP_MAPPING.get(role, "Default"))
        groups.add(gid)
    return max(1, len(groups))


def main():
    st.set_page_config(page_title="话题热度模拟", layout="wide")
    st.title("多智能体舆论模拟：话题与热度可视化")
    # 顶部 Loss 流式展示

    # 控制面板
    st.sidebar.header("模拟参数")
    T = st.sidebar.slider("模拟时间步数", min_value=10, max_value=400, value=350, step=5)
    base_seed = st.sidebar.number_input("随机种子", min_value=0, max_value=9999, value=DEFAULT_SIM_SEED)
    delay_sec = st.sidebar.slider("每步界面延迟（秒）", 0.0, 2.0, 0.2, 0.05)
    request_delay = st.sidebar.slider("API 请求间隔（秒）", 0.0, 2.0, 0.2, 0.05)

    st.sidebar.subheader("LLM 费用保护")
    use_llm = st.sidebar.checkbox("启用 LLM（会产生费用）", value=False)
    llm_confirm_ok = True
    llm_budget_ok = True
    if use_llm:
        per_step_calls = estimate_llm_calls_per_step()
        est_calls = int(per_step_calls * int(T))
        st.sidebar.warning("启用 LLM 会产生费用；关闭浏览器不会停止后台 Streamlit 进程。")
        st.sidebar.caption(f"预计请求次数（粗略）：每步约 {per_step_calls} 次，总计约 {est_calls} 次。")
        max_requests = st.sidebar.number_input(
            "单次运行最大请求次数上限（建议设置）",
            min_value=0,
            max_value=200000,
            value=2000,
            step=100,
        )
        confirm = st.sidebar.text_input("二次确认：输入 CONFIRM 才允许运行", value="", type="password")
        llm_confirm_ok = confirm.strip().upper() == "CONFIRM"
        llm_budget_ok = (int(max_requests) <= 0) or (est_calls <= int(max_requests))

        if not os.getenv("CLOSEAI_API_KEY"):
            st.sidebar.error("未检测到 CLOSEAI_API_KEY：无法启用 LLM。")
            llm_confirm_ok = False

        if not llm_budget_ok:
            st.sidebar.error("预计请求次数超过上限，请降低 T 或提高上限。")
        if not llm_confirm_ok:
            st.sidebar.info("未确认：输入 CONFIRM 后才能开始模拟。")

    st.sidebar.subheader("动力学参数（拟合用）")
    base_weight = st.sidebar.slider("真实曲线引导权重 (base_weight)", 0.0, 1.0, 0.75, 0.05)
    bg_intensity_ratio = st.sidebar.slider("背景流量比例 (bg_intensity_ratio)", 0.0, 0.05, 0.01, 0.001)
    volume_factor = st.sidebar.slider("Agent 声量系数 (volume_factor)", 0.0, 0.02, 0.002, 0.0005)

    # 默认话题：使用数据集的全部 35 个，可在前端修改
    default_topics = pick_default_topics(seed=base_seed)
    if "topics_input_user_set" not in st.session_state:
        st.session_state["topics_input_user_set"] = False
    if not st.session_state["topics_input_user_set"]:
        st.session_state["topics_input"] = "\n".join(default_topics)

    st.sidebar.markdown(f"**默认话题数量：{len(default_topics)} 个**")
    topics_input = st.sidebar.text_area(
        "自定义话题（逗号或换行分隔）",
        st.session_state["topics_input"],
        height=80,
        placeholder="示例：\n数据安全\n全运会夺冠\n明星结婚",
    )
    if topics_input != st.session_state.get("topics_input", ""):
        st.session_state["topics_input"] = topics_input
        st.session_state["topics_input_user_set"] = True
    raw_topics = topics_input.replace("\n", ",")
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    real_file = st.sidebar.file_uploader("上传真实热度 CSV (列: time, topic, heat)", type=["csv"])
    use_default_real = st.sidebar.checkbox("使用默认真实数据（最新训练集）", value=True)
    real_heat_scale = 1.0
    initial_heats: Dict[str, float] = {}
    real_heat_trajectory: Dict[str, List[float]] = {}

    if st.button("开始模拟"):
        if use_llm and (not llm_confirm_ok or not llm_budget_ok):
            st.error("LLM 费用保护未通过：请完成二次确认并确保请求上限满足预估。")
            return
        st.info("正在创建环境并运行，请稍候...")
        # 读取真实数据并推断 scale/初始热度/轨迹
        real_df = None
        if real_file:
            try:
                real_df = pd.read_csv(real_file)
            except Exception as e:
                st.error(f"真实数据读取失败: {e}")
                real_df = None
        elif use_default_real and DEFAULT_REAL_DATA_PATH.exists():
            real_df = pd.read_csv(DEFAULT_REAL_DATA_PATH)

        if real_df is not None and "heat" in real_df.columns:
            cols = set(real_df.columns)
            # 针对选择的话题计算量级，不存在的直接忽略
            df_candidates = real_df
            if "topic" in cols and topics:
                df_candidates = real_df[real_df["topic"].isin(topics)]
                if df_candidates.empty:
                    df_candidates = real_df  # 兜底
            real_heat_scale = float(df_candidates["heat"].max())
            st.success(f"检测到真实数据量级 (Scale): {real_heat_scale:,.0f}")
            if {"topic", "timestamp", "heat"} <= cols:
                df_sorted = real_df.sort_values("timestamp")
                # 仅遍历存在于真实数据的所选话题
                topic_list = topics if topics else df_sorted["topic"].unique()
                for tp in topic_list:
                    sub_df = df_sorted[df_sorted["topic"] == tp]
                    if sub_df.empty:
                        continue
                    traj = sub_df["heat"].astype(float).tolist()
                    if traj:
                        real_heat_trajectory[tp] = traj
                        initial_heats[tp] = traj[0]
        else:
            st.warning("未检测到真实数据或缺少 heat 列，使用默认 Scale=1.0")

        # ==================== 量级对齐 ====================
        if real_df is not None and "heat" in real_df.columns:
            df_candidates = real_df[real_df["topic"].isin(topics)] if topics else real_df
            if df_candidates.empty:
                df_candidates = real_df
            real_heat_scale = float(df_candidates["heat"].max())
        else:
            real_heat_scale = 100.0

        env, steps, heat_history, emergent_heat_history = simulate_steps(
            T=T,
            seed=base_seed,
            topics=topics,
            request_delay=request_delay,
            hawkes_params={"heat_scale": real_heat_scale},
            fixed_heat_scale=real_heat_scale,
            initial_topic_heats=initial_heats,
            real_heat_trajectory=real_heat_trajectory,
            population_scale=real_heat_scale,
            base_weight=float(base_weight),
            bg_intensity_ratio=float(bg_intensity_ratio),
            volume_factor=float(volume_factor),
            use_llm=bool(use_llm),
        )
        st.success("模拟完成")

        topic_feed, topic_ranking, ordered_topics = build_topic_feed(steps, env, heat_history)
        st.subheader("话题热榜 & 互动流")
        if topic_ranking:
            for idx, (tp, score) in enumerate(topic_ranking, start=1):
                score_val = float(score) if isinstance(score, (int, float)) else score
                st.markdown(f"**{idx}. {tp}** · 热度/计数：{score_val:.2f}" if isinstance(score_val, float) else f"**{idx}. {tp}** · 热度/计数：{score_val}")
                actions_by_type = topic_feed.get(tp, {})
                summary_counts = {k: len(v) for k, v in actions_by_type.items()}
                bg_total = sum(float(i.get("count") or 0.0) for i in actions_by_type.get("background", []))
                st.markdown(
                    "点赞: {like}｜评论: {comment}｜转发: {retweet}｜发帖: {post}｜种子: {seed}｜背景: {bg}".format(
                        like=summary_counts.get("like", 0),
                        comment=summary_counts.get("comment", 0),
                        retweet=summary_counts.get("retweet", 0),
                        post=summary_counts.get("post", 0),
                        seed=summary_counts.get("seed", 0),
                        bg=int(bg_total),
                    )
                )

                for label, key in [
                    ("发帖", "post"),
                    ("转发", "retweet"),
                    ("评论", "comment"),
                    ("点赞", "like"),
                    ("种子", "seed"),
                    ("背景", "background"),
                ]:
                    items = actions_by_type.get(key, [])
                    if not items:
                        continue
                    with st.expander(f"{label}（{len(items)}）", expanded=False):
                        # 避免背景/长序列一次渲染过多
                        show_items = items
                        max_show = 30 if key in ("background",) else 80
                        if len(show_items) > max_show:
                            show_items = show_items[-max_show:]
                            st.caption(f"仅显示最近 {max_show} 条（共 {len(items)} 条）")
                        for item in show_items:
                            target_txt = f" ↪ {item['target']}" if item.get("target") else ""
                            content_txt = item.get("content", "")
                            extra = ""
                            if key == "background" and item.get("count") is not None:
                                extra = f"（+{int(float(item['count']))}）"
                            st.markdown(
                                f"- t={item['time']:>3}｜{item['agent']} {item['action']}{extra}：{content_txt}{target_txt}"
                            )
        else:
            st.info("暂无互动记录。")

        # 话题热度折线图
        if heat_history:
            heat_df = collect_heat_history_df(heat_history).set_index("time")
            st.subheader("话题热度随时间变化")
            st.line_chart(heat_df)
        else:
            st.info("当前未配置话题或无热度数据。")

        # 简化展示：仅总览表 + Agent 时间线
        st.subheader("全部事件汇总")
        posts_df = collect_posts_df(env)
        st.dataframe(posts_df)

        # Agent 行为时间线
        st.subheader("Agent 行为时间线")
        agent_timeline = collect_agent_timeline(steps, env.agents)
        st.dataframe(agent_timeline)

        # 真实数据对比（默认加载最新训练集，也可上传覆盖）
        real_df = None
        if real_file:
            try:
                real_df = pd.read_csv(real_file)
            except Exception as e:
                st.error(f"真实数据读取失败: {e}")
                real_df = None
        elif use_default_real and DEFAULT_REAL_DATA_PATH.exists():
            real_df = pd.read_csv(DEFAULT_REAL_DATA_PATH)

        if real_df is not None:
            norm_real = normalize_real_df(real_df, topics)
            if norm_real is None or norm_real.empty:
                st.warning("真实数据需包含列: time/topic/heat 或 timestamp/topic/heat。")
            else:
                topic_max = norm_real.groupby("topic")["heat_real"].max()
                pred_df = compute_pred_df(heat_history)
                overall_mse, overall_mape, overall_wmape, per_topic = compute_metrics(pred_df, norm_real, topic_max=topic_max)
                st.subheader("真实数据对比 (MAPE / MSE)")
                if overall_mse is None:
                    st.info("无法对齐真实数据，请确认 time/topic 匹配。")
                else:
                    st.write(f"总体 MAPE: {overall_mape:.3f}%")
                    st.write(f"总体 WMAPE: {overall_wmape:.3f}%")
                    st.write(f"总体 MSE: {overall_mse:.6f}")
                    st.dataframe(per_topic)
                    fig = plot_metrics(per_topic)
                    if fig:
                        st.pyplot(fig)


if __name__ == "__main__":
    main()
