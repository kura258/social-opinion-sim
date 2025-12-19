# 多智能体舆论模拟（Hawkes + LLM / Synthetic Policy）
本项目用于模拟多话题舆论热度随时间的演化：后端采用双衰减核 Hawkes 过程（每话题独立记忆项），前端用 Streamlit 做热榜/互动流展示，并支持与真实热度曲线对比（MAPE / WMAPE / MSE）。

核心目标：在“严格话题隔离”的前提下，让模拟曲线在不同热度量级的小事件/大事件上都能保持拟合稳定。

---

## 你会得到什么
- **多话题热度演化**：`env/social_env.py` 的 `TopicManager` 使用双衰减记忆项（fast/slow）更新每个 topic 的 Hawkes 热度。
- **冷启动底座**：仿照 TrendSim，在 `t=0` 注入每个话题的“官方种子贴”，保证后续互动一定能锁定话题语境。
- **严格话题隔离**：互动类动作（like/comment/retweet）会强制继承目标帖的 topic，避免“串味”。
- **五类动作**：`silent / like / comment / retweet / post`，并区分不同动作对热度的贡献权重。
- **认知状态**：Agent 具备 `emotion` 与 `social_confidence`（0~1），LLM 在提示词中走“感知→决策→反思”的认知回路。
- **两种运行模式**：
  - `LLM 模式`：需要 `CLOSEAI_API_KEY`，产出更像社交媒体的内容与互动。
  - `Synthetic 模式`：不依赖 LLM，用启发式策略生成动作，适合离线调参/回归验证（可重复）。
- **拟合指标更稳**：前端对比指标采用过滤版 MAPE（过滤掉极小分母点）并额外给出 WMAPE。

---

## 目录结构（按阅读顺序）
- `app_streamlit.py`：Streamlit 前端（话题热榜、互动流、曲线、指标），支持上传真实 CSV 或使用默认数据。
- `simulate.py`：组装 Agent/图关系/环境并运行模拟；包含离线对比入口 `run_and_compare()`。
- `env/social_env.py`：环境主逻辑（种子注入、背景流量、TopicManager、topic 强制对齐、动作落盘）。
- `agents/agent.py`：Agent 状态（emotion/confidence）、观察数据打包、动作解析与目标选择策略。
- `agents/batch_processor.py`：批处理 Prompt 构造（5 动作 + 认知回路 + 话题对齐约束）与 JSON 解析。
- `config/settings.py`：默认话题、真实数据路径、Hawkes/LLM 默认配置。
- `dataset_peak350/classified_events_35_2024Q1-Q4_peak350_v2.csv`：默认真实热度数据（35 话题）。
- `scripts/tune_params.py`：离线坐标下降式调参脚本（近似“梯度下降”的可执行版本）。

---

## 安装（Windows / PowerShell）
建议 Python 3.10+。

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 真实数据 CSV 格式
支持两种格式之一（列名必须匹配）：
- `topic, heat, timestamp`（推荐；每个 topic 内按 timestamp 排序后映射到 time=1..T）
- `topic, heat, time`（time 直接用于对齐）

说明：
- `topic`：话题名称（字符串）
- `heat`：热度（数值，允许百万级）
- `timestamp`：时间戳（用于排序）

---

## 运行方式

### 1) Streamlit 前端（LLM 模式）
需要在环境变量或 `.env` 中配置：`CLOSEAI_API_KEY`。

```bash
streamlit run app_streamlit.py
```

侧边栏关键参数：
- `base_weight`：真实曲线引导权重（越大越贴近真实；过大可能压制 Hawkes/Agent 的贡献）
- `bg_intensity_ratio`：背景流量比例（解决冷启动、尾部平滑；过大可能顶起长尾）
- `volume_factor`：Agent 声量系数（大事件拟合差时通常需要提高/或提高 role 权重）

### 2) 命令行离线对比（Synthetic 模式，默认不需要 Key）
`simulate.py` 默认会读取真实数据、自动对齐 `heat_scale` 并生成对比图：

```bash
python simulate.py
```

如果你想用 LLM 跑离线对比：

```bash
python -c "from simulate import run_and_compare; run_and_compare(T=350, seed=123, use_llm=True)"
```

### 3) 离线“梯度下降式”调参（推荐）
该脚本使用 Synthetic policy（可复现、适合搜索参数），输出最优参数与最差话题列表：

```bash
python scripts\\tune_params.py --steps 350 --rounds 3
```

使用建议（坐标下降 / 近似梯度下降）：
1. 先调 `base_weight`（通常对整体误差最敏感）
2. 再调 `bg_intensity_ratio`（控制冷启动与长尾）
3. 最后调 `volume_factor`（控制大事件的“嗓门”是否推得动）

把脚本输出的参数填回 Streamlit 侧边栏，再用 LLM 实跑验证内容质量与互动真实性。

---

## 关键机制说明

### 话题种子注入（防“凭空发起话题”）
环境在 `t=0` 对每个 topic 注入一条“官方种子贴”，后续互动可自然指向种子贴，锁定语境：
- 位置：`env/social_env.py:_inject_seed_posts()`
- 保存：`self.seed_post_ids[topic] = post.id`

### 严格话题隔离（防串味）
- LLM 输出中，like/comment/retweet 必须指定 `target_post_id`（来自 observation_items）并让 `topic=target.topic`
- 即使 LLM 没给或给错，后处理也会：
  - 强制互动类动作继承目标帖的 topic
  - 找不到目标则降级为 `silent`（避免“张冠李戴”）

### 背景流量（TrendSim 底座）
- `BackgroundTrafficGenerator` 生成 background count，并按话题权重分摊注入 `TopicManager.add_volume()`（只影响强度，不写入帖子列表）。
- 可通过 `bg_intensity_ratio` 与 `heat_scale` 同步缩放。

### 动作→热度贡献（Volume）
不同动作权重不同（轻量到重）：
- like: 0.1
- comment: 0.5
- post: 1.0
- retweet: 1.2

再乘以：
- `topic_scale * volume_factor`（每个话题按自身真实峰值量级伸缩）
- `agent.weight_ratio` 与 KOL/Official 加成

---

## 指标口径
前端会显示：
- `MAPE`：过滤版 MAPE（真实值过小的点不计入）
- `WMAPE`：加权 MAPE（用真实值做权重，更适合“大事件”）
- `MSE`

指标实现位置：`app_streamlit.py: compute_metrics()`。

---

## 关键环境变量
- `CLOSEAI_API_KEY`（LLM 模式必填）：API Key
- `CLOSEAI_BASE_URL`：兼容 OpenAI 风格的 base_url
- `CLOSEAI_MODEL`：模型名
- `REAL_DATA_PATH`：真实热度 CSV 路径（不提供则回退 `dataset_peak350/...csv`）
- `HAWKES_PARAM_PATH`：Hawkes 参数文件路径（默认读取 `artifacts/hawkes_params.json`/内置默认）
- `DATA_DIR`：训练/加载数据目录（默认按 `config/settings.py`）

---

## 常见问题（Troubleshooting）
- **误差偏大**：确认真实数据 CSV 的 `topic/heat/(timestamp|time)` 列是否正确；并确保 `heat_scale` 已对齐（Streamlit 会自动推断，离线对比也会从真实曲线推断）。
- **话题不够“隔离”**：互动类动作需要指向明确目标帖；项目会优先引导同话题互动并在后处理中强制继承目标 topic。
- **没有 API Key 也想运行**：使用 `python simulate.py` 或 `python scripts\\tune_params.py ...`（Synthetic 模式不需要 Key）。
