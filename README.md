# 多智能体舆论模拟（Hawkes + LLM）
基于 Hawkes 双衰减核和 20 个人设的舆情演化实验，支持真实热度对齐、前端可视化与全局参数训练。

## 主要特性
- Hawkes 驱动的热度演化：`env/social_env.py` 的 `TopicManager` 使用纯 Hawkes 记忆项（双衰减核），按真实量级推断 `heat_scale`，可接入真实热度轨迹做数据制导。
- 角色画像与风格约束：`config/personas.py` 提供 20 个人设，`config/styles.py` 定义语气/关键词/句式；`agents/agent.py` 将 LLM 输出与快反概率决策结合。
- 环境场与配额调度：`SocialEnv.step` 将 Hawkes 强度映射为可见度/风险/奖励场，计算角色出场倾向并采样动作，支持强制发声模式。
- 可视化与对比：`app_streamlit.py` 展示话题曲线、帖子时间线、MAPE/MSE 实时指标；`simulate.py` 的 `run_and_compare` 生成 `simulation_vs_real.png`。
- 训练与校准：`train.py` 用 CMA-ES + 多起点 L-BFGS-B 拟合全局 Hawkes 参数，并提供基于真实曲线的角色参数校准。

## 目录与模块
- `config/settings.py`：默认话题、真实数据路径、Hawkes/LLM 配置加载。
- `config/personas.py` / `config/styles.py`：角色画像与语言风格。
- `agents/agent.py`：LLM+概率混合决策、记忆流、强制发声接口；`agents/llm_client.py` 读取 `CLOSEAI_*` 环境变量；`agents/memory.py` 记忆检索。
- `env/social_env.py`：Hawkes 主题管理、环境场生成、两种步进逻辑（`step_legacy` 为旧版，`step` 为现用）。
- `simulate.py`：构建代理与关系图，运行多步模拟，并提供真实数据对比与绘图。
- `app_streamlit.py`：前端交互，支持上传/默认 CSV，动态选话题与种子。
- `train.py`：全局 Hawkes 参训与角色参数校准。
- `utils/topic_helper.py`：话题背景生成；`utils/data_loader.py` / `utils/spread_model.py`：训练数据加载与 Hawkes 预测工具。
- 数据：`dataset_peak350/classified_events_35_2024Q1-Q4_peak350_v2.csv` 为默认真实轨迹回退（可用 `REAL_DATA_PATH` 覆盖），`artifacts/hawkes_params.json` 存储拟合结果。

## 安装
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell，其他终端按需调整
pip install -r requirements.txt
```

## 运行模拟（命令行）
```bash
python simulate.py            # 使用默认话题与 Hawkes 参数，生成 simulation_vs_real.png
```
- 可通过 `config/settings.py` 修改默认话题/种子/路径，或覆盖 `CLOSEAI_API_KEY`、`REAL_DATA_PATH`、`HAWKES_PARAM_PATH`、`DATA_DIR` 等环境变量。

## 前端可视化
```bash
streamlit run app_streamlit.py
```
- 可上传自定义 `topic,heat,timestamp` CSV 或使用默认数据；实时展示每话题曲线、帖子时间线、MAPE/MSE。

## 训练 Hawkes 参数
```bash
# 直接多起点 L-BFGS-B
python train.py

# 全局搜索 + 精调
python train.py --use_global_init --global_n_starts 30 --cma_maxiter 40 \
  --cma_popsize 16 --cma_sigma0 0.3 --perturb_scale 0.15 --lbfgs_maxiter 300
```
训练结果会更新 `artifacts/hawkes_params.json`，供模拟加载。

## 关键环境变量
- `CLOSEAI_API_KEY`（必填）：LLM 调用 Key；`CLOSEAI_BASE_URL`、`CLOSEAI_MODEL` 可选。
- `REAL_DATA_PATH`：真实热度 CSV 路径，未提供时回退 `dataset_peak350/...csv`。
- `HAWKES_PARAM_PATH`：训练好的参数文件，未提供时用 `config.settings.DEFAULT_HAWKES_PARAMS`。
- `DATA_DIR`：训练/加载数据目录，默认 `datasets_huoju_norm`，不存在则回退 `dataset_peak350`。
