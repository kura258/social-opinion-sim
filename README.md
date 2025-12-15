# 多智能体舆论博弈模拟（话题热度预测版）

基于 20 个人设 + Hawkes 双衰减核的舆论演化实验，支持训练、数据制导仿真、前端可视化与真实数据对比。

## 近期核心特性
- **数据制导仿真**：上传/默认 CSV 的真实热度轨迹作为“指挥棒”，按话题刚性配额；缺失时回退 Hawkes 惯性。自动识别量级、初始热度，并同步显示值避免曲线漂移。
- **Persona 与话题背景**：`config/personas.py` 定义 20 个细粒度画像；`utils/topic_helper.py` 扩展话题背景；Agent 在指定话题上结合背景与“广场声音”发声，减少复读。
- **离散化与漂移校正**：`utils/simulation_core.py` 提供抖动量化与 PID 校正，解决连续热度到离散动作的能量丢失与累计漂移。
- **前端实时指标**：Streamlit 顶部实时 MAPE/MSE 数值与折线；按话题热度曲线、帖子列表、行为时间线。

## 环境准备
```bash
python -m venv venv
# macOS/Linux: source venv/bin/activate
# PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置与数据
- 默认话题：`config/settings.py::DEFAULT_TOPICS`
- 数据目录：`DATA_DIR` 或默认 `datasets_huoju_norm`，不存在则回退 `dataset_peak350`
- 真实数据：`config.settings.DEFAULT_REAL_DATA_PATH`（含 `classified_events_35_2024Q1-Q4_peak350_v2.csv`）
- Hawkes 参数：优先 `artifacts/hawkes_params.json`，否则默认值（含 `heat_scale`）
- LLM：`CLOSEAI_API_KEY/CLOSEAI_BASE_URL/CLOSEAI_MODEL` 可覆盖默认

## 训练 Hawkes 参数
```bash
python train.py           # 默认 80/10/10 切分，归一化开启
# CMA-ES + 多起点 L-BFGS-B
python train.py --use_global_init --global_n_starts 30 --cma_maxiter 40 \
  --cma_popsize 16 --cma_sigma0 0.3 --perturb_scale 0.15 --lbfgs_maxiter 300
```

## 模拟与对比
```bash
python simulate.py             # CLI 快速对比，生成 simulation_vs_real.png
streamlit run app_streamlit.py # 前端交互
```
- 前端上传/默认 CSV 后自动识别量级、初始热度、轨迹；话题缺失自动忽略；可自定义话题或用真实数据话题。
- 仿真按话题精英制配额（每话题最多 3 人），写回真实权重；数据制导时强制同步热度显示，确保曲线与目标对齐。

## 主要模块
- `config/settings.py`：默认话题/路径/Hawkes/LLM 参数，加载工件
- `config/personas.py`：20 个画像（权重比 + 人设）
- `utils/topic_helper.py`：话题背景生成
- `utils/simulation_core.py`：抖动量化、PID 控制
- `agents/agent.py`：话题定向发声（背景+广场上下文）
- `env/social_env.py`：数据制导/预测混合调度、精英配额、状态同步
- `simulate.py`：支持 `fixed_heat_scale`、初始热度、真实轨迹
- `app_streamlit.py`：前端交互与实时指标
- `train.py`：CMA-ES + L-BFGS-B 拟合

## 小贴士
- 上传的 CSV 建议含 `topic/timestamp/heat`，时间序按行序处理；缺失话题会自动忽略。
- 若全局预测漂移，可调节 `HeatDither.unit_heat` 或 PID 参数；如需纯预测，可不提供真实轨迹。
