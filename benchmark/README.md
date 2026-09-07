# 统一评估框架 benchmark/

本目录是"同一任务、多家 LLM 实现"的**统一再评估**框架。

## 设计原则

> **读取数据的代码、直方图校验、TPIR@FPIR 指标、计时/超时/内存控制对所有模型完全一致，
> 各家 LLM 的差异只体现在 `cores/` 中的"核心计算"上。**

- 网格统一：相似度直方图一律 200,000 bins（bin 宽 1e-5）覆盖 [-1,1]
- 校验严格：`pos_hist` 计数必须恰等于理论正样本对数、`neg_hist` 恰等于理论负样本对数，
  否则该次运行判 `invalid`（防止"计数不一致但照样出报告"）
- 指标统一：TPIR@FPIR = {1e-5, 1e-4, 1e-3, 1e-2}，全部由统一直方图计算

## 结构

| 文件 | 作用 |
|------|------|
| `common.py` | 统一实现：数据读取、网格/直方图工具、计数校验、TPIR@FPIR 指标 |
| `cores/baseline.py` | cluster_utils 基准方法（仓库内标准副本 `cluster_utils.py`，默认 v5；可用 `CLUSTER_UTILS_PATH`/`CLUSTER_UTILS_VER` 覆盖） |
| `cores/<model>.py` | 各 LLM 实现的核心计算（从各模型目录原代码提炼，注释标明出处与参数） |
| `run_one.py` | 单模型执行器：读取 -> core -> 校验 -> 指标 -> JSON |
| `run_eval.py` | 两阶段调度器：超时/内存监控/晋级流程 |
| `smoke_check.py` | 冒烟测试：小切片朴素参考直方图与 core 结果逐项对比 |

## 评估流程（两阶段）

1. **200K 阶段**（`test_data_10min.pkl`，20 亿对）：全部模型 + 基准，单模型
   60s 超时、进程树内存上限 400GB —— 与基准对比，筛出"200K ≤60s 且校验通过"的模型；
2. **2M 阶段**（`test_data_200w.pkl`，2 万亿对）：只跑合格模型 + 基准，单模型
   30min 超时、进程树内存上限 400GB。

```bash
# 环境：conda activate cvlface（或任意有 torch+CUDA 的环境）

# 1. 冒烟验证（可选，N=25000 切片对比朴素参考）
python benchmark/smoke_check.py --models glm,codex-sol-2 --max-n 25000

# 2. 两阶段全流程（200K 自动筛选后跑 2M）
python benchmark/run_eval.py --stage all

# 只跑 200K / 只跑 2M（2M 需先有 qualified.json，或 --models 指定）
python benchmark/run_eval.py --stage 200k
python benchmark/run_eval.py --stage 2m

# 覆盖默认参数
python benchmark/run_eval.py --stage 200k --timeout 60 --mem-cap-gb 400 \
    --models baseline,glm,grok --outdir logs/unified_eval/my_run
```

输出统一在 `logs/unified_eval/<run>/`：每模型 `<model>.json`（校验+指标+耗时）、
`<model>.log`（原始输出）、`<stage>/summary.json`（排序汇总）、`qualified.json`（200K 合格名单）。

## core 契约

```python
def compute(feats: np.ndarray,   # (N,512) float32，已 L2 归一化（统一 loader 产出）
             ids: np.ndarray,    # (N,) int64
             gpus: list[int],    # 可用 GPU 编号
             workdir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (pos_hist, neg_hist, meta)
    pos_hist/neg_hist: (200000,) int64，bin k 覆盖 [LO+k*w, LO+(k+1)*w)
    meta: 模型/方法/出处/并行/精度/分块等说明（不含本机路径）"""
```

只允许使用 `numpy`/`torch`/`torch.multiprocessing`/标准库及 `common.py` 提供的工具；
多进程 worker 必须模块级顶层定义（spawn 安全）；禁止在 core 内读取数据文件或做指标计算。

## 各 core 出处

| core | 来源（模型目录原实现） | 核心特征 |
|------|------------------------|----------|
| `baseline` | 仓库内 `benchmark/cluster_utils.py` `get_sim_matrix_large_scale_v5` | 动态调度分块 + 内部 20M bins（聚合到统一网格；裁剪版仅保留评估相关代码） |
| `glm` | `glm/eval_similar.py` | spawn + 动态 tile 队列；fp32（关 TF32） |
| `grok` | `grok/eval_similarity.py` | 线程池（每 GPU 一线程）；对数切分；fp32 分块 histc |
| `qwen` | `qwen/eval_v2_multi_gpu.py` | fork；按行工作量静态均衡；对角块列跳过；bincount |
| `codex-sol` | `codex-sol/face_similarity_eval.py` | ... |
| `codex-sol-2` | `codex-sol-2/face_similarity_evaluator.py` | ... |
| `gemini` | `gemini/face_eval_system.py` | ... |
| `deepseek` | `deepseek/run_eval.py` | ... |
| `glm-pro` | `glm-pro/step3_multi_gpu.py` | ... |
| `deepseek-pro` | `deepseek-pro/step4_eval_optimized.py` | ... |
| `claude-opus` | `claude-opus/eval_v5_final.py` | ... |
| `claude-opus-m` | `claude-opus-m/eval_similarity_final.py` | ... |
| `codex-sol-m` | `codex-sol-m/face_similarity_eval.py` | ... |

（表格"核心特征"列待全部 core 完成后回填，各 core 文件 docstring 中有详细出处。）
