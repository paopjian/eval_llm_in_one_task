#!/bin/bash
# 200万数据集 - 11个模型完整评估

# 激活 conda 环境 cvlface（conda 不在默认位置时，请设置 CONDA_BASE=<你的conda目录>）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/bin/activate" cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_200w_evaluation.py 2>&1 | tee logs/evaluation_200w.log
