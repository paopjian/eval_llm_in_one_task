#!/bin/bash
# 200万数据集 - 11个模型完整评估

source /root/miniconda3/bin/activate cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_200w_evaluation.py 2>&1 | tee logs/evaluation_200w.log
