#!/bin/bash
# 快速测试脚本：运行step3并记录性能数据
# 注意：此脚本引用外部目录glm-5.3，请根据实际环境修改路径

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLM_DIR="$(cd "$SCRIPT_DIR/../../glm-5.3" 2>/dev/null && pwd || echo "$SCRIPT_DIR")"

# Python 解释器：默认使用 PATH 中的 python（通常是已激活的 conda 环境）。
# 如需指定环境：PYTHON_BIN="$HOME/miniconda3/envs/cvlface/bin/python" bash quick_test.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$GLM_DIR"

echo "========================================="
echo "快速性能测试 (7卡并行)"
echo "========================================="

"$PYTHON_BIN" step3_multi_gpu.py \
    --gpus 0,1,2,3,4,5,6 \
    --block-rows 8192 \
    --block-cols 65536 \
    --no-plot \
    --no-extract \
    2>&1 | tee run_performance.log

echo ""
echo "========================================="
echo "关键性能指标提取："
echo "========================================="
grep -E "加载\+排序|多GPU执行|吞吐|总耗时" run_performance.log
