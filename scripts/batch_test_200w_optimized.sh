#!/bin/bash
# 200万数据集批量评估 - 优化版（显存管理）

# 激活 conda 环境 cvlface（conda 不在默认位置时，请设置 CONDA_BASE=<你的conda目录>）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/bin/activate" cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "200万数据集 - 批量评估（优化版）"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 结果汇总文件
SUMMARY="results_200w_final.txt"
> $SUMMARY

# 已完成的模型列表
COMPLETED="qwen glm gemini grok codex-sol codex-sol-2"

echo "已完成的模型（跳过）：$COMPLETED" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# 测试函数
test_model() {
    local name=$1
    local dir=$2
    local script=$3
    shift 3
    local args=("$@")

    # 检查是否已完成
    if echo "$COMPLETED" | grep -q "$name"; then
        echo "⏭️  $name: 已完成，跳过" | tee -a "$SUMMARY"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "[$name] 开始测试"
    echo "========================================"

    # 清理GPU显存
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 2

    local log="${name}_200w_retry.log"
    local start=$(date +%s)

    cd "$dir"
    if timeout 1200 python "$script" "${args[@]}" > "../$log" 2>&1; then
        local end=$(date +%s)
        local elapsed=$((end - start))
        echo "✅ $name: 成功 (${elapsed}秒)" | tee -a "../$SUMMARY"
        tail -30 "../$log" | grep -E "(TPIR|总耗时|完成|threshold)" | head -10
    else
        local end=$(date +%s)
        local elapsed=$((end - start))
        echo "❌ $name: 失败 (${elapsed}秒)" | tee -a "../$SUMMARY"
        echo "错误信息:" | tee -a "../$SUMMARY"
        tail -30 "../$log" | grep -E "(Error|错误|OOM)" | head -5 | tee -a "../$SUMMARY"
    fi
    cd ..
}

# 待评估的模型（按优先级排序）

# 1. codex-sol-m - 跳过（代码实现有问题）
echo ""
echo "========================================"
echo "[codex-sol-m] 跳过"
echo "原因: multiprocessing并行失败，只有1个GPU工作"
echo "========================================"
echo "⚠️  codex-sol-m: 跳过 (代码实现问题)" | tee -a "$SUMMARY"

# 2. glm-pro - 跳过（系统内存爆满）
echo ""
echo "========================================"
echo "[glm-pro] 跳过"
echo "原因: 代码设计问题，系统内存持续增长直至爆满"
echo "========================================"
echo "⚠️  glm-pro: 跳过 (代码实现问题)" | tee -a "$SUMMARY"

# 3. deepseek-pro - 跳过（系统内存爆满）
echo ""
echo "========================================"
echo "[deepseek-pro] 跳过"
echo "原因: 代码设计问题，系统内存持续增长直至爆满"
echo "========================================"
echo "⚠️  deepseek-pro: 跳过 (代码实现问题)" | tee -a "$SUMMARY"

# 4. claude-opus (需修改路径 - 传递参数)
echo ""
echo "========================================"
echo "[claude-opus] 尝试传递数据路径"
echo "========================================"
cd claude-opus
# 检查是否支持参数
if python eval_v5_final.py --help 2>&1 | grep -q "pkl\|data"; then
    cd ..
    test_model "claude-opus" "claude-opus" "eval_v5_final.py" --pkl ../test_data_200w.pkl
else
    cd ..
    echo "⚠️  claude-opus: 脚本需要修改硬编码路径，跳过" | tee -a "$SUMMARY"
fi

# 5. deepseek (需修改路径)
echo ""
echo "========================================"
echo "[deepseek] 尝试传递数据路径"
echo "========================================"
cd deepseek
if python run_eval.py --help 2>&1 | grep -q "pkl\|data"; then
    cd ..
    test_model "deepseek" "deepseek" "run_eval.py" --pkl ../test_data_200w.pkl
else
    cd ..
    echo "⚠️  deepseek: 脚本需要修改硬编码路径，跳过" | tee -a "$SUMMARY"
fi

echo ""
echo "========================================"
echo "批量评估完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 统计结果
echo "========== 最终结果汇总 ==========" | tee -a "$SUMMARY"
cat "$SUMMARY"

# 统计成功/失败数量
SUCCESS=$(grep "✅" "$SUMMARY" | wc -l)
FAILED=$(grep "❌" "$SUMMARY" | wc -l)
SKIPPED=$(grep "⚠️\|⏭️" "$SUMMARY" | wc -l)

echo ""
echo "========== 统计 =========="
echo "✅ 成功: $SUCCESS"
echo "❌ 失败: $FAILED"
echo "⚠️  跳过: $SKIPPED"
