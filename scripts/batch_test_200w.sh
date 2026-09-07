#!/bin/bash
# 200万数据集批量评估 - 依次运行所有模型

# 激活 conda 环境 cvlface（conda 不在默认位置时，请设置 CONDA_BASE=<你的conda目录>）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/bin/activate" cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "200万数据集 - 批量评估"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 结果汇总文件
SUMMARY="results_200w_summary.txt"
> $SUMMARY

# 测试函数
test_model() {
    local name=$1
    local dir=$2
    local script=$3
    shift 3
    local args=("$@")

    echo ""
    echo "========================================"
    echo "[$name] 开始测试"
    echo "========================================"

    local log="${name}_200w.log"
    local start=$(date +%s)

    cd "$dir"
    if timeout 600 python "$script" "${args[@]}" > "../$log" 2>&1; then
        local end=$(date +%s)
        local elapsed=$((end - start))
        echo "✅ $name: 成功 (${elapsed}秒)" | tee -a "../$SUMMARY"
        tail -30 "../$log" | grep -E "(TPIR|总耗时|完成)" | head -10
    else
        local end=$(date +%s)
        local elapsed=$((end - start))
        echo "❌ $name: 失败 (${elapsed}秒)" | tee -a "../$SUMMARY"
        tail -20 "../$log"
    fi
    cd ..
}

# 1. qwen (已完成，跳过)
echo "✅ qwen: 已完成 (796秒)" | tee -a "$SUMMARY"

# 2. glm-pro
test_model "glm-pro" "glm-pro" "step3_multi_gpu.py" --pkl ../test_data_200w.pkl

# 3. glm
test_model "glm" "glm" "eval_similar.py" --pkl ../test_data_200w.pkl

# 4. deepseek-pro
test_model "deepseek-pro" "deepseek-pro" "step4_eval_optimized.py" --data ../test_data_200w.pkl

# 5. gemini
test_model "gemini" "gemini" "face_eval_system.py" --data_path ../test_data_200w.pkl --font_path ../font/SourceHanSansSC-Normal.otf

# 6. grok
test_model "grok" "grok" "eval_similarity.py" --pkl ../test_data_200w.pkl

# 7. codex-sol
test_model "codex-sol" "codex-sol" "face_similarity_eval.py" --input ../test_data_200w.pkl --font ../font/SourceHanSansSC-Normal.otf

# 8. codex-sol-2
test_model "codex-sol-2" "codex-sol-2" "face_similarity_evaluator.py" --input ../test_data_200w.pkl --font-path ../font/SourceHanSansSC-Normal.otf

# 9. codex-sol-m
test_model "codex-sol-m" "codex-sol-m" "face_similarity_eval.py" --input ../test_data_200w.pkl

echo ""
echo "========================================"
echo "批量评估完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
cat "$SUMMARY"
