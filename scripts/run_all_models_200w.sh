#!/bin/bash
# 200万数据集 - 11个模型完整评估
# 使用cvlface环境

# 激活 conda 环境 cvlface（conda 不在默认位置时，请设置 CONDA_BASE=<你的conda目录>）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/bin/activate" cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "200万数据集 - 11个模型完整评估"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# 初始化结果文件
cat > results_200w_final.json << 'EOF'
{
  "test_time": "$(date -Iseconds)",
  "test_data": "test_data_200w.pkl",
  "data_size": "2000000 samples, 2000B pairs",
  "results": {}
}
EOF

# 测试函数
run_test() {
    local model=$1
    local script=$2
    shift 2
    local args=("$@")

    echo ""
    echo "========================================="
    echo "测试: $model"
    echo "========================================="

    cd "$model" || return 1

    start_time=$(date +%s)

    if python "$script" "${args[@]}" > "../${model}_200w.log" 2>&1; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "✅ 成功 - 耗时: ${elapsed}秒"
        echo "$model: SUCCESS (${elapsed}s)" >> ../results_200w_summary.txt
    else
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "❌ 失败 - 耗时: ${elapsed}秒"
        echo "$model: FAILED (${elapsed}s)" >> ../results_200w_summary.txt
        tail -20 "../${model}_200w.log"
    fi

    cd ..
}

# 清空汇总
> results_200w_summary.txt

# 1. qwen
run_test "qwen" "eval_v1_single_gpu.py" --pkl ../test_data_200w.pkl

# 2. glm
run_test "glm" "eval_similar.py" --pkl ../test_data_200w.pkl

# 3. glm-pro
run_test "glm-pro" "step3_multi_gpu.py" --pkl ../test_data_200w.pkl

# 4. deepseek-pro
run_test "deepseek-pro" "step4_eval_optimized.py" --data ../test_data_200w.pkl

# 5. gemini
run_test "gemini" "face_eval_system.py" --data_path ../test_data_200w.pkl --font_path ../font/SourceHanSansSC-Normal.otf

# 6. grok
run_test "grok" "eval_similarity.py" --pkl ../test_data_200w.pkl

# 7. codex-sol
run_test "codex-sol" "face_similarity_eval.py" --input ../test_data_200w.pkl --font ../font/SourceHanSansSC-Normal.otf

# 8. codex-sol-2
run_test "codex-sol-2" "face_similarity_evaluator.py" --input ../test_data_200w.pkl --font-path ../font/SourceHanSansSC-Normal.otf

# 9. codex-sol-m
run_test "codex-sol-m" "face_similarity_eval.py" --input ../test_data_200w.pkl

# 10. claude-opus (需要修改数据路径)
echo "⚠️  claude-opus 需要修改脚本中的硬编码路径，跳过"

# 11. deepseek (需要修改数据路径)
echo "⚠️  deepseek 需要修改脚本中的硬编码路径，跳过"

echo ""
echo "========================================="
echo "评估完成"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo ""
echo "结果汇总:"
cat results_200w_summary.txt
