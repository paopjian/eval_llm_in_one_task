#!/bin/bash
# 全模型性能基准测试
# 测试所有模型代码在1分钟级和10分钟级数据集上的性能

set -e

echo "========================================================================"
echo "全模型性能基准测试"
echo "========================================================================"
echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 激活 conda 环境 cvlface（conda 不在默认位置时，请设置 CONDA_BASE=<你的conda目录>）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/bin/activate" cvlface

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查测试数据
echo "检查测试数据..."
if [ ! -f "test_data_1min.pkl" ]; then
    echo "❌ 缺少 test_data_1min.pkl"
    exit 1
fi

if [ ! -f "test_data_10min.pkl" ]; then
    echo "❌ 缺少 test_data_10min.pkl"
    exit 1
fi

echo "✅ 测试数据就绪"
echo "  - test_data_1min.pkl (75K样本, 2.8B对)"
echo "  - test_data_10min.pkl (200K样本, 20B对)"
echo ""

# 创建结果目录
mkdir -p benchmark_results
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_FILE="benchmark_results/all_models_${TIMESTAMP}.json"

echo "结果将保存到: $RESULT_FILE"
echo ""

# 初始化结果文件
cat > "$RESULT_FILE" <<EOF
{
  "test_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "test_datasets": {
    "1min": {
      "file": "test_data_1min.pkl",
      "samples": 75000,
      "pairs": "2.8B"
    },
    "10min": {
      "file": "test_data_10min.pkl",
      "samples": 200000,
      "pairs": "20B"
    }
  },
  "results": {}
}
EOF

echo "========================================================================"
echo "开始测试所有模型"
echo "========================================================================"
echo ""

# 测试函数
test_model() {
    local model_name=$1
    local model_dir=$2
    local main_script=$3
    local data_file=$4
    local dataset_name=$5
    local timeout=$6

    echo "--------------------------------------------------------------------"
    echo "测试: $model_name ($dataset_name)"
    echo "脚本: $main_script"
    echo "数据: $data_file"
    echo "--------------------------------------------------------------------"

    # 复制数据文件到模型目录
    cp -f "$data_file" "$model_dir/"
    data_filename=$(basename "$data_file")

    cd "$model_dir"

    # 运行测试
    start_time=$(date +%s)

    if timeout ${timeout}s python "$main_script" "$data_filename" > /tmp/model_output.log 2>&1; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        status="success"
        echo "✅ 测试成功"
        echo "⏱️  耗时: ${elapsed}秒"

        # 提取性能数据（如果有）
        if grep -q "耗时\|time\|seconds" /tmp/model_output.log; then
            echo "📊 输出摘要:"
            grep -E "耗时|time|seconds|TPIR|performance" /tmp/model_output.log | head -5
        fi
    else
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        if [ $elapsed -ge $timeout ]; then
            status="timeout"
            echo "⏰ 测试超时 (>${timeout}秒)"
        else
            status="failed"
            echo "❌ 测试失败"
            echo "错误信息:"
            tail -10 /tmp/model_output.log
        fi
    fi

    # 记录结果
    python3 -c "
import json
import sys

result_file = '$RESULT_FILE'
with open(result_file, 'r') as f:
    data = json.load(f)

if '$model_name' not in data['results']:
    data['results']['$model_name'] = {}

data['results']['$model_name']['$dataset_name'] = {
    'status': '$status',
    'time_seconds': $elapsed,
    'script': '$main_script'
}

with open(result_file, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
"

    cd - > /dev/null
    echo ""
}

# 定义所有模型及其主脚本
declare -A models=(
    ["claude-opus"]="eval_v5_final.py"
    ["gemini"]="face_eval_system.py"
    ["codex-sol-2"]="face_similarity_evaluator.py"
    ["deepseek"]="run_eval.py"
    ["grok"]="eval_similarity.py"
    ["glm"]="eval_similar.py"
)

# 1分钟级测试（较小数据集，所有模型都测）
echo "###################################################################"
echo "# 第一轮：1分钟级数据集 (75K样本)"
echo "###################################################################"
echo ""

for model in "${!models[@]}"; do
    script="${models[$model]}"
    test_model "$model" "$model" "$script" "test_data_1min.pkl" "1min" 300
done

# 10分钟级测试（大数据集，只测试成功的模型）
echo ""
echo "###################################################################"
echo "# 第二轮：10分钟级数据集 (200K样本)"
echo "###################################################################"
echo ""

for model in "${!models[@]}"; do
    # 检查1分钟级测试是否成功
    success=$(python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
status = data.get('results', {}).get('$model', {}).get('1min', {}).get('status', '')
print(status)
" 2>/dev/null || echo "")

    if [ "$success" = "success" ]; then
        script="${models[$model]}"
        test_model "$model" "$model" "$script" "test_data_10min.pkl" "10min" 1800
    else
        echo "⚠️  跳过 $model (1分钟级测试未成功)"
        echo ""
    fi
done

# 生成测试报告
echo "========================================================================"
echo "测试完成！生成报告..."
echo "========================================================================"

python3 -c "
import json
from datetime import datetime

with open('$RESULT_FILE') as f:
    data = json.load(f)

print()
print('='*70)
print('测试结果摘要')
print('='*70)
print(f\"测试时间: {data['test_time']}\")
print()

for dataset in ['1min', '10min']:
    dataset_info = data['test_datasets'][dataset]
    print(f\"【{dataset}数据集】- {dataset_info['samples']:,}样本, {dataset_info['pairs']}对\")
    print(f\"{'模型':<20} {'状态':<10} {'耗时':<15}\")
    print('-' * 50)

    results = []
    for model, model_data in data['results'].items():
        if dataset in model_data:
            result = model_data[dataset]
            results.append((model, result))

    # 按时间排序
    results.sort(key=lambda x: x[1]['time_seconds'] if x[1]['status'] == 'success' else float('inf'))

    for model, result in results:
        status = result['status']
        time_s = result['time_seconds']

        status_icon = {
            'success': '✅',
            'failed': '❌',
            'timeout': '⏰'
        }.get(status, '❓')

        if time_s < 60:
            time_str = f\"{time_s}秒\"
        else:
            time_str = f\"{time_s/60:.1f}分钟\"

        print(f\"{model:<20} {status_icon} {status:<8} {time_str:<15}\")
    print()

print('='*70)
print(f\"详细结果已保存到: $RESULT_FILE\")
print('='*70)
"

echo ""
echo "✅ 全部测试完成！"
