#!/bin/bash
# 监控评估进程，30分钟超时

PID=3287553
TIMEOUT=1800  # 30分钟
START=$(date +%s)

while kill -0 $PID 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "========================================"
        echo "超时警告: codex-sol-m运行超过30分钟"
        echo "开始时间: $(date -d @$START '+%Y-%m-%d %H:%M:%S')"
        echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "已运行: $((ELAPSED/60))分钟"
        echo "========================================"
        
        # 记录到文件
        echo "❌ codex-sol-m: 超时 (>1800秒)" >> results_200w_final.txt
        echo "原因: 运行时间超过30分钟限制" >> results_200w_final.txt
        
        # 终止进程
        pkill -9 -f "face_similarity_eval"
        exit 1
    fi
    
    sleep 30
done

# 正常完成
END=$(date +%s)
ELAPSED=$((END - START))
echo "codex-sol-m完成，耗时: $((ELAPSED/60))分钟$((ELAPSED%60))秒"
