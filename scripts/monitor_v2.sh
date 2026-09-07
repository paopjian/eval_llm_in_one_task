#!/bin/bash
TIMEOUT=1800  # 30分钟
START=$(date +%s)

while true; do
    # 检查进程是否存在
    if ! ps -p 3287553 > /dev/null 2>&1; then
        echo "进程已结束"
        exit 0
    fi
    
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "========================================"
        echo "超时: codex-sol-m运行超过30分钟"
        echo "已运行: $((ELAPSED/60))分钟"
        echo "========================================"
        echo "❌ codex-sol-m: 超时 (>1800秒)" >> results_200w_final.txt
        pkill -9 -f "face_similarity_eval"
        exit 1
    fi
    
    sleep 30
done
