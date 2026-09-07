#!/bin/bash
# 内存监控脚本 - 当内存用量达到450G时停止进程

THRESHOLD_GB=450
CHECK_INTERVAL=5  # 每5秒检查一次

echo "=========================================="
echo "内存监控启动"
echo "阈值: ${THRESHOLD_GB}GB"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo "=========================================="

while true; do
    # 获取当前内存使用（GB）
    USED_GB=$(free -g | awk '/^Mem:/ {print $3}')

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 内存使用: ${USED_GB}GB / ${THRESHOLD_GB}GB"

    # 检查是否超过阈值
    if [ "$USED_GB" -ge "$THRESHOLD_GB" ]; then
        echo ""
        echo "⚠️  内存使用超过阈值: ${USED_GB}GB >= ${THRESHOLD_GB}GB"
        echo "正在停止相关进程..."

        # 停止deepseek相关进程
        ps aux | grep -E "run_eval.py|gpu_worker.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

        echo "✅ 进程已停止"
        echo "监控结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        exit 0
    fi

    sleep "$CHECK_INTERVAL"
done
