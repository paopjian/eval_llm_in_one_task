#!/usr/bin/env python3
"""
生成全模型基准测试的可视化图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# 设置中文字体（仓库内相对路径，兼容外部克隆）
font_path = str(Path(__file__).resolve().parent.parent / 'font' / 'SourceHanSansSC-Normal.otf')
if Path(font_path).exists():
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    print("⚠️ 未找到中文字体，使用默认字体")

# 读取测试结果
with open('all_models_benchmark_20260906_123510.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

baseline = data['baseline']['results']
models = data['models']

# 提取成功的模型数据
model_names = []
time_1min = []
time_10min = []
scaling_ratio = []

for name, model_data in models.items():
    tests = model_data.get('tests', {})

    if '1分钟级' in tests and tests['1分钟级']['status'] == 'success':
        if '10分钟级' in tests and tests['10分钟级']['status'] == 'success':
            t1 = tests['1分钟级']['time']
            t10 = tests['10分钟级']['time']

            model_names.append(name)
            time_1min.append(t1)
            time_10min.append(t10)
            scaling_ratio.append(t10 / t1)

# 添加基准
model_names.insert(0, 'cluster_utils\n(基准)')
time_1min.insert(0, baseline['1分钟级']['total_time'])
time_10min.insert(0, baseline['10分钟级']['total_time'])
scaling_ratio.insert(0, baseline['10分钟级']['total_time'] / baseline['1分钟级']['total_time'])

# 图1: 性能对比（耗时）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(model_names))
width = 0.35

# 1分钟级 vs 10分钟级
bars1 = ax1.barh(x, time_1min, width, label='1分钟级 (75K样本)', alpha=0.8, color='#3B6DFF')
bars2 = ax1.barh(x + width, time_10min, width, label='10分钟级 (200K样本)', alpha=0.8, color='#6EE7B7')

ax1.set_xlabel('耗时 (秒)', fontsize=12)
ax1.set_ylabel('模型', fontsize=12)
ax1.set_title('全模型性能对比 - 绝对耗时', fontsize=14, fontweight='bold')
ax1.set_yticks(x + width / 2)
ax1.set_yticklabels(model_names, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(baseline['1分钟级']['total_time'], color='red', linestyle='--', alpha=0.5, label='基准')

# 添加数值标签
for i, (v1, v10) in enumerate(zip(time_1min, time_10min)):
    if v1 < 100:
        ax1.text(v1 + 1, i, f'{v1:.1f}s', va='center', fontsize=8)
    else:
        ax1.text(v1 + 5, i, f'{v1:.0f}s', va='center', fontsize=8)

    if v10 < 100:
        ax1.text(v10 + 1, i + width, f'{v10:.1f}s', va='center', fontsize=8)
    else:
        ax1.text(v10 + 5, i + width, f'{v10:.0f}s', va='center', fontsize=8)

# 缩放比
colors = ['green' if r < 1.0 else 'orange' if r < 2.0 else 'red' for r in scaling_ratio]
bars3 = ax2.barh(x, scaling_ratio, color=colors, alpha=0.7)

ax2.set_xlabel('缩放比 (10min耗时 / 1min耗时)', fontsize=12)
ax2.set_ylabel('模型', fontsize=12)
ax2.set_title('性能缩放分析 (数据量7.1×)', fontsize=14, fontweight='bold')
ax2.set_yticks(x)
ax2.set_yticklabels(model_names, fontsize=10)
ax2.axvline(1.0, color='blue', linestyle='--', alpha=0.5, label='理想缩放')
ax2.axvline(7.1, color='red', linestyle='--', alpha=0.3, label='数据量缩放')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# 添加缩放比数值
for i, v in enumerate(scaling_ratio):
    ax2.text(v + 0.1, i, f'{v:.2f}×', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 图表1已保存: benchmark_comparison.png")

# 图2: vs基准倍数
fig, ax = plt.subplots(figsize=(12, 8))

baseline_1min = baseline['1分钟级']['total_time']
baseline_10min = baseline['10分钟级']['total_time']

vs_baseline_1min = [t / baseline_1min for t in time_1min]
vs_baseline_10min = [t / baseline_10min for t in time_10min]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, vs_baseline_1min, width, label='1分钟级', alpha=0.8, color='#3B6DFF')
bars2 = ax.bar(x + width/2, vs_baseline_10min, width, label='10分钟级', alpha=0.8, color='#6EE7B7')

ax.set_ylabel('相对基准的倍数', fontsize=12)
ax.set_xlabel('模型', fontsize=12)
ax.set_title('各模型 vs cluster_utils基准的性能差距', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='基准线 (1.0×)')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (v1, v10) in enumerate(zip(vs_baseline_1min, vs_baseline_10min)):
    if v1 < 50:
        ax.text(i - width/2, v1 + 2, f'{v1:.1f}×', ha='center', fontsize=8)
    else:
        ax.text(i - width/2, v1 + 10, f'{v1:.0f}×', ha='center', fontsize=8)

    if v10 < 50:
        ax.text(i + width/2, v10 + 2, f'{v10:.1f}×', ha='center', fontsize=8)
    else:
        ax.text(i + width/2, v10 + 10, f'{v10:.0f}×', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('benchmark_vs_baseline.png', dpi=150, bbox_inches='tight')
print("✅ 图表2已保存: benchmark_vs_baseline.png")

# 图3: 开发时间 vs 性能
fig, ax = plt.subplots(figsize=(10, 8))

dev_times = {
    'gemini': 8.7,
    'deepseek': 26.4,
    'codex-sol-2': 77.3,
    'claude-opus': 231.3
}

# 提取有开发时间数据的模型
dev_models = []
dev_time_list = []
perf_1min = []
perf_10min = []

for name in model_names:
    if name in dev_times:
        idx = model_names.index(name)
        dev_models.append(name)
        dev_time_list.append(dev_times[name])
        perf_1min.append(time_1min[idx])
        perf_10min.append(time_10min[idx])

# 散点图
ax.scatter(dev_time_list, perf_1min, s=200, alpha=0.6, label='1分钟级', color='#3B6DFF', edgecolors='black', linewidth=2)
ax.scatter(dev_time_list, perf_10min, s=200, alpha=0.6, label='10分钟级', color='#6EE7B7', edgecolors='black', linewidth=2)

# 添加标签
for i, name in enumerate(dev_models):
    ax.annotate(name, (dev_time_list[i], perf_1min[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    ax.annotate(name, (dev_time_list[i], perf_10min[i]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

ax.set_xlabel('开发时间 (分钟)', fontsize=12)
ax.set_ylabel('运行耗时 (秒)', fontsize=12)
ax.set_title('开发时间 vs 性能 (无明显相关性)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('dev_time_vs_performance.png', dpi=150, bbox_inches='tight')
print("✅ 图表3已保存: dev_time_vs_performance.png")

print("\n📊 所有图表生成完成！")
print("   - benchmark_comparison.png (性能对比+缩放分析)")
print("   - benchmark_vs_baseline.png (vs基准倍数)")
print("   - dev_time_vs_performance.png (开发时间vs性能)")
