#!/usr/bin/env python3
"""
全模型性能基准测试
统一测试所有12个模型生成的代码在相同数据集上的性能
"""

import os
import sys
import time
import json
import shutil
import subprocess
import pickle
from pathlib import Path
from datetime import datetime
import importlib.util

# 模型配置：模型名 -> (主脚本名, 数据加载函数, 主函数)
MODEL_CONFIGS = {
    'claude-opus': {
        'script': 'eval_v5_final.py',
        'description': 'Claude Opus high强度，3次提示，231.3分钟'
    },
    'claude-opus-m': {
        'script': 'eval_similarity_final.py',
        'description': 'Claude Opus medium强度'
    },
    'codex-sol': {
        'script': 'face_similarity_eval.py',
        'description': 'Claude Opus max强度（首测读答案，不计入）'
    },
    'codex-sol-2': {
        'script': 'face_similarity_evaluator.py',
        'description': 'Claude Opus max强度重测，77.3分钟'
    },
    'codex-sol-m': {
        'script': 'face_similarity_eval.py',
        'description': 'Claude Opus medium强度'
    },
    'deepseek': {
        'script': 'run_eval.py',
        'description': 'DeepSeek v4-flash，26.4分钟'
    },
    'deepseek-pro': {
        'script': 'step4_eval_optimized.py',
        'description': 'DeepSeek v4-pro'
    },
    'gemini': {
        'script': 'face_eval_system.py',
        'description': 'Claude Opus场景测试，8.7分钟'
    },
    'glm': {
        'script': 'eval_similar.py',
        'description': 'GLM-5.3-flash'
    },
    'glm-pro': {
        'script': 'step3_multi_gpu.py',
        'description': 'GLM-5.3-pro'
    },
    'grok': {
        'script': 'eval_similarity.py',
        'description': 'Claude Opus场景测试'
    },
    'qwen': {
        'script': 'eval_v1_single_gpu.py',
        'description': 'Qwen 3.8（欠费未完成）'
    }
}

def load_baseline_results():
    """加载cluster_utils基准结果"""
    baseline_file = 'cluster_utils_benchmark_results.json'
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            return json.load(f)
    return None

def prepare_test_data(model_dir, data_file):
    """将测试数据复制到模型目录，统一命名为s4_0618_enhance.pkl"""
    # 所有模型都硬编码了数据文件名为s4_0618_enhance.pkl
    dest = os.path.join(model_dir, 's4_0618_enhance.pkl')
    shutil.copy2(data_file, dest)
    return dest

def run_model_test(model_name, config, data_file, timeout=600):
    """运行单个模型测试"""
    print(f"\n{'='*70}")
    print(f"测试模型: {model_name}")
    print(f"描述: {config['description']}")
    print(f"脚本: {config['script']}")
    print(f"数据: {os.path.basename(data_file)}")
    print(f"{'='*70}")

    model_dir = Path(__file__).parent / model_name
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return {'status': 'missing', 'error': '目录不存在'}

    script_path = model_dir / config['script']
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return {'status': 'missing', 'error': '脚本不存在'}

    # 准备数据文件
    try:
        local_data = prepare_test_data(model_dir, data_file)
        print(f"✅ 数据文件已复制到: {local_data}")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return {'status': 'failed', 'error': f'数据准备失败: {e}'}

    # 运行测试
    print(f"\n开始执行...")
    start_time = time.time()

    try:
        # 使用subprocess运行，不传参数（数据文件已复制为固定名称）
        result = subprocess.run(
            [sys.executable, config['script']],
            cwd=str(model_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ 测试成功")
            print(f"⏱️  耗时: {elapsed:.2f}秒")

            # 尝试从输出中提取性能信息
            output_lines = result.stdout.split('\n')
            perf_info = {}
            for line in output_lines[-20:]:  # 只看最后20行
                if '耗时' in line or 'time' in line.lower():
                    perf_info['output'] = line.strip()
                    break

            return {
                'status': 'success',
                'time': elapsed,
                'output_sample': '\n'.join(output_lines[-10:]),
                **perf_info
            }
        else:
            print(f"❌ 测试失败 (返回码: {result.returncode})")
            print(f"错误输出:\n{result.stderr[:500]}")

            return {
                'status': 'failed',
                'time': elapsed,
                'error': result.stderr[:200],
                'returncode': result.returncode
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ 测试超时 (>{timeout}秒)")
        return {
            'status': 'timeout',
            'time': elapsed,
            'timeout': timeout
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 执行异常: {e}")
        return {
            'status': 'error',
            'time': elapsed,
            'error': str(e)
        }

def main():
    print("="*70)
    print("全模型性能基准测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 加载基准结果
    baseline = load_baseline_results()
    if baseline:
        print("✅ 加载cluster_utils基准结果:")
        for dataset, result in baseline.get('results', {}).items():
            if result['status'] == 'success':
                print(f"   {dataset}: {result['total_time']:.2f}秒, {result['throughput']:.2f}B对/秒")
    else:
        print("⚠️  未找到cluster_utils基准结果")

    print()

    # 测试数据集
    datasets = [
        ('test_data_1min.pkl', '1分钟级', 300),   # 5分钟超时
        ('test_data_10min.pkl', '10分钟级', 1200)  # 20分钟超时
    ]

    results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'baseline': baseline,
        'models': {}
    }

    # 第一轮：1分钟级测试（所有模型）
    print("\n" + "="*70)
    print("第一轮：1分钟级数据集测试 (75K样本, 2.81B对)")
    print("="*70)

    data_file, dataset_name, timeout = datasets[0]
    if not os.path.exists(data_file):
        print(f"❌ 测试数据不存在: {data_file}")
        return

    for model_name in sorted(MODEL_CONFIGS.keys()):
        config = MODEL_CONFIGS[model_name]

        if model_name not in results['models']:
            results['models'][model_name] = {
                'description': config['description'],
                'tests': {}
            }

        result = run_model_test(model_name, config, data_file, timeout)
        results['models'][model_name]['tests'][dataset_name] = result

        # 保存中间结果
        with open('all_models_test_results_progress.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 第二轮：10分钟级测试（仅测试1分钟级成功的模型）
    print("\n" + "="*70)
    print("第二轮：10分钟级数据集测试 (200K样本, 20B对)")
    print("="*70)

    data_file, dataset_name, timeout = datasets[1]
    if not os.path.exists(data_file):
        print(f"❌ 测试数据不存在: {data_file}")
    else:
        for model_name in sorted(MODEL_CONFIGS.keys()):
            # 检查1分钟级是否成功
            if results['models'][model_name]['tests']['1分钟级']['status'] != 'success':
                print(f"\n⚠️  跳过 {model_name} (1分钟级测试未成功)")
                continue

            config = MODEL_CONFIGS[model_name]
            result = run_model_test(model_name, config, data_file, timeout)
            results['models'][model_name]['tests'][dataset_name] = result

            # 保存中间结果
            with open('all_models_test_results_progress.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存最终结果
    output_file = f"all_models_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成报告
    print("\n" + "="*70)
    print("测试完成！生成报告...")
    print("="*70)

    generate_report(results, baseline)

    print(f"\n详细结果已保存到: {output_file}")

def generate_report(results, baseline):
    """生成测试报告"""
    print("\n" + "="*70)
    print("测试结果摘要")
    print("="*70)

    if baseline:
        print("\n【基准: cluster_utils】")
        for dataset, result in baseline.get('results', {}).items():
            if result['status'] == 'success':
                print(f"  {dataset}: {result['total_time']:.2f}秒, {result['throughput']:.2f}B对/秒")

    for dataset_name in ['1分钟级', '10分钟级']:
        print(f"\n【{dataset_name}数据集】")
        print(f"{'模型':<20} {'状态':<10} {'耗时':<15} {'vs基准':<15}")
        print("-" * 65)

        # 收集并排序结果
        model_results = []
        for model_name, model_data in results['models'].items():
            if dataset_name in model_data['tests']:
                test_result = model_data['tests'][dataset_name]
                model_results.append((model_name, test_result))

        # 按时间排序
        model_results.sort(key=lambda x: x[1].get('time', float('inf')))

        baseline_time = None
        if baseline and dataset_name in baseline.get('results', {}):
            baseline_time = baseline['results'][dataset_name].get('total_time')

        for model_name, test_result in model_results:
            status = test_result['status']
            elapsed = test_result.get('time', 0)

            status_icons = {
                'success': '✅',
                'failed': '❌',
                'timeout': '⏰',
                'error': '⚠️',
                'missing': '❓'
            }
            status_icon = status_icons.get(status, '?')

            if elapsed < 60:
                time_str = f"{elapsed:.2f}秒"
            else:
                time_str = f"{elapsed/60:.1f}分钟"

            vs_baseline = '-'
            if status == 'success' and baseline_time:
                ratio = elapsed / baseline_time
                vs_baseline = f"{ratio:.2f}×"
                if ratio < 1.0:
                    vs_baseline = f"🚀 {vs_baseline}"

            print(f"{model_name:<20} {status_icon} {status:<8} {time_str:<15} {vs_baseline:<15}")

if __name__ == '__main__':
    main()
