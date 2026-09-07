#!/usr/bin/env python3
"""
200万数据集 - 11个模型完整评估
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 11个模型配置（排除claude-opus-m）
MODELS = {
    'claude-opus': {
        'script': 'eval_v5_final.py',
        'description': 'Claude Opus'
    },
    'codex-sol': {
        'script': 'face_similarity_eval.py',
        'description': 'GPT-4 Codex Solution'
    },
    'codex-sol-2': {
        'script': 'face_similarity_evaluator.py',
        'description': 'GPT-4 Codex Solution 2'
    },
    'codex-sol-m': {
        'script': 'face_similarity_eval.py',
        'description': 'GPT-4 Codex Solution Mini'
    },
    'deepseek': {
        'script': 'run_eval.py',
        'description': 'DeepSeek'
    },
    'deepseek-pro': {
        'script': 'step4_eval_optimized.py',
        'description': 'DeepSeek Pro'
    },
    'gemini': {
        'script': 'face_eval_system.py',
        'description': 'Gemini'
    },
    'glm': {
        'script': 'eval_similar.py',
        'description': 'GLM-4-Flash'
    },
    'glm-pro': {
        'script': 'step3_multi_gpu.py',
        'description': 'GLM-5.3-Pro'
    },
    'grok': {
        'script': 'eval_similarity.py',
        'description': 'Grok'
    },
    'qwen': {
        'script': 'eval_v1_single_gpu.py',
        'description': 'Qwen'
    }
}

TEST_DATA = 'test_data_200w.pkl'
RESULTS_FILE = 'results_200w_evaluation.json'

def run_model_test(model_name, config):
    """运行单个模型测试"""
    print(f"\n{'='*80}")
    print(f"开始测试: {config['description']} ({model_name})")
    print(f"{'='*80}")

    script_path = Path(model_name) / config['script']
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return None

    start_time = time.time()

    try:
        # 运行评估脚本
        result = subprocess.run(
            ['python', str(script_path), TEST_DATA],
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=str(Path.cwd())
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ 测试完成，耗时: {elapsed:.2f}秒")
            print(f"输出:\n{result.stdout}")

            # 尝试解析结果
            try:
                # 查找结果文件
                result_files = list(Path(model_name).glob('*result*.json'))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        test_result = json.load(f)
                    return {
                        'status': 'success',
                        'time': elapsed,
                        'result': test_result
                    }
            except:
                pass

            return {
                'status': 'success',
                'time': elapsed,
                'stdout': result.stdout
            }
        else:
            print(f"❌ 测试失败")
            print(f"错误:\n{result.stderr}")
            return {
                'status': 'failed',
                'time': elapsed,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        print(f"❌ 测试超时（>600秒）")
        return {
            'status': 'timeout',
            'time': 600
        }
    except Exception as e:
        print(f"❌ 运行异常: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def main():
    print(f"{'='*80}")
    print(f"200万数据集 - 11个模型完整评估")
    print(f"测试数据: {TEST_DATA}")
    print(f"模型数量: {len(MODELS)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # 检查测试数据
    if not Path(TEST_DATA).exists():
        print(f"❌ 测试数据不存在: {TEST_DATA}")
        sys.exit(1)

    all_results = {}

    # 逐个运行模型
    for i, (model_name, config) in enumerate(MODELS.items(), 1):
        print(f"\n进度: {i}/{len(MODELS)}")
        result = run_model_test(model_name, config)
        all_results[model_name] = {
            'description': config['description'],
            'result': result
        }

        # 保存中间结果
        with open(RESULTS_FILE, 'w') as f:
            json.dump({
                'test_time': datetime.now().isoformat(),
                'test_data': TEST_DATA,
                'results': all_results
            }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"评估完成！")
    print(f"结果已保存到: {RESULTS_FILE}")
    print(f"{'='*80}")

    # 统计结果
    success = sum(1 for r in all_results.values() if r['result'] and r['result'].get('status') == 'success')
    failed = len(all_results) - success
    print(f"\n✅ 成功: {success}")
    print(f"❌ 失败: {failed}")

if __name__ == '__main__':
    main()
