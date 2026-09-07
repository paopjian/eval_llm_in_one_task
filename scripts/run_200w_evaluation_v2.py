#!/usr/bin/env python3
"""
200万数据集 - 11个模型完整评估 v2
根据每个模型的参数格式正确调用
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 11个模型配置（使用正确的参数格式）
MODELS = {
    'claude-opus': {
        'script': 'eval_v5_final.py',
        'args': [],  # 硬编码在脚本中
        'description': 'Claude Opus'
    },
    'codex-sol': {
        'script': 'face_similarity_eval.py',
        'args': ['--input', '../test_data_200w.pkl', '--font', '../font/SourceHanSansSC-Normal.otf'],
        'description': 'GPT-4 Codex Solution'
    },
    'codex-sol-2': {
        'script': 'face_similarity_evaluator.py',
        'args': ['--input', '../test_data_200w.pkl', '--font-path', '../font/SourceHanSansSC-Normal.otf'],
        'description': 'GPT-4 Codex Solution 2'
    },
    'codex-sol-m': {
        'script': 'face_similarity_eval.py',
        'args': ['--input', '../test_data_200w.pkl'],
        'description': 'GPT-4 Codex Solution Mini'
    },
    'deepseek': {
        'script': 'run_eval.py',
        'args': [],  # 硬编码在脚本中
        'description': 'DeepSeek'
    },
    'deepseek-pro': {
        'script': 'step4_eval_optimized.py',
        'args': ['--data', '../test_data_200w.pkl'],
        'description': 'DeepSeek Pro'
    },
    'gemini': {
        'script': 'face_eval_system.py',
        'args': ['--data_path', '../test_data_200w.pkl', '--font_path', '../font/SourceHanSansSC-Normal.otf'],
        'description': 'Gemini'
    },
    'glm': {
        'script': 'eval_similar.py',
        'args': ['--pkl', '../test_data_200w.pkl'],
        'description': 'GLM-4-Flash'
    },
    'glm-pro': {
        'script': 'step3_multi_gpu.py',
        'args': ['--pkl', '../test_data_200w.pkl'],
        'description': 'GLM-5.3-Pro'
    },
    'grok': {
        'script': 'eval_similarity.py',
        'args': ['--pkl', '../test_data_200w.pkl'],
        'description': 'Grok'
    },
    'qwen': {
        'script': 'eval_v1_single_gpu.py',
        'args': ['--pkl', '../test_data_200w.pkl'],
        'description': 'Qwen'
    }
}

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
        cmd = ['python', config['script']] + config['args']
        print(f"执行命令: cd {model_name} && {' '.join(cmd)}")

        # 运行评估脚本
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=model_name  # 切换到模型目录
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ 测试完成，耗时: {elapsed:.2f}秒")
            print(f"输出:\n{result.stdout[-500:]}")  # 最后500字符

            return {
                'status': 'success',
                'time': elapsed,
                'stdout': result.stdout[-1000:]  # 保存最后1000字符
            }
        else:
            print(f"❌ 测试失败")
            print(f"错误:\n{result.stderr[-500:]}")
            return {
                'status': 'failed',
                'time': elapsed,
                'error': result.stderr[-1000:]
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
    print(f"200万数据集 - 11个模型完整评估 v2")
    print(f"测试数据: test_data_200w.pkl")
    print(f"模型数量: {len(MODELS)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # 检查测试数据
    if not Path('test_data_200w.pkl').exists():
        print(f"❌ 测试数据不存在: test_data_200w.pkl")
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
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'test_time': datetime.now().isoformat(),
                'test_data': 'test_data_200w.pkl',
                'results': all_results
            }, f, indent=2, ensure_ascii=False)

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
