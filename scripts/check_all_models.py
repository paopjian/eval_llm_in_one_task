#!/usr/bin/env python3
"""
简化版全模型性能测试
直接使用各模型已有的数据文件进行测试
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
from datetime import datetime

def find_all_models():
    """查找所有模型文件夹及其主执行脚本"""
    base_dir = Path(__file__).parent
    models = {}

    # 所有模型文件夹
    model_folders = [
        'claude-opus',
        'claude-opus-m',
        'codex-sol',
        'codex-sol-2',
        'codex-sol-m',
        'deepseek',
        'deepseek-pro',
        'gemini',
        'grok',
        'glm',
        'glm-5.3',
        'qwen'
    ]

    for folder in model_folders:
        model_dir = base_dir / folder
        if not model_dir.exists():
            continue

        # 查找该文件夹下的主要Python脚本
        py_files = list(model_dir.glob('*.py'))

        # 排除测试和工具脚本
        main_scripts = []
        for f in py_files:
            name = f.name.lower()
            if any(x in name for x in ['test_', 'util', 'tool', 'helper']):
                continue
            if any(x in name for x in ['eval', 'face', 'similar', 'main']):
                main_scripts.append(f)

        if main_scripts:
            # 优先选择final/main，否则选第一个
            script = None
            for s in main_scripts:
                if 'final' in s.name.lower() or 'main' in s.name.lower():
                    script = s
                    break
            if not script:
                script = main_scripts[0]

            models[folder] = {
                'name': folder,
                'dir': str(model_dir),
                'script': str(script),
                'script_name': script.name
            }

    return models

def check_data_file(model_dir):
    """检查模型目录下是否有数据文件"""
    data_files = list(Path(model_dir).glob('*.pkl'))
    if data_files:
        return str(data_files[0])
    return None

def run_simple_import_test(model_info):
    """简单的导入测试，看脚本是否能正常加载"""
    script_path = model_info['script']
    work_dir = model_info['dir']

    test_code = f"""
import sys
sys.path.insert(0, '{work_dir}')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", r'{script_path}')
    module = importlib.util.module_from_spec(spec)
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
"""

    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        return 'IMPORT_SUCCESS' in result.stdout
    except:
        return False

def main():
    print("="*70)
    print("全模型代码检查与统计")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 查找所有模型
    models = find_all_models()
    print(f"\n找到 {len(models)} 个模型文件夹\n")

    results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }

    # 统计每个模型
    print(f"{'模型名称':<20} {'主脚本':<30} {'数据文件':<15} {'可导入':<10}")
    print("-" * 80)

    for model_name in sorted(models.keys()):
        info = models[model_name]

        # 检查数据文件
        data_file = check_data_file(info['dir'])
        has_data = '✅' if data_file else '❌'

        # 检查脚本可导入性
        can_import = run_simple_import_test(info)
        import_status = '✅' if can_import else '❌'

        # 统计代码行数
        try:
            with open(info['script'], 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
        except:
            lines = 0

        print(f"{model_name:<20} {info['script_name']:<30} {has_data:<15} {import_status:<10}")

        results['models'][model_name] = {
            'script': info['script_name'],
            'script_path': info['script'],
            'has_data': data_file is not None,
            'data_file': data_file,
            'can_import': can_import,
            'code_lines': lines
        }

    # 保存结果
    output_file = 'models_inventory.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✅ 统计完成！详细信息已保存到: {output_file}")

    # 统计汇总
    total = len(models)
    with_data = sum(1 for m in results['models'].values() if m['has_data'])
    can_import = sum(1 for m in results['models'].values() if m['can_import'])

    print(f"\n📊 统计汇总:")
    print(f"   总模型数: {total}")
    print(f"   有数据文件: {with_data} ({with_data/total*100:.0f}%)")
    print(f"   可正常导入: {can_import} ({can_import/total*100:.0f}%)")
    print(f"{'='*70}\n")

    # 推荐可测试的模型
    print("✅ 可直接测试的模型 (有数据+可导入):")
    testable = [name for name, info in results['models'].items()
                if info['has_data'] and info['can_import']]
    for name in testable:
        print(f"   - {name}")

    if testable:
        print(f"\n💡 建议: 使用这些模型的代码创建统一benchmark测试")

    return results

if __name__ == '__main__':
    main()
