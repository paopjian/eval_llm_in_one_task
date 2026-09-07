#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 快速启动脚本
用法: python run_evaluation.py
"""
import subprocess
import sys

def main():
    print("=" * 80)
    print("启动人脸特征相似度评估系统")
    print("=" * 80)
    print()
    print("配置:")
    print("  - 使用GPU: 0,1,2,3,4,5,6 (7卡并行)")
    print("  - 分块大小: 2000")
    print("  - 输出文件: evaluation_results.png")
    print()
    print("开始计算...")
    print("=" * 80)
    print()

    cmd = [
        "/root/miniconda3/envs/cvlface/bin/python",
        "eval_similarity_final.py",
        "--devices", "0,1,2,3,4,5,6",
        "--chunk_size", "2000",
        "--output", "evaluation_results.png"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("=" * 80)
        print("✅ 评估完成！")
        print("=" * 80)
        print()
        print("输出文件:")
        print("  - evaluation_results.png (评估图表)")
        print()
        print("查看结果:")
        print("  详细报告请查看 README.md")
    else:
        print()
        print("❌ 评估失败，请查看错误信息")
        sys.exit(1)

if __name__ == '__main__':
    main()
