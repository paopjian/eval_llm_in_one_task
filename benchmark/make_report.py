#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总统一评估结果并输出 Markdown 报告。

用法:
    python benchmark/make_report.py --outdir logs/unified_eval/run_200k_final \
        [--md 03-统一基准测试报告/04-统一评估结果.md]
报告内容:
    * 200K 阶段表（vs 基准，吞吐率=总对数/墙钟，60s 筛选结果）
    * 2M 阶段表（30min 超时/400G 上限，只有 200K 合格者）
    * 直方图计数校验摘要（strict_ok / delta）
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stage(outdir, stage):
    sp = os.path.join(outdir, stage, 'summary.json')
    if not os.path.exists(sp):
        return None
    with open(sp) as f:
        return json.load(f)


def row_line(r):
    d = ''
    if r.get('validation'):
        v = r['validation']
        d = ('✓' if v.get('strict_ok') else
             f"≈Δpos{v.get('delta_pos', 0):+d}/Δneg{v.get('delta_neg', 0):+d}")
    else:
        d = r.get('reason', r.get('status', ''))[:40]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--md', default='')
    args = ap.parse_args()

    s200 = load_stage(args.outdir, '200k')
    s2m = load_stage(args.outdir, '2m')
    if s200 is None and s2m is None:
        print(f'没有找到结果: {args.outdir}')
        sys.exit(1)

    pairs = {'200k': 19_999_900_000, '2m': 1_999_999_000_000}
    lines = []
    lines.append('# 统一评估结果（重新评估）\n')
    lines.append(f'- 结果目录: `{args.outdir}`\n')

    for stage, label, n_pairs in (('200k', '200K', pairs['200k']),
                                  ('2m', '2M', pairs['2m'])):
        s = s200 if stage == '200k' else s2m
        if s is None:
            lines.append(f'\n## {label} 阶段：未执行\n')
            continue
        cfg = s['config']
        lines.append(f'\n## {label} 阶段'
                     f'（timeout={cfg["timeout_s"]}s，mem_cap={cfg["mem_cap_gb"]}GB）\n')
        res = s['results']
        res_sorted = sorted(res, key=lambda r: r.get('wall_s', 1e18))
        bl = next((r for r in res if r['model'] == 'baseline'), None)
        base_wall = bl.get('wall_s') if bl else None
        lines.append('| 排名 | 模型 | 状态 | 耗时(s) | 核心计算(s) | 吞吐率 | vs基准 | 峰值RSS | 计数校验 |')
        lines.append('|------|------|------|---------|------------|--------|--------|---------|----------|')
        for i, r in enumerate(res_sorted, 1):
            ok = r['status'] == 'success'
            thr = (f'{n_pairs / r["wall_s"] / 1e9:.2f}G对/s' if ok and r.get('wall_s') else '-')
            vs = (f'{r["wall_s"] / base_wall:.2f}×' if ok and base_wall and r['model'] != 'baseline' else
                  ('基准' if r['model'] == 'baseline' and ok else '-'))
            lines.append(f'| {i} | {r["model"]} | {"✅" if ok else "❌"} '
                         f'| {r.get("wall_s", "-")} | {r.get("core_s", "-")} '
                         f'| {thr} | {vs} '
                         f'| {r.get("peak_rss_gb", "-")}GB | {row_line(r)} |')
        lines.append('')

    # 汇总结论
    if s200 and s2m:
        lines.append('## 结论\n')
        lines.append('**200K 阶段（60s 筛选）**：')
        q = [r['model'] for r in s200['results']
             if r['status'] == 'success' and r.get('wall_s', 1e9) <= 60]
        lines.append('- 合格进入 2M：' + ', '.join(q))
        f = [r for r in s200['results'] if r['status'] != 'success']
        if f:
            lines.append('- 不合格/失败：' + '; '.join(
                f"{r['model']}({r.get('reason', r['status'])[:60]})" for r in f))
        lines.append('')
        lines.append('**2M 阶段（30min / 400G）**：')
        for r in sorted(s2m['results'], key=lambda r: r.get('wall_s', 1e18)):
            st = '✅' if r['status'] == 'success' else '❌'
            extra = f"{r.get('wall_s', '-')}s" if r['status'] == 'success' \
                else r.get('reason', r['status'])[:80]
            lines.append(f'- {st} {r["model"]}: {extra}')
    txt = '\n'.join(lines) + '\n'

    if args.md:
        md = os.path.join(REPO_ROOT, args.md) if not os.path.isabs(args.md) else args.md
        os.makedirs(os.path.dirname(md), exist_ok=True)
        with open(md, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f'报告已写入: {md}')
    else:
        print(txt)


if __name__ == '__main__':
    main()
