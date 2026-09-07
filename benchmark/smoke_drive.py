#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 冒烟驱动器：直接 GPU 多卡实测（用户要求不用 CPU 逻辑测试）。
对每个模型在 N 切片上运行 run_one（经 run_eval.run_model 监督：超时/内存上限/
进程组整树击杀），并与"朴素 GPU 参考直方图"逐项对比指标与计数。

用法:
    python benchmark/smoke_drive.py                     # 全部模型, N=25000, 单模型300s
    python benchmark/smoke_drive.py --models glm,gemini --max-n 25000 --timeout 300
"""
import argparse
import importlib.util
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from benchmark import common  # noqa: E402
from benchmark.run_eval import run_model  # noqa: E402
from benchmark.cores import CORE_NAMES  # noqa: E402
from benchmark.smoke_check import reference_hists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--data', default='test_data_10min.pkl')
    ap.add_argument('--max-n', type=int, default=25000)
    ap.add_argument('--timeout', type=float, default=300)
    ap.add_argument('--mem-cap-gb', type=float, default=400)
    ap.add_argument('--outdir', default='')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()] or CORE_NAMES
    data_path = os.path.join(REPO_ROOT, args.data)
    outdir = args.outdir or os.path.join('logs', 'unified_eval', 'smoke_gpu')
    os.makedirs(outdir, exist_ok=True)

    feats, ids = common.load_data(data_path)
    feats, ids = feats[:args.max_n], ids[:args.max_n]
    total, pos, neg = common.pair_stats(ids)
    print(f'切片 N={len(ids)}: total={total:,} pos={pos:,} neg={neg:,}', flush=True)

    t0 = time.time()
    print('计算朴素 GPU 参考直方图（cuda:0）...', flush=True)
    ref_pos, ref_neg = reference_hists(feats, ids)
    print(f'  参考耗时 {time.time()-t0:.0f}s | 自校验: '
          f'{common.validate_hists(ref_pos, ref_neg, ids)[0]}', flush=True)
    ref_metrics = common.compute_metrics(ref_pos, ref_neg)

    stage_cfg = {'label': f'smokeN{args.max_n}', 'timeout_s': args.timeout,
                 'mem_cap_gb': args.mem_cap_gb}
    gpus = list(range(7))
    logf = open(os.path.join(outdir, 'driver.log'), 'w')
    results = []
    for model in models:
        t1 = time.time()
        res = run_model(stage_cfg, model, data_path, outdir, gpus, logf,
                        max_n=args.max_n)
        wall = time.time() - t1
        verdict = '?'
        detail = ''
        if res.get('status') == 'success':
            v = res['validation']
            m = res['metrics']
            diff_thr = max(abs(a['threshold'] - b['threshold'])
                           for a, b in zip(m, ref_metrics))
            diff_tp = max(abs(a['tpir'] - b['tpir'])
                          for a, b in zip(m, ref_metrics))
            ok_sum = v['got_pos'] == v['pos'] and v['got_neg'] == v['neg']
            # fp16/tf32 类方法有固有量化：计数精确即合格，指标与 fp32 参考允许
            # 阈值差 <=10 bins、TPIR 差 <=1e-3（量化效应，见各 core 的 precision 说明）
            verdict = 'PASS' if (ok_sum and diff_thr <= 10 * common.W
                                 and diff_tp < 1e-3) else 'FAIL'
            detail = (f'pos {v["got_pos"]:,}/{v["pos"]:,} '
                      f'neg {v["got_neg"]:,}/{v["neg"]:,} '
                      f'阈值最大差 {diff_thr:.2e} TPIR最大差 {diff_tp:.2e}')
        else:
            verdict = 'FAIL'
            detail = res.get('reason', res.get('status', ''))
            if res.get('traceback'):
                detail += ' | ' + res['traceback'].strip().splitlines()[-1]
        line = (f'{model:<14} {verdict:<6} core={res.get("core_s", "-")}s '
                f'wall={wall:6.1f}s peakRSS={res.get("peak_rss_gb", 0):.1f}GB  {detail}')
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
        results.append({'model': model, 'verdict': verdict, 'res': res})
    logf.close()

    with open(os.path.join(outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果: {outdir}/summary.json')


if __name__ == '__main__':
    main()
