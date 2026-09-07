#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冒烟测试/参考校验：在 N 切片上用朴素 GPU 实现直接统计上三角直方图，
与统一框架跑出的结果逐 bin 对比（用于验证 core 的正确性，不参与正式评估）。
用法: python benchmark/smoke_check.py --model glm --max-n 25000 [--models a,b]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from benchmark import common  # noqa: E402


def reference_hists(feats, ids, chunk=4096):
    """朴素参考：GPU 分块计算，掩码只保留全局索引 a<b 的对（严格上三角），一次覆盖全部对"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda:0'
    F = torch.from_numpy(np.ascontiguousarray(feats)).to(dev)
    I = torch.from_numpy(np.ascontiguousarray(ids)).to(dev)
    N = len(ids)
    bins = common.BINS
    ha = torch.zeros(bins, dtype=torch.int64, device=dev)
    hp = torch.zeros(bins, dtype=torch.int64, device=dev)
    nrow = (N + chunk - 1) // chunk
    with torch.no_grad():
        for i in range(nrow):
            r0, r1 = i * chunk, min((i + 1) * chunk, N)
            A = F[r0:r1]
            rows = torch.arange(r0, r1, device=dev)
            for j in range(i, nrow):
                c0, c1 = j * chunk, min((j + 1) * chunk, N)
                C = F[c0:c1]
                sim = A @ C.t()                       # (b1,b2)
                valid = rows[:, None] < torch.arange(c0, c1, device=dev)[None, :]
                sim.clamp_(common.LO, common.HI)
                sv = sim[valid]
                if sv.numel():
                    ha += torch.histc(sv, bins=bins, min=common.LO,
                                      max=common.HI).round().to(torch.int64)
                pv = sim[valid & (I[r0:r1, None] == I[None, c0:c1])]
                if pv.numel():
                    hp += torch.histc(pv, bins=bins, min=common.LO,
                                      max=common.HI).round().to(torch.int64)
    return hp.cpu().numpy(), ha.cpu().numpy() - hp.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='glm,baseline')
    ap.add_argument('--data', default='test_data_10min.pkl')
    ap.add_argument('--max-n', type=int, default=25000)
    args = ap.parse_args()

    feats, ids = common.load_data(os.path.join(REPO_ROOT, args.data))
    feats, ids = feats[:args.max_n], ids[:args.max_n]
    total, pos, neg = common.pair_stats(ids)
    print(f'切片 N={len(ids)} pairs: total={total} pos={pos} neg={neg}')

    print('计算参考直方图（朴素 GPU 全量）...')
    ref_pos, ref_neg = reference_hists(feats, ids)
    ok_ref, det = common.validate_hists(ref_pos, ref_neg, ids)
    print('参考直方图自校验:', ok_ref, det['issues'] or 'OK')

    for model in [m.strip() for m in args.models.split(',') if m.strip()]:
        print(f'\n--- {model} ---')
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'res.json')
            cmd = [sys.executable, os.path.join(REPO_ROOT, 'benchmark', 'run_one.py'),
                   '--model', model, '--data', os.path.join(REPO_ROOT, args.data),
                   '--gpus', '0,1,2,3,4,5,6', '--workdir', td, '--out', out,
                   '--max-n', str(args.max_n)]
            t0 = time.time()
            subprocess.run(cmd, cwd=REPO_ROOT, check=False)
            wall = time.time() - t0
            with open(out) as f:
                res = json.load(f)
            print(f'  status={res.get("status")} reason={res.get("reason", "-")} '
                  f'core={res.get("core_s")}s wall={wall:.1f}s')
            if res.get('status') == 'success':
                v = res['validation']
                print(f'  校验: pos {v["got_pos"]:,}/{v["pos"]:,} '
                      f'neg {v["got_neg"]:,}/{v["neg"]:,} issues={v["issues"] or "无"}')
                # 与参考直方图对比
                import importlib
                coremod = importlib.import_module(
                    f'benchmark.cores.{model.replace("-", "_")}')
                # 直接用 core 计算结果文件中的 metrics 与参考 metrics 对比
                ref_metrics = common.compute_metrics(ref_pos, ref_neg)
                got_metrics = res['metrics']
                print('  指标对比(FPIR 阈值 TPIR):')
                for a, b in zip(got_metrics, ref_metrics):
                    diff = abs(a['threshold'] - b['threshold'])
                    mark = '✓' if diff <= 2 * common.W else '✗'
                    print(f'    {a["fpir"]:>7.0e} core t={a["threshold"]:.6f} tp={a["tpir"]*100:.4f}%'
                          f' | ref t={b["threshold"]:.6f} tp={b["tpir"]*100:.4f}% {mark}')
            else:
                print('  traceback tail:', (res.get('traceback') or '')[-500:])


if __name__ == '__main__':
    main()
