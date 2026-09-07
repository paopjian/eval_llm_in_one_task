#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_utils v6 实验：对比 v6 各配置 vs v5 vs glm-线程版。

覆盖:
  1. v6 fp32（extras 关）         —— 快速路径
  2. v6 fp32 + 进度条             —— 进度条开销
  3. v6 fp32 + 进度条 + 样本对收集 —— 附加功能开销
  4. v6 fp32 row_cache 关         —— 行块缓存收益
  5. v6 fp16 / v6 tf32            —— 精度档速度与误差
  6. v5（基准, 20M bins）         —— 参照
  7. glm-线程版（特征常驻显存）     —— 显存充足时的速度上限参照

用法:
    python benchmark/experiments/test_cluster_utils_v6.py --data test_data_10min.pkl   # 200K
    python benchmark/experiments/test_cluster_utils_v6.py --data test_data_200w.pkl   # 2M
输出: logs/unified_eval/cluster_utils_v6/summary.json
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from benchmark import common  # noqa: E402
from benchmark.experiments import glm_threaded as glm_thr  # noqa: E402

# cluster_utils.py 顶层有 cv2/onnxruntime 等重依赖，按 cores/baseline.py 同款方式加载
_cu_path = os.path.join(REPO_ROOT, 'benchmark', 'cluster_utils.py')
_cu_spec = importlib.util.spec_from_file_location('cluster_utils', _cu_path)
cu = importlib.util.module_from_spec(_cu_spec)
_cu_spec.loader.exec_module(cu)

OUT = os.path.join('logs', 'unified_eval', 'cluster_utils_v6')


def _proc_rss(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        return 0.0


def _child_pids(pid):
    out = []
    try:
        for p in os.listdir('/proc'):
            if not p.isdigit() or p == str(pid):
                continue
            try:
                with open(f'/proc/{p}/stat') as f:
                    rest = f.read().rsplit(')', 1)[1].split()
                if int(rest[1]) == pid:
                    out.append(int(p))
            except Exception:
                pass
    except Exception:
        pass
    return out


def tree_rss(pid):
    total = _proc_rss(pid)
    for c in _child_pids(pid):
        total += tree_rss(c)
    return total


class PeakRSS:
    def __init__(self, interval=0.15):
        self.interval = interval
        self.peak = 0.0
        self._stop = False
        self._th = None

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        pid = os.getpid()
        while not self._stop:
            try:
                r = tree_rss(pid)
                if r > self.peak:
                    self.peak = r
            except Exception:
                pass
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        self._th.join(timeout=3)
        return False


def _finish(rec, pos_hist, neg_hist, ids):
    total = len(ids) * (len(ids) - 1) // 2
    rec['throughput_Gpairs_s'] = round(total / rec['wall_s'] / 1e9, 2)
    ok, vd = common.validate_hists(pos_hist, neg_hist, ids)
    rec['strict_ok'] = vd['strict_ok']
    rec['status'] = 'success' if ok else 'invalid'
    m = common.compute_metrics(pos_hist, neg_hist)
    rec['metrics'] = m
    return rec, m


def run_v6(name, feats, ids, gpus, **kw):
    rec = {'name': name}
    print(f'===== {name} =====', flush=True)
    with PeakRSS() as pm:
        t0 = time.perf_counter()
        ret = cu.get_sim_matrix_large_scale_v6(
            query_feats_list=feats, query_ids=ids, num_gpus=len(gpus), **kw)
        wall = time.perf_counter() - t0
    rec['wall_s'] = round(wall, 3)
    rec['peak_rss_gb'] = round(pm.peak, 2)
    if len(ret) > 2:
        pos_hist, neg_hist, pairs = ret[0], ret[1], ret[2]
        rec['collected'] = len(pairs)
    else:
        pos_hist, neg_hist = ret[0], ret[1]
    rec, m = _finish(rec, pos_hist, neg_hist, ids)
    print(f"  {rec['status']} wall={wall:.2f}s peakRSS={rec['peak_rss_gb']}GB "
          f"strict={rec['strict_ok']} collected={rec.get('collected', '-')}", flush=True)
    return rec, pos_hist, neg_hist


def run_v5(name, feats, ids, gpus):
    rec = {'name': name}
    print(f'===== {name} =====', flush=True)
    with PeakRSS() as pm:
        t0 = time.perf_counter()
        pos20, neg20 = cu.get_sim_matrix_large_scale_v5(
            query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
            block_size=2048 * 5, hist_bins=20_000_000, hist_range=(-1.0, 1.0),
            collect_pairs_config=None, memory_mode='low_memory',
            show_progress=False, precision='fp32')
        wall = time.perf_counter() - t0
    rec['wall_s'] = round(wall, 3)
    rec['peak_rss_gb'] = round(pm.peak, 2)
    pos_hist = common.rebin(np.asarray(pos20, dtype=np.int64), 100)
    neg_hist = common.rebin(np.asarray(neg20, dtype=np.int64), 100)
    rec, m = _finish(rec, pos_hist, neg_hist, ids)
    print(f"  {rec['status']} wall={wall:.2f}s peakRSS={rec['peak_rss_gb']}GB "
          f"strict={rec['strict_ok']}", flush=True)
    return rec, pos_hist, neg_hist


def run_glm_threaded(name, feats, ids, gpus):
    rec = {'name': name}
    print(f'===== {name} =====', flush=True)
    workdir = os.path.join(OUT, 'work', 'glm_threads')
    with PeakRSS() as pm:
        t0 = time.perf_counter()
        pos_hist, neg_hist, meta = glm_thr.compute(feats, ids, gpus, workdir)
        wall = time.perf_counter() - t0
    rec['wall_s'] = round(wall, 3)
    rec['peak_rss_gb'] = round(pm.peak, 2)
    rec, m = _finish(rec, pos_hist, neg_hist, ids)
    print(f"  {rec['status']} wall={wall:.2f}s peakRSS={rec['peak_rss_gb']}GB "
          f"strict={rec['strict_ok']}", flush=True)
    return rec, pos_hist, neg_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='test_data_10min.pkl')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',') if g != '']

    feats, ids = common.load_data(os.path.join(REPO_ROOT, args.data))
    total, pos, neg = common.pair_stats(ids)
    print(f'\nN={len(ids):,} pairs total={total:,} pos={pos:,} neg={neg:,}', flush=True)

    # 预热：初始化各卡 CUDA 上下文 + JIT 编译 GEMM/histc kernel，消除首个配置的冷启动偏差
    print('预热各卡 CUDA 上下文...', flush=True)
    for g in gpus:
        with torch.cuda.device(g):
            a = torch.randn(16384, 512, device=f'cuda:{g}')
            b = torch.randn(16384, 512, device=f'cuda:{g}')
            _ = a @ b.t()
            _ = torch.histc(torch.randn(4096, device=f'cuda:{g}'),
                            bins=200000, min=-1.0, max=1.0)
    torch.cuda.synchronize()

    collect_cfg = {'sample_type': 'neg', 'threshold_mode': 'above', 'threshold': 0.5}
    results = []

    def go(name, fn, *a, **kw):
        try:
            rec, ph, nh = fn(name, *a, **kw)
            results.append(rec)
            return rec, ph, nh
        except Exception as e:  # noqa: BLE001
            rec = {'name': name, 'status': 'error', 'reason': f'{type(e).__name__}: {e}'}
            results.append(rec)
            print(f'  error: {rec["reason"]}', flush=True)
            return rec, None, None

    # 基准参照
    go('glm-threaded(特征常驻)', run_glm_threaded, feats, ids, gpus)
    go('v5-fp32(20M bins)', run_v5, feats, ids, gpus)

    # v6 fp32 快速路径（作为 fp32 锚点）
    r_anchor, pos_a, neg_a = go(
        'v6-fp32', run_v6, feats, ids, gpus,
        block_size=16384, hist_bins=200_000, precision='fp32',
        memory_mode='low_memory', row_cache=True, show_progress=False)

    # 附加功能开销
    go('v6-fp32+进度条', run_v6, feats, ids, gpus,
       block_size=16384, hist_bins=200_000, precision='fp32',
       memory_mode='low_memory', row_cache=True, show_progress=True)
    go('v6-fp32+进度条+样本对收集', run_v6, feats, ids, gpus,
       block_size=16384, hist_bins=200_000, precision='fp32',
       memory_mode='low_memory', row_cache=True, show_progress=True,
       collect_pairs_config=collect_cfg)

    # 行块缓存收益
    go('v6-fp32-row_cache关', run_v6, feats, ids, gpus,
       block_size=16384, hist_bins=200_000, precision='fp32',
       memory_mode='low_memory', row_cache=False, show_progress=False)

    # 精度档
    _, pos_fp16, neg_fp16 = go('v6-fp16', run_v6, feats, ids, gpus,
                               block_size=16384, hist_bins=200_000, precision='fp16',
                               memory_mode='low_memory', row_cache=True, show_progress=False)
    _, pos_tf32, neg_tf32 = go('v6-tf32', run_v6, feats, ids, gpus,
                               block_size=16384, hist_bins=200_000, precision='tf32',
                               memory_mode='low_memory', row_cache=True, show_progress=False)

    # 精度档相对 fp32 的直方图/阈值差异
    def hist_diff(ph, nh, pa, na):
        return int(abs(ph.astype('int64') - pa.astype('int64')).sum()) + \
               int(abs(nh.astype('int64') - na.astype('int64')).sum())

    if pos_fp16 is not None:
        d = hist_diff(pos_fp16, neg_fp16, pos_a, neg_a)
        m16 = common.compute_metrics(pos_fp16, neg_fp16)
        ma = r_anchor.get('metrics')
        max_bin16 = max(abs(a['threshold'] - b['threshold']) / common.W
                        for a, b in zip(m16, ma))
        print(f'[精度] fp16 vs fp32: 直方图总差 {d}  阈值最大差 {max_bin16:.1f} bin')
        for r in results:
            if r['name'] == 'v6-fp16':
                r['vs_fp32_hist_diff'] = d
                r['vs_fp32_max_bin'] = round(max_bin16, 2)
    if pos_tf32 is not None:
        d = hist_diff(pos_tf32, neg_tf32, pos_a, neg_a)
        m32 = common.compute_metrics(pos_tf32, neg_tf32)
        ma = r_anchor.get('metrics')
        max_bin32 = max(abs(a['threshold'] - b['threshold']) / common.W
                        for a, b in zip(m32, ma))
        print(f'[精度] tf32 vs fp32: 直方图总差 {d}  阈值最大差 {max_bin32:.1f} bin')
        for r in results:
            if r['name'] == 'v6-tf32':
                r['vs_fp32_hist_diff'] = d
                r['vs_fp32_max_bin'] = round(max_bin32, 2)

    # 汇总表
    print('\n========== 汇总 ==========')
    print(f"{'配置':<30}{'wall_s':>9}{'吞吐(G对/s)':>13}{'峰值RSS':>10}{'strict':>8}{'附加':>10}")
    for r in results:
        extra = r.get('collected', '')
        print(f"{r['name']:<30}{str(r.get('wall_s','-')):>9}"
              f"{str(r.get('throughput_Gpairs_s','-')):>13}"
              f"{str(r.get('peak_rss_gb','-'))+'GB':>10}"
              f"{str(r.get('strict_ok','-')):>8}{str(extra):>10}")

    # 关键对比
    by_name = {r['name']: r for r in results}
    def spd(a, b):
        wa, wb = by_name[a].get('wall_s'), by_name[b].get('wall_s')
        if wa and wb:
            return wa / wb
        return None
    base = by_name.get('v6-fp32', {}).get('wall_s')
    pbar = by_name.get('v6-fp32+进度条', {}).get('wall_s')
    coll = by_name.get('v6-fp32+进度条+样本对收集', {}).get('wall_s')
    v5w = by_name.get('v5-fp32(20M bins)', {}).get('wall_s')
    glmw = by_name.get('glm-threaded(特征常驻)', {}).get('wall_s')
    print('\n----- 关键结论 -----')
    if base and pbar:
        print(f'进度条开销: {pbar/base - 1:+.2%} (v6-fp32 {base:.2f}s -> +进度条 {pbar:.2f}s)')
    if base and coll:
        print(f'样本对收集开销: {coll/base - 1:+.2%} (v6-fp32 {base:.2f}s -> +收集 {coll:.2f}s)')
    if base and v5w:
        print(f'v6-fp32 vs v5-fp32: 快 {v5w/base - 1:+.1%} ({v5w:.2f}s -> {base:.2f}s)')
    if base and glmw:
        print(f'v6-fp32 vs glm-threaded(常驻): 慢 {base/glmw - 1:+.1%} '
              f'(glm {glmw:.2f}s -> v6 {base:.2f}s，low_memory 的 H2D 代价)')

    os.makedirs(OUT, exist_ok=True)
    summary = {'data': os.path.basename(args.data), 'N': len(ids),
               'pairs': {'total': total, 'pos': pos, 'neg': neg}, 'results': results}
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果: {os.path.join(OUT, "summary.json")}')


if __name__ == '__main__':
    main()
