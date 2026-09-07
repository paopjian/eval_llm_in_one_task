#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B 实验：glm 多进程 vs glm 多线程 —— 验证“进程改线程是否保持速度 + 是否省内存”。

两种模式:
  1) CPU 正确性冒烟（无需 GPU）:
       python benchmark/experiments/test_glm_threads.py --smoke
     用小数据、小分块走 CPU，验证多线程 tile 队列/三角掩码/正样本收集/多线程
     聚合的计数与理论值严格一致（与正式评估同一套 validate_hists 口径）。

  2) GPU 速度/内存对比:
       python benchmark/experiments/test_glm_threads.py --data test_data_10min.pkl   # 200K 快
       python benchmark/experiments/test_glm_threads.py --data test_data_200w.pkl   # 2M 全量
     同一进程内先后跑 cores/glm.py（多进程）与 experiments/glm_threaded.py（多线程），
     比较 wall / core / 吞吐率 / 峰值RSS(进程树) / 校验。

输出目录: logs/unified_eval/glm_proc_vs_threads/summary.json
"""
import argparse
import json
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from benchmark import common  # noqa: E402
from benchmark.cores import glm as glm_proc  # noqa: E402
from benchmark.experiments import glm_threaded as glm_thr  # noqa: E402

OUT = os.path.join('logs', 'unified_eval', 'glm_proc_vs_threads')


# ---------------------------------------------------------------- 峰值 RSS（进程树）
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
    """后台采样整棵进程树 RSS 峰值（GiB）。线程版=单进程；进程版=父+spawn 子进程。"""
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


# ---------------------------------------------------------------- CPU 冒烟
def smoke():
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    N, d, block = 260, 16, 64            # 小数据 + 小分块，覆盖对角/非对角/尾部残块
    feats = rng.standard_normal((N, d)).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)   # L2 归一化 -> sim 在 [-1,1]
    ids = rng.integers(0, 8, size=N).astype(np.int64)       # 有重复 -> 存在正样本对

    total, pos, neg = common.pair_stats(ids)
    print(f'[冒烟] N={N} d={d} block={block} 理论对 total={total:,} pos={pos:,} neg={neg:,}')

    workdir = os.path.join(OUT, 'smoke_work')
    pos_hist, neg_hist, meta = glm_thr.compute(
        feats, ids, [0, 1, 2], workdir, device='cpu', block=block)

    ok, vd = common.validate_hists(pos_hist, neg_hist, ids)
    got_total = int(pos_hist.sum() + neg_hist.sum())
    print(f'[冒烟] got_total={got_total:,} (期望 {total:,})  '
          f'got_pos={vd["got_pos"]:,} (期望 {pos:,})  '
          f'got_neg={vd["got_neg"]:,} (期望 {neg:,})')
    print(f'[冒烟] validate ok={ok} strict_ok={vd["strict_ok"]} '
          f'delta_pos={vd["delta_pos"]} delta_neg={vd["delta_neg"]}')

    assert got_total == total, f'总数不符: {got_total} != {total}'
    assert vd['got_pos'] == pos and vd['got_neg'] == neg, '正/负样本计数不符'
    assert ok and vd['strict_ok'], f'严格校验未通过: {vd["issues"]}'
    print('[冒烟] PASS：多线程 tile 队列/三角掩码/正样本收集/聚合 计数严格一致\n')
    return True


# ---------------------------------------------------------------- GPU A/B
def run_core(core, name, feats, ids, gpus, workdir):
    rec = {'name': name}
    print(f'===== {name} =====', flush=True)
    with PeakRSS() as pm:
        t0 = time.perf_counter()
        pos_hist, neg_hist, meta = core.compute(feats, ids, gpus, workdir)
        wall = time.perf_counter() - t0
    rec['peak_rss_gb'] = round(pm.peak, 2)
    rec['wall_s'] = round(wall, 3)
    rec['core_s'] = round(meta.get('core_s', wall), 3)
    rec['matmul_s'] = round(meta.get('matmul_s', 0.0), 3)
    rec['hist_s'] = round(meta.get('hist_s', 0.0), 3)

    ok, vd = common.validate_hists(pos_hist, neg_hist, ids)
    rec['strict_ok'] = vd['strict_ok']
    rec['got_pos'] = vd['got_pos']
    rec['got_neg'] = vd['got_neg']
    rec['status'] = 'success' if ok else 'invalid'
    total = len(ids) * (len(ids) - 1) // 2
    rec['throughput_Gpairs_s'] = round(total / wall / 1e9, 2)
    print(f"  {rec['status']} wall={wall:.2f}s core={rec['core_s']}s "
          f"throughput={rec['throughput_Gpairs_s']}G对/s peakRSS={rec['peak_rss_gb']}GB "
          f"strict={vd['strict_ok']}", flush=True)
    return rec, pos_hist, neg_hist


def ab(data_path, gpus):
    feats, ids = common.load_data(data_path)
    total, pos, neg = common.pair_stats(ids)
    print(f'\nN={len(ids):,} pairs total={total:,} pos={pos:,} neg={neg:,}', flush=True)

    results = []
    rp, pos_p, neg_p = run_core(glm_proc, 'glm-进程版(cores/glm.py)', feats, ids, gpus,
                                os.path.join(OUT, 'work', 'proc'))
    results.append(rp)
    rt, pos_t, neg_t = run_core(glm_thr, 'glm-线程版(experiments/glm_threaded.py)', feats, ids,
                                gpus, os.path.join(OUT, 'work', 'threads'))
    results.append(rt)

    # 两版直方图逐位一致（算法同、精度同，应完全相同）
    same = bool((pos_p == pos_t).all() and (neg_p == neg_t).all())
    diff = int(abs(pos_p.astype('int64') - pos_t.astype('int64')).sum()) + \
           int(abs(neg_p.astype('int64') - neg_t.astype('int64')).sum())
    print(f'\n[一致性] 两版直方图逐位相同={same} (总计数差 {diff})')

    print('\n========== 对比 ==========')
    print(f"{'变体':<34}{'wall_s':>9}{'core_s':>9}{'吞吐(G对/s)':>14}{'峰值RSS':>11}{'strict':>8}")
    for r in results:
        print(f"{r['name']:<34}{r['wall_s']:>9}{r['core_s']:>9}"
              f"{r['throughput_Gpairs_s']:>14}{str(r['peak_rss_gb'])+'GB':>11}"
              f"{str(r['strict_ok']):>8}")

    if rt['status'] == 'success':
        speed = rp['wall_s'] / rt['wall_s']
        mem = rp['peak_rss_gb'] / max(rt['peak_rss_gb'], 1e-9)
        print(f'\n线程版相对进程版: 速度 {"快" if speed > 1 else "慢"} {abs(speed-1)*100:.1f}% '
              f'(进程wall/线程wall={speed:.3f})；'
              f'峰值RSS 线程/进程={rt["peak_rss_gb"]/max(rp["peak_rss_gb"],1e-9):.2f}× '
              f'(进程 {rp["peak_rss_gb"]}GB -> 线程 {rt["peak_rss_gb"]}GB)')

    os.makedirs(OUT, exist_ok=True)
    summary = {'data': os.path.basename(data_path), 'N': len(ids),
               'pairs': {'total': total, 'pos': pos, 'neg': neg},
               'hist_identical': same, 'hist_total_diff': diff, 'results': results}
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果: {os.path.join(OUT, "summary.json")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='CPU 正确性冒烟（无需 GPU）')
    ap.add_argument('--data', default='test_data_10min.pkl', help='数据 pkl（默认 200K）')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    ab(os.path.join(REPO_ROOT, args.data), gpus)


if __name__ == '__main__':
    main()
