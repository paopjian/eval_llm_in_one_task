#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B 实验：glm-线程版 vs grok —— 两个都是“每 GPU 一线程”的实现，比较算法差异带来的
速度/内存差异。

    grok (cores/grok.py):         BLOCK=8192 + 静态行区间划分 + 2^23 分块 histc
    glm-线程版 (experiments/glm_threaded.py): BLOCK=16384 + 动态tile队列(宽优先) + 每 tile 一次 histc

用法:
    python benchmark/experiments/test_grok_vs_glm_threads.py --data test_data_10min.pkl   # 200K
    python benchmark/experiments/test_grok_vs_glm_threads.py --data test_data_200w.pkl   # 2M
输出: logs/unified_eval/grok_vs_glm_threads/summary.json
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
from benchmark.cores import grok as grok_core  # noqa: E402
from benchmark.experiments import glm_threaded as glm_thr  # noqa: E402

OUT = os.path.join('logs', 'unified_eval', 'grok_vs_glm_threads')


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
    rg, pos_g, neg_g = run_core(grok_core, 'grok(cores/grok.py)', feats, ids, gpus,
                                os.path.join(OUT, 'work', 'grok'))
    results.append(rg)
    rt, pos_t, neg_t = run_core(glm_thr, 'glm-线程版(experiments/glm_threaded.py)', feats, ids,
                                gpus, os.path.join(OUT, 'work', 'glm_threads'))
    results.append(rt)

    same = bool((pos_g == pos_t).all() and (neg_g == neg_t).all())
    diff = int(abs(pos_g.astype('int64') - pos_t.astype('int64')).sum()) + \
           int(abs(neg_g.astype('int64') - neg_t.astype('int64')).sum())
    print(f'\n[一致性] 两版直方图逐位相同={same} (总计数差 {diff})')

    print('\n========== 对比 ==========')
    print(f"{'变体':<36}{'wall_s':>9}{'core_s':>9}{'吞吐(G对/s)':>14}{'峰值RSS':>11}{'strict':>8}")
    for r in results:
        print(f"{r['name']:<36}{r['wall_s']:>9}{r['core_s']:>9}"
              f"{r['throughput_Gpairs_s']:>14}{str(r['peak_rss_gb'])+'GB':>11}"
              f"{str(r['strict_ok']):>8}")

    if rg['status'] == 'success' and rt['status'] == 'success':
        speed = rg['wall_s'] / rt['wall_s']
        mem = rg['peak_rss_gb'] / max(rt['peak_rss_gb'], 1e-9)
        print(f'\nglm-线程版相对 grok: 速度 {"快" if speed > 1 else "慢"} {abs(speed-1)*100:.1f}% '
              f'(grok wall/线程 wall={speed:.3f})；'
              f'峰值RSS 线程/grok={rt["peak_rss_gb"]/max(rg["peak_rss_gb"],1e-9):.2f}× '
              f'(grok {rg["peak_rss_gb"]}GB -> 线程 {rt["peak_rss_gb"]}GB)')

    os.makedirs(OUT, exist_ok=True)
    summary = {'data': os.path.basename(data_path), 'N': len(ids),
               'pairs': {'total': total, 'pos': pos, 'neg': neg},
               'hist_identical': same, 'hist_total_diff': diff, 'results': results}
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果: {os.path.join(OUT, "summary.json")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='test_data_10min.pkl')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    ab(os.path.join(REPO_ROOT, args.data), gpus)


if __name__ == '__main__':
    main()
