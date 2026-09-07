#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度 A/B 实验：fp16 vs fp32（cluster_utils v4/v5 + glm + codex-sol，2M 全量，7 卡）。

只做实验对比，不参与正式评估。输出:
    logs/unified_eval/ab_fp16_fp32/<name>.json   每项明细
    logs/unified_eval/ab_fp16_fp32/summary.json  汇总
"""
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from benchmark import common  # noqa: E402
from benchmark.cores import baseline as base_core  # noqa: E402
from benchmark.cores import glm as glm_core  # noqa: E402
from benchmark.cores import codex_sol as cs_core  # noqa: E402
from benchmark.experiments import glm_fp16  # noqa: E402
from benchmark.experiments import codex_sol_fp32  # noqa: E402

MATRIX = [
    ('baseline-v4-fp32', base_core, {}, 'v4/fp32(历史)'),
    ('baseline-v5-fp32', base_core, {'CLUSTER_UTILS_VER': 'v5'}, 'v5/fp32'),
    ('baseline-v5-fp16', base_core, {'CLUSTER_UTILS_VER': 'v5',
                                     'CLUSTER_UTILS_PREC': 'fp16'}, 'v5/fp16'),
    ('glm-fp32', glm_core, {}, '动态队列/fp32(关TF32)'),
    ('glm-fp16', glm_fp16, {}, '动态队列/fp16'),
    ('codex-sol-fp16', cs_core, {}, 'LPT静态/fp16'),
    ('codex-sol-fp32', codex_sol_fp32, {}, 'LPT静态/fp32'),
]

OUT = os.path.join('logs', 'unified_eval', 'ab_fp16_fp32')
os.makedirs(OUT, exist_ok=True)


def main():
    feats, ids = common.load_data(os.path.join(REPO_ROOT, 'test_data_200w.pkl'))
    total, pos, neg = common.pair_stats(ids)
    print(f'N={len(ids):,} pairs total={total:,} pos={pos:,} neg={neg:,}', flush=True)
    gpus = list(range(7))
    anchor = None
    results = []
    for name, core, env, desc in MATRIX:
        for k, v in env.items():
            os.environ[k] = v
        print(f'\n===== {name} ({desc}) =====', flush=True)
        workdir = os.path.join(OUT, 'work', name)
        os.makedirs(workdir, exist_ok=True)
        rec = {'name': name, 'desc': desc}
        try:
            t0 = time.perf_counter()
            pos_hist, neg_hist, meta = core.compute(feats, ids, gpus, workdir)
            wall = time.perf_counter() - t0
            rec['wall_s'] = round(wall, 3)
            rec['core_s'] = round(meta.get('core_s', wall), 3)
            rec['meta'] = {k: v for k, v in meta.items()
                           if k in ('model', 'parallel', 'precision',
                                    'cluster_utils_version', 'matmul_s')}
            ok, vd = common.validate_hists(pos_hist, neg_hist, ids)
            rec['validation'] = {k: vd[k] for k in
                                 ('strict_ok', 'got_pos', 'got_neg', 'delta_pos',
                                  'delta_neg', 'issues')}
            rec['status'] = 'success' if ok else 'invalid'
            if ok:
                m = common.compute_metrics(pos_hist, neg_hist)
                rec['metrics'] = m
                if anchor is None:
                    anchor = m
                max_bin = max(abs(round(a['threshold'] - b['threshold'], 6)) / common.W
                              for a, b in zip(m, anchor))
                max_tp = max(abs(a['tpir'] - b['tpir']) for a, b in zip(m, anchor))
                rec['vs_anchor_max_bin'] = round(float(max_bin), 2)
                rec['vs_anchor_max_tpir'] = float(max_tp)
                rec['throughput_Gpairs_s'] = round(total / wall / 1e9, 2)
                print(f"  {rec['status']} wall={wall:.1f}s core={rec['core_s']}s "
                      f"throughput={rec['throughput_Gpairs_s']}G对/s "
                      f"strict={vd['strict_ok']} | 阈值最大差 {max_bin:.1f} bin "
                      f"TPIR最大差 {max_tp:.2e}", flush=True)
            else:
                print(f"  invalid: {vd['issues']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec['status'] = 'error'
            rec['reason'] = f'{type(e).__name__}: {e}'
            print(f"  error: {rec['reason']}", flush=True)
        finally:
            for k in env:
                os.environ.pop(k, None)
        with open(os.path.join(OUT, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
        results.append(rec)

    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果目录: {OUT}/summary.json')


if __name__ == '__main__':
    main()
