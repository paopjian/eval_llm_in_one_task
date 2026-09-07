#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型执行器（由 run_eval.py 以子进程方式调用）:
    读取数据（统一 loader）-> 运行模型 core -> 直方图校验 -> TPIR@FPIR 指标
    -> 结果 JSON

用法:
    python benchmark/run_one.py --model glm --data test_data_10min.pkl \
        --gpus 0,1,2,3,4,5,6 --workdir <临时目录> --out <结果json>
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark import common
from benchmark.cores import get_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    ap.add_argument('--workdir', default='logs/unified_eval/work')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-n', type=int, default=0, help='仅取前N条样本（冒烟测试用，0=全部）')
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    os.makedirs(args.workdir, exist_ok=True)
    workdir = os.path.join(args.workdir, args.model)
    os.makedirs(workdir, exist_ok=True)

    result = {'model': args.model, 'data': os.path.basename(args.data),
              'start': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        t0 = time.perf_counter()
        feats, ids = common.load_data(args.data)
        if args.max_n:
            feats, ids = feats[:args.max_n], ids[:args.max_n]
        result['load_s'] = round(time.perf_counter() - t0, 3)
        total, pos, neg = common.pair_stats(ids)
        result['N'] = int(len(ids))
        result['pairs'] = {'total': total, 'pos': pos, 'neg': neg}

        core = get_core(args.model)
        t1 = time.perf_counter()
        pos_hist, neg_hist, meta = core.compute(feats, ids, gpus, workdir)
        core_s = time.perf_counter() - t1
        result['core_s'] = round(core_s, 3)
        result['meta'] = meta

        ok, detail = common.validate_hists(pos_hist, neg_hist, ids)
        result['validation'] = {'ok': ok, **detail}
        if not ok:
            result['status'] = 'invalid'
            result['reason'] = '直方图计数校验失败: ' + '; '.join(detail['issues'])
        else:
            result['metrics'] = common.compute_metrics(pos_hist, neg_hist)
            result['status'] = 'success'
            print('\n'.join(['TPIR@FPIR:'] + common.fmt_tpir(result['metrics']).splitlines()),
                  flush=True)
        result['wall_s'] = round(time.perf_counter() - t0, 3)
    except Exception as e:  # noqa: BLE001
        import traceback
        result['status'] = 'error'
        result['reason'] = f'{type(e).__name__}: {e}'
        result['traceback'] = traceback.format_exc()[-4000:]
        print(result['traceback'], file=sys.stderr, flush=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[完成] {args.model} status={result.get('status')} "
          f"core={result.get('core_s', '?')}s wall={result.get('wall_s', '?')}s", flush=True)


if __name__ == '__main__':
    main()
