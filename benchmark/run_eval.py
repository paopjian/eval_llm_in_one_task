#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估调度器（两阶段）:
    阶段200K: test_data_10min.pkl，单模型 60s 超时 + 400G 内存上限
              —— 与基准(cluster_utils)对比，筛出 200K 中 <=60s 且校验成功的模型
    阶段2M:   test_data_200w.pkl，单模型 30min 超时 + 400G 内存上限（基准同跑）
              —— 只运行 200K 阶段合格的模型

用法:
    python benchmark/run_eval.py --stage 200k          # 只跑 200K 阶段
    python benchmark/run_eval.py --stage 2m            # 跑 2M（需先有200K结果/qualified.json，或--models指定）
    python benchmark/run_eval.py --stage all           # 依次 200K -> 2M(自动取合格者)
    python benchmark/run_eval.py --stage 200k --models glm,codex-sol-2
    python benchmark/run_eval.py --stage 200k --outdir logs/unified_eval/xxx --data test_data_10min.pkl
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = {
    '200k': {'data': 'test_data_10min.pkl', 'label': '200K', 'timeout_s': 60,
             'mem_cap_gb': 400},
    '2m': {'data': 'test_data_200w.pkl', 'label': '2M', 'timeout_s': 1800,
           'mem_cap_gb': 400},
}
ALL_MODELS = ['baseline', 'claude-opus', 'claude-opus-m', 'codex-sol',
              'codex-sol-2', 'codex-sol-m', 'deepseek', 'deepseek-pro',
              'gemini', 'glm', 'glm-pro', 'grok', 'qwen']


def _rss(pid):
    """单进程 RSS (GB)"""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def child_pids(pid):
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
    """进程树总 RSS (GB)"""
    total = _rss(pid)
    for c in child_pids(pid):
        total += tree_rss(c)
    return total


def run_model(stage_cfg, model, data_path, outdir, gpus, logf, max_n=None):
    """跑单个模型：wall 超时 + 进程树内存上限，返回结果 dict（进程组整树击杀，无残留）"""
    os.makedirs(os.path.join(outdir, 'work', model), exist_ok=True)
    out_json = os.path.join(outdir, f'{model}.json')
    log_path = os.path.join(outdir, f'{model}.log')
    if os.path.exists(out_json):
        os.remove(out_json)

    cmd = [sys.executable, os.path.join(REPO_ROOT, 'benchmark', 'run_one.py'),
           '--model', model, '--data', data_path, '--gpus', ','.join(map(str, gpus)),
           '--workdir', os.path.join(outdir, 'work'), '--out', out_json]
    if max_n:
        cmd += ['--max-n', str(max_n)]

    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, start_new_session=True)
        t0 = time.monotonic()
        deadline = t0 + stage_cfg['timeout_s']
        cap_gb = stage_cfg['mem_cap_gb']
        peak_rss = 0.0
        reason = None
        while True:
            rc = proc.poll()
            now = time.monotonic()
            if rc is not None:
                break
            if now > deadline:
                reason = f'timeout>{stage_cfg["timeout_s"]}s'
                logf.write(f'[超时] {model} 超过 {stage_cfg["timeout_s"]}s，终止\n')
                logf.flush()
                _kill_tree(proc)
                proc.wait()
                break
            rss = tree_rss(proc.pid)
            peak_rss = max(peak_rss, rss)
            if rss > cap_gb:
                reason = f'mem>{cap_gb}GB(实际{ rss:.1f}GB)'
                logf.write(f'[内存超限] {model} RSS {rss:.1f}GB > {cap_gb}GB，终止\n')
                logf.flush()
                _kill_tree(proc)
                proc.wait()
                break
            time.sleep(0.3)
        wall = time.monotonic() - t0
        # 结束后补一次峰值
        for _ in range(5):
            rss = tree_rss(proc.pid)
            peak_rss = max(peak_rss, rss)
            if rss <= 0:
                break
            time.sleep(0.2)

    res = {'model': model, 'stage': stage_cfg['label'], 'wall_s': round(wall, 3),
           'peak_rss_gb': round(peak_rss, 2)}
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                inner = json.load(f)
            res.update({k: inner[k] for k in ('status', 'core_s', 'load_s', 'N',
                                              'pairs', 'validation', 'metrics',
                                              'meta', 'reason', 'wall_s')
                        if k in inner})
            res['wall_s'] = round(wall, 3)
        except Exception as e:  # noqa: BLE001
            res['status'] = 'error'
            res['reason'] = f'结果JSON解析失败: {e}'
    else:
        res['status'] = 'failed'
        res['reason'] = reason or f'exit={proc.returncode}'
    if reason and res.get('status') in ('success', None):
        res['status'] = 'killed'
        res['reason'] = reason
    return res


def _kill_tree(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['200k', '2m', 'all'])
    ap.add_argument('--models', default='', help='逗号分隔；默认全部（2M默认取合格名单）')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    ap.add_argument('--data', default='', help='覆盖数据集文件')
    ap.add_argument('--outdir', default='')
    ap.add_argument('--timeout', type=float, default=0, help='覆盖超时秒数')
    ap.add_argument('--mem-cap-gb', type=float, default=0)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(',') if g]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = args.outdir or os.path.join('logs', 'unified_eval', f'run_{ts}')
    os.makedirs(outdir, exist_ok=True)

    stages = ['200k', '2m'] if args.stage == 'all' else [args.stage]
    qualified = []

    for st in stages:
        cfg = dict(STAGES[st])
        if args.timeout:
            cfg['timeout_s'] = args.timeout
        if args.mem_cap_gb:
            cfg['mem_cap_gb'] = args.mem_cap_gb
        data_path = args.data or os.path.join(REPO_ROOT, cfg['data'])
        if not os.path.exists(data_path):
            print(f'❌ 数据集不存在: {data_path}')
            sys.exit(1)
        st_dir = os.path.join(outdir, st)
        os.makedirs(st_dir, exist_ok=True)
        log_path = os.path.join(st_dir, 'runner.log')

        if args.models:
            models = [m.strip() for m in args.models.split(',') if m.strip()]
        elif st == '200k':
            models = ALL_MODELS
        else:
            qf = os.path.join(outdir, 'qualified.json')
            if os.path.exists(qf):
                with open(qf) as f:
                    qualified = json.load(f)
                models = [m['model'] for m in qualified]
            else:
                print(f'❌ 2M 阶段需要先运行 200K 阶段生成 {qf}')
                sys.exit(1)

        print(f'\n========== 阶段 {cfg["label"]}: {cfg["data"]} '
              f'timeout={cfg["timeout_s"]}s mem_cap={cfg["mem_cap_gb"]}GB '
              f'共 {len(models)} 个 ==========')
        with open(log_path, 'w') as logf:
            logf.write(f'stage={st} start={datetime.now()}\n')
            results = []
            for i, model in enumerate(models, 1):
                print(f'\n[{i}/{len(models)}] {model} ...', flush=True)
                res = run_model(cfg, model, data_path, st_dir, gpus, logf)
                logf.write(json.dumps(res, ensure_ascii=False) + '\n')
                logf.flush()
                results.append(res)
                print(f'   {model}: status={res["status"]} wall={res.get("wall_s")}s '
                      f'core={res.get("core_s")}s peakRSS={res.get("peak_rss_gb")}GB '
                      f'{res.get("reason", "")}', flush=True)
            logf.write(f'stage end={datetime.now()}\n')

        results.sort(key=lambda r: (r.get('wall_s') if r['status'] == 'success' else 1e18))
        with open(os.path.join(st_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'stage': st, 'config': cfg, 'results': results}, f,
                      ensure_ascii=False, indent=2)

        # 表格
        print(f'\n===== {cfg["label"]} 阶段汇总（按耗时排序）=====')
        print(f'{"模型":<14}{"状态":<10}{"耗时s":>9}{"core_s":>8}{"峰值RSS GB":>12}')
        for r in results:
            print(f'{r["model"]:<14}{r["status"]:<10}{r.get("wall_s", 0):>9.1f}'
                  f'{r.get("core_s", 0):>8.1f}{r.get("peak_rss_gb", 0):>12.1f}')

        if st == '200k':
            qualified = [r for r in results
                         if r['status'] == 'success' and r['wall_s'] <= cfg['timeout_s']]
            with open(os.path.join(outdir, 'qualified.json'), 'w', encoding='utf-8') as f:
                json.dump(qualified, f, ensure_ascii=False, indent=2)
            print(f'\n✅ 200K 合格（≤{cfg["timeout_s"]}s 且校验通过，{len(qualified)}个）: '
                  + ', '.join(r['model'] for r in qualified))

    print(f'\n结果目录: {outdir}')


if __name__ == '__main__':
    main()
