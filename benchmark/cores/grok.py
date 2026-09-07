#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grok (grok-4.6) core —— 提炼自 grok/eval_similarity.py

原实现要点（行号见 grok/eval_similarity.py）:
    * 并行: ThreadPoolExecutor 同进程每 GPU 一线程，线程内 configure_device()
      (L108-113: set_device + allow_tf32=False + matmul_precision='highest')。
      刻意不用 spawn 多进程 —— 总结.md: spawn 直方图 4.38s vs 线程版 1.31s
      （省掉 7 次进程启动 / 重复 import / 重复 CUDA 初始化）
    * 负载均衡: row_splits() (L83-98) 按"上三角对数"均分行区间（静态划分，
      非动态队列）；每对 (i,j), i<j 由"较小行号所在行块"统计一次，
      pairs_in_row_range() (L101-106) 用于校核每卡对数
    * 分块: 每行块内列从 rs 扫到 n（block=8192）；列块完全在行块右侧
      (cs>=re) 时整块计入上三角；与行块重叠的列块（每行块首个）用 i<j 掩码
    * 精度: fp32 精确矩阵乘（关闭 TF32 + highest），sim clamp 到 [-1,1]
    * 直方图: torch.histc 按 HIST_CHUNK=2^23 (L23, 8_388_608) 元素分块累加进
      int64 —— float32 histc 计数每 bin <= 2^24 才精确，2^23 分块保证精确
    * 正样本: 原实现用单独 batched bmm 按身份分组算精确分数
      (compute_positive_pairs L244-290)；本 core 为保证统一网格下
      all/pos 两直方图 bin 语义逐位一致（避免减法负计数、计数严格相等），
      改为扫描 tile 内用 ids 等值掩码对"同一份 sim 值"做同一套分块 histc
      （与 glm core 同套路；原代码 L554-557 曾出现 bin 边界差并被裁剪，
      说明分开两趟统计在严格校验下不可靠）
    * 适配: bins 100_000 -> 统一 200_000；去掉数据读取/绘图/高分负对
      topk 提取（neg_cap/dyn_thr 等非核心部分）

    neg_hist = hist_all - hist_pos：同一份 sim 值的同一套 histc =>
    逐 bin 非负、两计数之和恒等于 N*(N-1)//2。
"""
import threading
import time

import numpy as np
import torch

from .. import common

MODEL_NAME = 'grok'
MODEL_DESC = ('Grok: 线程池每GPU一线程 + 按上三角对数均分行区间静态划分 + '
              'fp32精确(关TF32) + 2^23分块histc，正样本扫描内ids掩码同值收集')
ORIGIN = ('grok/eval_similarity.py (gpu_hist_worker/row_splits/pairs_in_row_range/'
          'accumulate_hist/configure_device；bins 100K->200K，负对topk提取与'
          '单独bmm正通道移除，正样本改扫描内掩码)')

BLOCK = 8192                      # 原默认 --block-size
HIST_CHUNK = 8_388_608            # 原 HIST_CHUNK = 2^23，float32 histc 计数精确


def _row_splits(n, n_gpu):
    """原 row_splits：按上三角对数均分行区间，使各 GPU 工作量接近。"""
    total = n * (n - 1) // 2
    splits = [0]
    for g in range(1, n_gpu):
        target = total * g // n_gpu
        lo, hi = splits[-1], n
        while lo < hi:
            mid = (lo + hi) // 2
            if mid * (2 * n - mid - 1) // 2 >= target:
                hi = mid
            else:
                lo = mid + 1
        splits.append(int(lo))
    splits.append(n)
    return splits


def _pairs_in_row_range(r0, r1, n):
    """原 pairs_in_row_range：行区间 [r0,r1) 内 (i<j) 的对数。"""
    def cum(r):
        return r * (2 * n - r - 1) // 2

    return cum(r1) - cum(r0)


def _configure_device(device_id):
    """原 configure_device：fp32 最高精度，不启用 TF32。"""
    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device(f"cuda:{device_id}")


def _accumulate_hist(values, hist_i64, n_bins):
    """原 accumulate_hist：按 HIST_CHUNK 分块 torch.histc，累加进 int64。

    每块 <= 2^23 个值 -> 任意 bin 计数 <= 2^23 < 2^24，float32 计数精确。
    """
    if values.numel() == 0:
        return
    flat = values.reshape(-1)
    n = flat.numel()
    for s in range(0, n, HIST_CHUNK):
        e = min(s + HIST_CHUNK, n)
        h = torch.histc(flat[s:e], bins=n_bins, min=common.LO, max=common.HI)
        hist_i64.add_(h.to(torch.int64))


def _gpu_hist_worker(device_id, feats_np, ids_np, r0, r1, n, n_bins, block, out, slot):
    """单卡 worker（原 gpu_hist_worker 的统计核心 + 扫描内正样本掩码收集）。

    统计 [r0,r1) x [r0,n) 中全部 i<j 对: 全体进 hist_all；
    其中 ids 相同的对进 hist_pos（与 hist_all 同一份 sim、同一套 histc）。
    """
    t0 = time.time()
    try:
        device = _configure_device(device_id)
        feats = torch.from_numpy(feats_np).to(device, non_blocking=True)
        ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        hist_all = torch.zeros(n_bins, dtype=torch.int64, device=device)
        hist_pos = torch.zeros(n_bins, dtype=torch.int64, device=device)
        n_done = 0
        done_row_blocks = 0
        n_row_blocks = (max(r1 - r0, 0) + block - 1) // block if r1 > r0 else 0

        for rs in range(r0, r1, block):
            re = min(rs + block, r1)
            feat_r = feats[rs:re]
            ids_r = ids[rs:re]
            br = re - rs
            for cs in range(rs, n, block):
                ce = min(cs + block, n)
                feat_c = feats[cs:ce]
                ids_c = ids[cs:ce]
                sim = torch.mm(feat_r, feat_c.t())
                sim.clamp_(common.LO, common.HI)

                if cs >= re:
                    # 列块完全在行块右侧：i<j 自动成立，整块计入
                    _accumulate_hist(sim, hist_all, n_bins)
                    n_done += br * (ce - cs)
                    posm = ids_r[:, None] == ids_c[None, :]
                else:
                    # 与行块重叠的列块（含主对角）：只统计 i<j，正样本同掩码
                    i_idx = torch.arange(rs, re, device=device).unsqueeze(1)
                    j_idx = torch.arange(cs, ce, device=device).unsqueeze(0)
                    valid = i_idx < j_idx
                    _accumulate_hist(sim[valid], hist_all, n_bins)
                    n_done += int(valid.sum().item())
                    posm = (ids_r[:, None] == ids_c[None, :]) & valid
                pv = sim[posm]
                if pv.numel() > 0:
                    _accumulate_hist(pv, hist_pos, n_bins)
                del sim

            done_row_blocks += 1
            if done_row_blocks in (1, n_row_blocks) or done_row_blocks % 2 == 0:
                print(
                    f"[GPU {device_id}] 行块 {done_row_blocks}/{n_row_blocks}  "
                    f"rows=[{rs},{re})  已统计对数={n_done:,}  用时={time.time()-t0:.1f}s",
                    flush=True,
                )

        torch.cuda.synchronize(device)
        dt = time.time() - t0
        out[slot] = {
            'device_id': int(device_id),
            'hist_all': hist_all.detach().cpu().numpy().copy(),
            'hist_pos': hist_pos.detach().cpu().numpy().copy(),
            'n_done': int(n_done),
            'time': dt,
        }
        print(f"[GPU {device_id}] 完成 rows=[{r0},{r1})  对数={n_done:,}  "
              f"hist_sum={int(out[slot]['hist_all'].sum()):,}  用时={dt:.2f}s",
              flush=True)
    except Exception as e:  # noqa: BLE001 —— 存回主线程再抛出，避免线程内静默死亡
        try:
            e.device_id = device_id
        except Exception:  # noqa: BLE001
            pass
        out[slot] = e


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    n_gpu = len(gpus)
    if n_gpu == 0:
        raise RuntimeError('至少需要一张 GPU')

    # 原 row_splits：按上三角对数均分行区间（静态划分，无动态调度）
    splits = _row_splits(N, n_gpu)
    total_pairs, n_pos, _ = common.pair_stats(ids)

    # 预热 CUDA 上下文，避免首个 kernel 抖动（原 main L507-509）
    for g in gpus:
        torch.zeros(1, device=f'cuda:{g}')

    print('\n  各 GPU 行划分（上三角对数均衡）:')
    for g, r0, r1 in zip(gpus, splits[:-1], splits[1:]):
        pc = _pairs_in_row_range(r0, r1, N)
        print(f'    cuda:{g}  rows=[{r0:6d}, {r1:6d})  pairs={pc:,}  '
              f'({pc / total_pairs * 100:.2f}%)', flush=True)

    # 原 ThreadPoolExecutor 线程并行：每 GPU 一线程（此处用标准库 threading 实现）
    t_hist = time.perf_counter()
    out = [None] * n_gpu
    threads = []
    for slot, (dev, r0, r1) in enumerate(zip(gpus, splits[:-1], splits[1:])):
        th = threading.Thread(
            target=_gpu_hist_worker,
            args=(dev, feats, ids, r0, r1, N, common.BINS, BLOCK, out, slot),
            name=f'gpu-{dev}',
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    core_s = time.perf_counter() - t_hist

    hist_all = hist_pos = None
    for res in out:
        if isinstance(res, Exception):
            dev = getattr(res, 'device_id', '?')
            raise RuntimeError(f'[gpu{dev}] worker 失败: {type(res).__name__}: {res}') from res
        ha, hp = res['hist_all'], res['hist_pos']
        hist_all = ha if hist_all is None else hist_all + ha
        hist_pos = hp if hist_pos is None else hist_pos + hp

    neg_hist = hist_all - hist_pos
    if int(hist_all.sum()) != total_pairs:
        raise RuntimeError(f'全体直方图计数 {int(hist_all.sum()):,} != 理论 {total_pairs:,}')
    if int(hist_pos.sum()) != n_pos:
        raise RuntimeError(f'正样本直方图计数 {int(hist_pos.sum()):,} != 理论 {n_pos:,}')
    if int(neg_hist.min()) < 0:
        raise RuntimeError('负样本直方图出现负计数')

    print(f'\n  直方图合计对数: {int(hist_all.sum()):,}  (期望 {total_pairs:,})')
    print(f'  正样本 hist: {int(hist_pos.sum()):,} / {n_pos:,}    '
          f'负样本 hist: {int(neg_hist.sum()):,} / {total_pairs - n_pos:,}')
    print(f'  线程并行直方图耗时: {core_s:.2f}s', flush=True)

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': ('同进程线程池(每GPU一线程, configure_device+set_device, '
                     '原ThreadPoolExecutor)；静态行区间划分(按上三角对数均衡, 无动态队列)'),
        'block': BLOCK,
        'precision': 'fp32 (allow_tf32=False, cudnn tf32 off, matmul_precision=highest)',
        'native_bins': 100_000,
        'hist_chunk': HIST_CHUNK,
        'bin_adaptation': 'bins 100K->统一200K；正样本由单独bmm通道改为扫描tile内ids掩码同值histc',
        'core_s': core_s,
        'per_gpu': [{'device': r['device_id'], 'pairs': r['n_done'], 's': round(r['time'], 3)}
                    for r in out],
    }
    return hist_pos, neg_hist, meta
