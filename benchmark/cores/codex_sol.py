#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex(SOL) core —— 提炼自 codex-sol/face_similarity_eval.py

真实运行参数见 codex-sol/results/evaluation_summary.json：block_size=2048、
bins=4096、precision=fp16、7 卡（test_data_200w.pkl 上约 110s）。

原实现要点（并行/分块/精度照原代码，见 face_similarity_eval.py 的
evaluate_similarity / _partition_ranges / _compute_ranges / _score_to_torch_hist）：
    * 并行：行块区间 [r0, r0+B) 静态切分（不做动态任务队列），按剩余工作量
      N-r0 降序做 LPT 贪心平衡分组到各 GPU；spawn 每 GPU 一进程各扫一个组
    * 分块：每个行块自其所在对角块向右扫到 N；对角块用 triu(diagonal=1)
      只保留严格上三角，非对角块整体有效 => 每个 (i<j) 样本对恰统计一次
    * 精度：fp16 —— GPU 上 fp32 特征转 half 后 matmul（tensor-core），结果
      .float() 回 fp32 再入直方图
    * 直方图非 torch.histc：手工映射 idx=floor((s+1)*bins/2)（先 clamp[-1,1]）
      再 torch.bincount；负样本直方图 = 有效全量直方图 - 同身份(正)直方图
      （逐块相减，int64 累计）
    * bins 适配：原 bins=4096，统一网格下改为 200_000（common.BINS），
      映射函数与其余逻辑不变
    * 读取/归一化/绘图/样本对提取等非核心部分已去掉，由统一框架承担
"""
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'codex-sol'
MODEL_DESC = ('Codex(SOL) 方法：静态行块LPT贪心分组(无动态队列) + '
              'fp16分块matmul + 手工bincount直方图(上三角逐块)')
ORIGIN = ('codex-sol/face_similarity_eval.py (evaluate_similarity/_partition_ranges/'
          '_compute_ranges/_score_to_torch_hist；bins 4096 -> 统一 200K)')

BLOCK = 2048            # 实际运行的 block_size
NATIVE_BINS = 4096      # 原默认 bins
PRECISION = 'fp16'      # CUDA 上的有效精度（原 precision='auto' -> fp16）


def _iter_row_ranges(start, stop, block_size):
    """原 _iter_row_ranges：按 block_size 产出 [s, s+block) 行区间"""
    for row_start in range(start, stop, block_size):
        yield row_start, min(row_start + block_size, stop)


def _partition_ranges(ranges, device_count, sample_count):
    """原 _partition_ranges：按剩余行数估算工作量，LPT 贪心平衡多卡负载。

    工作量权重取 sample_count - r[0]（该行块剩余的行数），最重的行块先分给
    当前累计负载最小的 GPU —— 静态分组，之后不再动态调度。
    """
    groups = [[] for _ in range(device_count)]
    loads = [0] * device_count
    weighted = sorted(ranges, key=lambda r: sample_count - r[0], reverse=True)
    for item in weighted:
        target = min(range(device_count), key=loads.__getitem__)
        groups[target].append(item)
        loads[target] += sample_count - item[0]
    return groups


def _score_to_torch_hist(scores, bins):
    """原 _score_to_torch_hist：手工 bin 映射 + bincount（不用 torch.histc）。

    索引 idx = floor((clamp(s,-1,1) + 1) * bins / 2)，与统一网格
    [-1, 1] / 200_000 bins 的 np.histogram 语义一致；范围外值经 clamp 也
    计入端点 bin（保证每对样本恰统计一次，计数总和恒等于理论值）。
    """
    if scores.numel() == 0:
        return torch.zeros(bins, dtype=torch.int64, device=scores.device)
    indices = ((scores.clamp(-1.0, 1.0) + 1.0) * (bins / 2.0)).to(torch.int64)
    indices.clamp_(0, bins - 1)
    return torch.bincount(indices, minlength=bins)


def _run_group(cfg, feats_shm, ids_shm, res_q=None):
    """单 GPU worker：计算一个静态行块组的全部上三角分块。

    顶层函数，spawn/fork 安全；cfg/共享张量均为可序列化对象。
    res_q 为 None 时（单组直跑）直接返回结果 dict，否则经队列回传。
    """
    dev = f'cuda:{cfg["gpu"]}'
    torch.cuda.set_device(dev)
    N, block, bins = cfg['N'], cfg['block'], cfg['bins']
    gpu, group, n_total = cfg['gpu'], cfg['group'], cfg['n_tiles']
    try:
        # 原 _compute_ranges：fp32 特征全量上卡后按精度转 half
        feature_tensor = feats_shm.to(dev)
        if cfg['precision'] == 'fp16':
            feature_tensor = feature_tensor.half()
        id_tensor = ids_shm.to(dev)
        torch.cuda.synchronize()

        hist_pos = torch.zeros(bins, dtype=torch.int64, device=dev)
        hist_neg = torch.zeros(bins, dtype=torch.int64, device=dev)
        pos_cnt = neg_cnt = 0
        n_done = 0
        t_mm = t_hist = 0.0
        with torch.inference_mode():
            for (r0, r1) in group:                       # 组内行块（原 row_ranges）
                for i0, i1 in _iter_row_ranges(r0, r1, block):
                    A = feature_tensor[i0:i1]
                    for j0, j1 in _iter_row_ranges(i0, N, block):
                        # fp16 tensor-core matmul，结果回 fp32
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        score_matrix = (A @ feature_tensor[j0:j1].T).float()
                        torch.cuda.synchronize(); t_mm += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        same_ids = id_tensor[i0:i1, None] == id_tensor[None, j0:j1]
                        if j0 == i0:
                            # 对角块只保留严格上三角；其余块天然满足全局 i<j
                            valid = torch.triu(
                                torch.ones_like(score_matrix, dtype=torch.bool),
                                diagonal=1)
                            valid_scores = score_matrix[valid]
                            pos = score_matrix[same_ids & valid]
                            n_valid = int(valid.sum().item())
                        else:
                            valid_scores = score_matrix.reshape(-1)
                            pos = score_matrix[same_ids]
                            n_valid = score_matrix.numel()
                        full_hist = _score_to_torch_hist(valid_scores, bins)
                        pos_hist = _score_to_torch_hist(pos, bins)
                        hist_pos += pos_hist
                        hist_neg += full_hist - pos_hist   # 负 = 全量有效 - 同身份
                        n_pos = int(pos.numel())
                        pos_cnt += n_pos
                        neg_cnt += n_valid - n_pos
                        torch.cuda.synchronize()
                        t_hist += time.perf_counter() - t0
                        n_done += 1
                        if n_done % 64 == 0 or n_done == n_total:
                            print(f'  [codex-sol gpu{gpu}] tiles {n_done}/{n_total} '
                                  f'pos={pos_cnt:,} neg={neg_cnt:,}', flush=True)
        torch.cuda.synchronize()
        res = {'ok': True,
               'pos_hist': hist_pos.cpu().numpy(),
               'neg_hist': hist_neg.cpu().numpy(),
               'pos_cnt': pos_cnt, 'neg_cnt': neg_cnt,
               'matmul_s': t_mm, 'hist_s': t_hist}
    except Exception as e:  # noqa: BLE001
        msg = f'{type(e).__name__}: {e}'
        print(f'  [codex-sol gpu{gpu}] worker 异常: {msg}', flush=True)
        res = {'ok': False, 'err': msg}
    if res_q is not None:
        res_q.put(res)
        return None
    return res


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    if not gpus:
        raise ValueError('codex-sol core 需要至少一个 GPU（fp16 CUDA matmul）')
    os.makedirs(workdir, exist_ok=True)
    _, pos_pairs, neg_pairs = common.pair_stats(ids)

    # spawn 前在主进程建立共享内存张量（不触碰 CUDA），子进程零拷贝接收
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    # 原 evaluate_similarity 的调度：静态行块区间 -> LPT 贪心分组（每 GPU 一组）
    ranges = list(_iter_row_ranges(0, N, BLOCK))
    groups = _partition_ranges(ranges, len(gpus), N)
    group_args = [(gpu, grp) for gpu, grp in zip(gpus, groups) if grp]

    t0 = time.perf_counter()
    if len(group_args) == 1:                       # 原 use_parallel=False：单设备直跑
        gpu, grp = group_args[0]
        cfg = {'N': N, 'block': BLOCK, 'bins': common.BINS, 'gpu': gpu,
               'group': grp, 'n_tiles': _count_tiles(grp, N, BLOCK),
               'precision': PRECISION}
        results = [_run_group(cfg, feats_shm, ids_shm, None)]
    else:                                          # spawn 每 GPU 一进程
        ctx = mp.get_context('spawn')
        res_q = ctx.Queue()
        procs = []
        for gpu, grp in group_args:
            cfg = {'N': N, 'block': BLOCK, 'bins': common.BINS, 'gpu': gpu,
                   'group': grp, 'n_tiles': _count_tiles(grp, N, BLOCK),
                   'precision': PRECISION}
            p = ctx.Process(target=_run_group,
                            args=(cfg, feats_shm, ids_shm, res_q))
            p.start()
            procs.append(p)
        results = [res_q.get(timeout=3600) for _ in procs]   # 先取回再 join（防 feeder 管道死锁）
        for p in procs:
            p.join()
    core_s = time.perf_counter() - t0

    pos_hist = np.zeros(common.BINS, dtype=np.int64)
    neg_hist = np.zeros(common.BINS, dtype=np.int64)
    got_pos = got_neg = 0
    t_mm = t_hist = 0.0
    for r in results:
        if not r['ok']:
            raise RuntimeError(f'codex-sol worker 失败: {r.get("err")}')
        pos_hist += r['pos_hist']
        neg_hist += r['neg_hist']
        got_pos += r['pos_cnt']
        got_neg += r['neg_cnt']
        t_mm += r['matmul_s']
        t_hist += r['hist_s']

    # 原 evaluate_similarity 的严格计数断言（框架侧另有统一校验）
    if got_pos != pos_pairs or got_neg != neg_pairs:
        raise RuntimeError(
            '上三角统计数量异常：'
            f'期望正/负={pos_pairs}/{neg_pairs}，'
            f'实际正/负={got_pos}/{got_neg}')
    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 静态行块LPT贪心分组（剩余行数加权，无动态队列）',
        'block': BLOCK,
        'precision': f'{PRECISION} (half matmul -> .float() 回 fp32)',
        'native_bins': NATIVE_BINS,
        'core_s': core_s, 'matmul_s': t_mm, 'hist_s': t_hist,
    }
    return pos_hist, neg_hist, meta


def _count_tiles(group, N, block):
    """组内行块的总分块数（含对角块），用于进度打印分母。"""
    total = 0
    for (r0, r1) in group:
        for i0 in range(r0, r1, block):
            total += (N - i0 + block - 1) // block
    return total
