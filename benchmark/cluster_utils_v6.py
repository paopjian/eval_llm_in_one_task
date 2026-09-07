#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_utils_v6 —— 大规模相似度矩阵正/负样本直方图评估（自包含单文件工具）。

在显存不足（low_memory，特征放 CPU 按需 H2D）的前提下，用 7 卡把 N 个 L2 归一化
特征的全体样本对 (i<j) 相似度统计成两张 int64 直方图：
    pos_hist: 同身份样本对（ids[i] == ids[j]）的相似度分布
    neg_hist: 不同身份样本对的相似度分布（= 全体 - 正样本，逐 bin 非负）

设计来源（各模型的优点汇总）:
    * glm           : 大块 16384、单 tile 动态队列(行块序+行内宽优先)、sim clamp、单趟 neg=full-pos
    * codex-sol-2/m : tf32 精度档（allow_tf32，实测对余弦相似度阈值 0 bin 损失、快 ~12%）
    * deepseek-pro  : skip_pos_check —— 静态预标含正样本 tile，热路径对无正样本 tile 免身份等值比对
    * grok          : 单进程多线程（threading，免 spawn 重复 import/CUDA 初始化）
    * 自研          : pin_inplace —— cudaHostRegister 原地锁定 numpy（省掉 pin_memory() 的整份副本）

用法:
    from cluster_utils_v6 import get_sim_matrix_large_scale_v6
    pos_hist, neg_hist = get_sim_matrix_large_scale_v6(
        query_feats_list=feats,        # (N, D) float32，需 L2 归一化
        query_ids=ids,                 # (N,) int64 身份标签
        num_gpus=7,
    )

命令行（读 4 元组 pkl）:
    python cluster_utils_v6.py --data test_data_200w.pkl --gpus 0,1,2,3,4,5,6

依赖: numpy, torch(CUDA)；进度条可选依赖 tqdm（缺失自动禁用）。

返回值:
    (pos_hist, neg_hist) 均为 np.int64，shape (hist_bins,)，bin k 覆盖
    [lo + k*w, lo + (k+1)*w)，w = (hi-lo)/hist_bins（默认 [-1,1] 200K bins，宽 1e-5）。
    若给定 collect_pairs_config，额外返回收集的样本对列表。
"""
import math
import queue
import threading
import time

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


# ============================================================
# 线程安全样本对收集器（v5 兼容）
# ============================================================
class PairCollector:
    """线程安全的样本对收集器，支持数量限制。"""

    def __init__(self, max_pairs: int = -1):
        self.max_pairs = max_pairs
        self.pairs = []
        self.lock = threading.Lock()
        self._is_full = False

    def is_full(self) -> bool:
        if self.max_pairs == -1:
            return False
        return self._is_full

    def add_pairs(self, new_pairs) -> int:
        if self.is_full() or not new_pairs:
            return 0
        with self.lock:
            if self._is_full:
                return 0
            if self.max_pairs == -1:
                self.pairs.extend(new_pairs)
                return len(new_pairs)
            remaining = self.max_pairs - len(self.pairs)
            if remaining <= 0:
                self._is_full = True
                return 0
            to_add = new_pairs[:remaining]
            self.pairs.extend(to_add)
            if len(self.pairs) >= self.max_pairs:
                self._is_full = True
            return len(to_add)

    def get_pairs(self):
        return self.pairs

    def count(self) -> int:
        return len(self.pairs)


def _collect_pairs(sim, mask_2d, row_start, col_start, pair_collector):
    """从 2D sim 矩阵中按掩码收集样本对 (global_i, global_j, score)。"""
    if pair_collector.is_full():
        return
    rows, cols = torch.where(mask_2d)
    if rows.numel() == 0:
        return
    scores = sim[rows, cols]
    global_i = (rows + row_start).cpu().numpy()
    global_j = (cols + col_start).cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    pair_collector.add_pairs(list(zip(global_i, global_j, scores_cpu)))


# ============================================================
# 静态预标含正样本 tile（deepseek-pro 式）
# ============================================================
def _pos_tile_flags(ids, block_size, n):
    """静态判定哪些上三角块 (bi,bj) 含 >=1 条同身份样本对（保守：绝不漏标）。

    对角块：块内某身份 >=2 个成员；非对角：某身份成员跨越块 [bf,bl] 时标记其间
    全部 (b1<b2)。热路径据此跳过无正样本 tile 的身份等值比对。
    """
    nb = math.ceil(n / block_size)
    flags = set()
    if n < 2:
        return flags
    ids = np.asarray(ids)
    for bi in range(nb):
        b0, b1 = bi * block_size, min((bi + 1) * block_size, n)
        _, cnt = np.unique(ids[b0:b1], return_counts=True)
        if cnt.size and int(cnt.max()) >= 2:
            flags.add((bi, bi))
    order = np.argsort(ids, kind='stable')
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1, n]
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < 2:
            continue
        bf = int(order[s]) // block_size
        bl = int(order[e - 1]) // block_size
        if bl > bf:
            for b1 in range(bf, bl):
                for b2 in range(b1 + 1, bl + 1):
                    flags.add((b1, b2))
    return flags


# ============================================================
# GPU worker
# ============================================================
def _gpu_worker(
    feats, ids, tile_q, gpu_id, block_size, n,
    hist_bins, hist_range,
    collect_pairs_config, pair_collector, pos_pair_collector, neg_pair_collector,
    memory_mode, precision, row_cache, hist_method, pbar, pbar_lock, out, slot
):
    """单卡 worker（线程）：单 tile 动态队列 + 行块缓存 + 单趟直方图(neg=full-pos)。"""
    with torch.cuda.device(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        LO, HI = float(hist_range[0]), float(hist_range[1])
        FILL = LO - 2.0                     # 越界填充值，histc 会丢弃
        scale = hist_bins / 2.0             # 量化 scale（bincount 用）
        use_bincount = (hist_method == 'bincount')

        # ---- 精度设置：fp32(关TF32) / tf32 / fp16 ----
        use_fp16 = (precision == 'fp16')
        use_tf32 = (precision == 'tf32')
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

        ids_full = torch.as_tensor(ids, device=device)   # (N,) 常驻显存

        use_gpu_feats = (memory_mode == 'high_performance')
        if use_gpu_feats:
            feats_source = feats.to(device)              # 特征常驻显存
        else:
            feats_source = feats                         # pinned CPU，按需 H2D

        # ---- 解析样本对收集配置 ----
        do_collect_single = False
        do_collect_dual = False
        sample_type = threshold_mode = threshold_val = None
        pos_cfg = neg_cfg = None
        if collect_pairs_config is not None:
            if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
                do_collect_dual = True
                pos_cfg = collect_pairs_config.get('pos', None)
                neg_cfg = collect_pairs_config.get('neg', None)
            elif pair_collector is not None:
                do_collect_single = True
                sample_type = collect_pairs_config.get('sample_type', 'neg')
                threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
                threshold_val = collect_pairs_config.get('threshold', 0.5)
        need_collect = (do_collect_single or do_collect_dual)

        pos_hist = torch.zeros(hist_bins, device=device, dtype=torch.int64)
        neg_hist = torch.zeros(hist_bins, device=device, dtype=torch.int64)

        # 行块缓存（low_memory 下避免同一行块重复 H2D）
        cached_bi = -1
        cached_row = None

        while True:
            tile = tile_q.get()
            if tile is None:
                break
            bi, bj, has_pos = tile
            r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
            c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
            is_diag = (bi == bj)

            # 行块：缓存命中复用，否则 H2D（或显存切片）
            if use_gpu_feats:
                block1 = feats_source[r0:r1]
            elif row_cache and bi == cached_bi:
                block1 = cached_row
            else:
                block1 = feats_source[r0:r1].to(device, non_blocking=True)
                if row_cache:
                    cached_bi, cached_row = bi, block1

            # 列块：对角块复用行块，否则 H2D（或显存切片）
            if is_diag:
                block2 = block1
            elif use_gpu_feats:
                block2 = feats_source[c0:c1]
            else:
                block2 = feats_source[c0:c1].to(device, non_blocking=True)

            # matmul
            if use_fp16:
                sim = torch.matmul(block1.half(), block2.half().T).float()
            else:
                sim = torch.matmul(block1, block2.T)     # fp32 / tf32
            sim.clamp_(LO, HI)

            # 身份等值掩码：仅本 tile 含正样本对 或 需要收集样本对 时才计算
            if has_pos or need_collect:
                label_eq = (ids_full[r0:r1, None] == ids_full[None, c0:c1])
            else:
                label_eq = None

            if is_diag:
                triu = torch.triu(
                    torch.ones(r1 - r0, c1 - c0, device=device, dtype=torch.bool),
                    diagonal=1)

            # 单趟直方图前半：full（对角块先把下三角填范围外）
            if use_bincount:
                q = ((sim + 1.0) * scale).floor_().clamp_(0, hist_bins - 1).to(torch.int32)
                if is_diag:
                    full_hist = torch.bincount(q[triu], minlength=hist_bins).to(torch.int64)
                else:
                    full_hist = torch.bincount(q.reshape(-1), minlength=hist_bins).to(torch.int64)
            else:
                if is_diag:
                    sim.masked_fill_(~triu, FILL)
                full_hist = torch.histc(sim, bins=hist_bins, min=LO, max=HI).to(torch.int64)

            # 样本对收集：在 pos 的 in-place masked_fill 之前做，直接用 sim（省掉整份 clone）
            if need_collect:
                if is_diag:
                    valid_mask = triu
                else:
                    valid_mask = torch.ones(r1 - r0, c1 - c0, device=device, dtype=torch.bool)

                if do_collect_single and not pair_collector.is_full():
                    base = label_eq if sample_type == 'pos' else ~label_eq
                    target = (base & valid_mask & (sim > threshold_val)) \
                        if threshold_mode == 'above' else (base & valid_mask & (sim < threshold_val))
                    _collect_pairs(sim, target, r0, c0, pair_collector)
                    del target

                if do_collect_dual:
                    if pos_cfg and pos_pair_collector and not pos_pair_collector.is_full():
                        pt = pos_cfg.get('threshold_mode', 'below')
                        pv = pos_cfg.get('threshold', 0.25)
                        cond = sim > pv if pt == 'above' else sim < pv
                        _collect_pairs(sim, label_eq & valid_mask & cond,
                                       r0, c0, pos_pair_collector)
                    if neg_cfg and neg_pair_collector and not neg_pair_collector.is_full():
                        nt = neg_cfg.get('threshold_mode', 'above')
                        nv = neg_cfg.get('threshold', 0.5)
                        cond = sim > nv if nt == 'above' else sim < nv
                        _collect_pairs(sim, (~label_eq) & valid_mask & cond,
                                       r0, c0, neg_pair_collector)

                del valid_mask

            # 单趟直方图后半：pos + neg = full - pos
            if use_bincount:
                pos_block = None
                if has_pos:
                    sel = (triu & label_eq) if is_diag else label_eq
                    pos_block = torch.bincount(q[sel], minlength=hist_bins).to(torch.int64)
                    del sel
                del q
            else:
                pos_block = None
                if has_pos:
                    sim.masked_fill_(~label_eq, FILL)
                    pos_block = torch.histc(sim, bins=hist_bins, min=LO, max=HI).to(torch.int64)

            if pos_block is not None:
                pos_hist += pos_block
                neg_hist += full_hist - pos_block
                del pos_block
            else:
                neg_hist += full_hist
            del full_hist, sim
            if label_eq is not None:
                del label_eq
            if is_diag:
                del triu

            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

        out[slot] = (pos_hist.cpu(), neg_hist.cpu())


# ============================================================
# 主入口
# ============================================================
def get_sim_matrix_large_scale_v6(
    query_feats_list, query_ids=None, num_gpus=7, block_size=16384,
    hist_bins=200_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None, memory_mode='low_memory', show_progress=True,
    precision='tf32', row_cache=True, pin_inplace=True,
    hist_method='histc', skip_pos_check=True,
):
    """
    大规模相似度矩阵正/负样本直方图（v6，多线程 + low_memory + tf32）。

    Args:
        query_feats_list: (N, D) float32，L2 归一化特征（numpy / list / torch.Tensor）。
        query_ids:        (N,) int64 身份标签；None 时视为全部不同身份。
        num_gpus:         使用的 GPU 数（cuda:0 .. cuda:num_gpus-1）。
        block_size:       分块大小（默认 16384，越大 GEMM 越高效）。
        hist_bins:        直方图 bin 数（默认 200_000，覆盖 hist_range）。
        hist_range:       (min, max)（默认 (-1, 1)，余弦相似度）。
        collect_pairs_config: 可选样本对收集，见 v5 语义：
            {'sample_type':'neg', 'threshold_mode':'above', 'threshold':0.5, 'max_pairs':1000}
            或双模式 {'pos':{...}, 'neg':{...}}。
        memory_mode:      'low_memory'（特征放 CPU 按需 H2D，显存不足用）
                          或 'high_performance'（特征常驻显存，更快但需显存装得下）。
        show_progress:    是否显示进度条（依赖 tqdm）。
        precision:        'fp32'(全精度) / 'tf32'(默认，快 ~12%，本任务实测 0 bin 差) / 'fp16'。
        row_cache:        low_memory 下缓存行块，避免同一行块重复 H2D。
        pin_inplace:      True 用 cudaHostRegister 原地锁定 numpy（零拷贝，省一份内存）；
                          False 用 pin_memory()（拷贝一份到页锁定内存）。
        hist_method:      'histc'（默认，实测更快）或 'bincount'（量化+GPU bincount，备选）。
        skip_pos_check:   静态预标含正样本 tile，热路径对无正样本 tile 免身份等值比对。

    Returns:
        (pos_hist, neg_hist) 两个 np.int64 (hist_bins,)；neg = 全体 - 正样本，逐 bin 非负。
        若给 collect_pairs_config 则额外返回收集到的样本对列表。
    """
    if precision not in ('fp32', 'tf32', 'fp16'):
        raise ValueError(f"precision 必须是 'fp32'/'tf32'/'fp16'，收到 {precision!r}")

    if precision == 'fp16' and hist_bins > 2000:
        print(f"[V6建议] fp16 有效分辨率约 0.001，当前 hist_bins={hist_bins:,} 偏细，"
              f"如需最佳性能可设 2000（当前仍按 {hist_bins:,} 计算）")

    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))
    N = len(query_ids)

    # ---- 准备数据 ----
    if isinstance(query_feats_list, list):
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif isinstance(query_feats_list, np.ndarray):
        query_feats_tensor = torch.from_numpy(query_feats_list)
    else:
        query_feats_tensor = query_feats_list

    registered_ptr = None
    if memory_mode == 'low_memory':
        if query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.cpu()
        if pin_inplace and isinstance(query_feats_list, np.ndarray):
            # 原地锁定 numpy 缓冲区（零拷贝），避免 pin_memory() 的整份 pinned 副本
            arr = np.ascontiguousarray(query_feats_list, dtype=np.float32)
            ptr = arr.ctypes.data
            torch.cuda.cudart().cudaHostRegister(ptr, arr.nbytes, 0)
            registered_ptr = ptr
            query_feats_tensor = torch.from_numpy(arr).float().contiguous()
        else:
            query_feats_tensor = query_feats_tensor.float().contiguous().pin_memory()
    else:
        if not query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.float().pin_memory()

    # ---- 静态预标含正样本 tile（deepseek-pro 式，热路径免身份比对）----
    if skip_pos_check:
        flags = _pos_tile_flags(np.asarray(query_ids), block_size, N)
    else:
        flags = None

    # ---- 建 tile 队列：行块序 + 行内宽优先（利于行块缓存 + 大块优先）----
    nb = math.ceil(N / block_size)
    tiles = []
    for bi in range(nb):
        for bj in range(bi, nb):
            has_pos = (flags is None) or ((bi, bj) in flags)
            tiles.append((bi, bj, has_pos))
    tiles.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    tile_q = queue.Queue()
    for t in tiles:
        tile_q.put(t)
    for _ in range(num_gpus):
        tile_q.put(None)
    n_pos_tiles = sum(1 for t in tiles if t[2]) if flags is not None else len(tiles)
    print(f"V6 tile 队列: 共 {len(tiles)} 块(含正样本 {n_pos_tiles}), block={block_size}, "
          f"precision={precision}, hist={hist_method}, row_cache={row_cache}, mode={memory_mode}")

    # ---- 收集器准备 ----
    pair_collector = pos_pair_collector = neg_pair_collector = None
    is_dual_mode = False
    if collect_pairs_config is not None:
        if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
            is_dual_mode = True
            if 'pos' in collect_pairs_config:
                pos_pair_collector = PairCollector(
                    max_pairs=collect_pairs_config['pos'].get('max_pairs', -1))
            if 'neg' in collect_pairs_config:
                neg_pair_collector = PairCollector(
                    max_pairs=collect_pairs_config['neg'].get('max_pairs', -1))
        else:
            pair_collector = PairCollector(
                max_pairs=collect_pairs_config.get('max_pairs', -1))

    start = time.time()
    pbar = tqdm(total=len(tiles), desc="Matrix Cal v6", disable=not show_progress) \
        if (show_progress and tqdm is not None) else None
    pbar_lock = threading.Lock()

    out = [None] * num_gpus
    threads = []
    try:
        for gpu_id in range(num_gpus):
            th = threading.Thread(
                target=_gpu_worker,
                args=(query_feats_tensor, query_ids, tile_q, gpu_id, block_size, N,
                      hist_bins, hist_range, collect_pairs_config,
                      pair_collector, pos_pair_collector, neg_pair_collector,
                      memory_mode, precision, row_cache, hist_method,
                      pbar, pbar_lock, out, gpu_id),
                name=f'v6-gpu-{gpu_id}',
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    finally:
        if registered_ptr is not None:
            torch.cuda.cudart().cudaHostUnregister(registered_ptr)
    if pbar is not None:
        pbar.close()

    results = []
    for r in out:
        if r is None or isinstance(r, Exception):
            raise RuntimeError(f'V6 worker 失败: {r!r}')
        results.append(r)

    print(f"计算总耗时: {time.time() - start:.2f} 秒")

    for gid in range(num_gpus):
        with torch.cuda.device(gid):
            torch.cuda.empty_cache()

    total_pos_hist = torch.zeros(hist_bins, dtype=torch.int64)
    total_neg_hist = torch.zeros(hist_bins, dtype=torch.int64)
    for p_hist, n_hist in results:
        total_pos_hist += p_hist
        total_neg_hist += n_hist

    print(f"Total Pos Pairs: {total_pos_hist.sum().item()}")
    print(f"Total Neg Pairs: {total_neg_hist.sum().item()}")

    if collect_pairs_config is not None:
        if is_dual_mode:
            pos_collected = pos_pair_collector.get_pairs() if pos_pair_collector else []
            neg_collected = neg_pair_collector.get_pairs() if neg_pair_collector else []
            if pos_collected:
                pos_tmode = collect_pairs_config['pos'].get('threshold_mode', 'below')
                pos_collected.sort(key=lambda x: x[2], reverse=(pos_tmode == 'above'))
                print(f"Collected POS Pairs: {len(pos_collected)}")
            if neg_collected:
                neg_tmode = collect_pairs_config['neg'].get('threshold_mode', 'above')
                neg_collected.sort(key=lambda x: x[2], reverse=(neg_tmode == 'above'))
                print(f"Collected NEG Pairs: {len(neg_collected)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), pos_collected, neg_collected
        else:
            collected_pairs = pair_collector.get_pairs()
            threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
            collected_pairs.sort(key=lambda x: x[2], reverse=(threshold_mode == 'above'))
            print(f"Collected Pairs: {len(collected_pairs)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), collected_pairs

    return total_pos_hist.numpy(), total_neg_hist.numpy()


# ============================================================
# 命令行入口
# ============================================================
if __name__ == '__main__':
    import argparse
    import pickle

    ap = argparse.ArgumentParser(
        description='cluster_utils_v6: 大规模相似度正/负样本直方图评估（多线程 + low_memory + tf32）')
    ap.add_argument('--data', help='输入 pkl: (feats[N,D] fp32 L2归一化, flip, ids[N] int64, paths)')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6', help='逗号分隔 GPU 编号')
    ap.add_argument('--block-size', type=int, default=16384)
    ap.add_argument('--hist-bins', type=int, default=200_000)
    ap.add_argument('--precision', default='tf32', choices=['fp32', 'tf32', 'fp16'])
    ap.add_argument('--memory-mode', default='low_memory',
                    choices=['low_memory', 'high_performance'])
    ap.add_argument('--no-skip-pos', action='store_true', help='关闭正样本 tile 预标跳过')
    args = ap.parse_args()

    if not args.data:
        ap.print_help()
        raise SystemExit(0)

    with open(args.data, 'rb') as f:
        feats, _flip, ids, _paths = pickle.load(f)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)  # L2 归一化
    gpus = [int(g) for g in args.gpus.split(',') if g != '']

    t0 = time.perf_counter()
    pos_hist, neg_hist = get_sim_matrix_large_scale_v6(
        query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
        block_size=args.block_size, hist_bins=args.hist_bins,
        memory_mode=args.memory_mode, precision=args.precision,
        skip_pos_check=not args.no_skip_pos)
    wall = time.perf_counter() - t0
    print(f'\n完成: wall={wall:.2f}s '
          f'正样本对={int(pos_hist.sum()):,} 负样本对={int(neg_hist.sum()):,} '
          f'bins={len(pos_hist)}')
