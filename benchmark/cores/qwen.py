#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen core —— 提炼自 qwen/eval_v2_multi_gpu.py（v2 多卡最终版，总结见 qwen/总结文档.md）

原实现要点（balanced_row_splits / _worker_impl / main）:
    * 按行工作量负载均衡：行 i 的有效样本对数为 (N-1-i)，前缀和
      W(r)=r(N-1)-r(r-1)/2；对每个目标 W = total*k/G 二分求最小 r 使 W(r)>=target，
      把行 [0,N) 静态切成 G 段"行带" —— 各卡样本对数严格相等（实测最大/平均≈1.00002，
      无动态任务队列）
    * 对角块列跳过：行带内每个行块 [start,end) 只与列 [start+1,N) 做矩阵乘
      （列从 start+1 起即不再含任何下三角块），总计算量减半；
      块内仍可能有 j<=i 的格子，用 valid = cols>rows 掩掉，保证严格上三角 i<j；
      每对 (i,j) 在行 i 被处理时恰好统计一次，行带切分不影响计数
    * 块高按剩余列数自适应：chunk = clamp(1e8/(N-start), 16, 8192)，每块 ~1e8 元素
      （峰值显存约 3GB），尾部行块自动加高保持 GPU 利用率
    * fp32 全精度（allow_tf32=False），sim clamp 到 [-1,1]；valid&same / valid&~same
      掩码分别取正/负样本对，idx = ((sim+1)*scale).long().clamp_(0, bins-1) 映射 bin，
      torch.bincount(pos_idx / neg_idx, minlength=bins) 分正/负独立累积直方图
    * 结果一律 .cpu().numpy() 后经 mp.Queue 回传（numpy 按原始字节 pickle；torch 张量
      走 Queue 的 fd 共享存储还原路径，worker 先退出时主进程会 EOFError）
    * 启动方式 fork：父进程不初始化 CUDA 即安全，免去 spawn 每子进程重复 import torch
      （原最终推荐 --mp-method fork）；平台不支持 fork 时自动降级 spawn
      （worker 为顶层函数 + 共享内存张量，天然 spawn 安全）

统一网格适配：原 nbins=2,000,000（bin 宽 1e-6）；映射 idx=floor((sim+1)*bins/2) 与 bins
成正比，直接取 bins=common.BINS=200,000 即精确落到统一网格
[-1+k*1e-5, -1+(k+1)*1e-5)，bin 语义与 common 完全一致，无需 rebin。
已去除样本对提取/绘图/argparse/直方图存盘等非核心。
"""
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'qwen'
MODEL_DESC = ('Qwen v2 最终版方法：fork每GPU一进程 + 按行工作量(N-1-i)前缀和静态均衡行带'
              ' + 对角块列跳过 + 自适应块高 + GPU bincount 分正/负直方图')
ORIGIN = ('qwen/eval_v2_multi_gpu.py (balanced_row_splits / _worker_impl / main；'
          'mp-method=fork；bins 2M -> 统一 200K，去掉提取/绘图)')

CHUNK_MAX = 8192          # 自适应块高上限（原值）
CHUNK_MIN = 16            # 自适应块高下限（原值）
CHUNK_ELEMS = int(1e8)    # 每块目标元素数（原值，~1e8 元素/块，峰值显存约 3GB）


# ---------------------------------------------------------------- 负载均衡划分
def _prefix_work(r, N):
    """前 r 行（行 0..r-1）的有效样本对数：W(r) = sum_{i<r}(N-1-i)"""
    return r * (N - 1) - r * (r - 1) // 2


def balanced_row_splits(N, G):
    """按行工作量均分行带。行 i 有效工作量 (N-1-i) 对，前缀和 W(r)；对每个
    k in 1..G-1 求最小 r 使 W(r) >= total*k/G（二分），返回 G+1 个切分点。
    原样取自 v2。"""
    total = N * (N - 1) // 2
    splits = [0]
    for k in range(1, G):
        target = total * k / G
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            w = mid * (N - 1) - mid * (mid - 1) // 2
            if w < target:
                lo = mid + 1
            else:
                hi = mid
        splits.append(lo)
    splits.append(N)
    return splits


# ---------------------------------------------------------------- GPU worker
def worker(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, queue):
    """fork/spawn 子进程入口：计算行带 [r0,r1) 内全部上三角样本对的正/负直方图，
    结果转 numpy 入队；任何异常以 error dict 入队（防止主进程 queue.get 挂死）。"""
    try:
        _worker_impl(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, queue)
    except Exception as e:
        queue.put(dict(rank=rank, gpu=gpu_id, error=f'{type(e).__name__}: {e}'))


def _worker_impl(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, queue):
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 全精度
    torch.set_num_threads(1)
    device = torch.device(f'cuda:{gpu_id}')
    N = feats_sh.shape[0]
    feats = feats_sh.to(device)                     # 全量特征驻留本卡 (N*512*4B)
    ids_dev = ids_sh.to(device)

    hist_pos = torch.zeros(nbins, dtype=torch.long, device=device)
    hist_neg = torch.zeros(nbins, dtype=torch.long, device=device)
    scale = nbins / 2.0                             # (sim+1)*scale -> [0, bins]

    total_rows = r1 - r0
    t0 = time.time()
    last_report = t0

    start = r0
    while start < r1:
        # 对角块列跳过：行块 [start,end) 只需与列 [start+1,N) 计算 —— 总计算量减半，
        # 且各卡元素数 == 各卡样本对数，负载均衡真正成立
        c0 = start + 1
        if c0 >= N:
            break
        # 块高按剩余列数自适应，保持每块 ~1e8 元素（尾部行块可以更高）
        chunk = max(CHUNK_MIN, min(CHUNK_MAX, int(CHUNK_ELEMS // max(N - start, 1))))
        end = min(start + chunk, r1)
        sim = feats[start:end] @ feats[c0:N].t()    # (c, N-c0) fp32
        sim.clamp_(common.LO, common.HI)
        rows = torch.arange(start, end, device=device).unsqueeze(1)
        cols = torch.arange(c0, N, device=device).unsqueeze(0)
        valid = cols > rows                         # 上三角 i<j
        same = ids_dev[start:end].unsqueeze(1) == ids_dev[c0:N].unsqueeze(0)
        pos_mask = valid & same
        neg_mask = valid & ~same

        # 先按掩码取相似度（数量约为块元素一半），再映射 bin，省显存
        pos_idx = ((sim[pos_mask] + 1.0) * scale).long().clamp_(0, nbins - 1)
        neg_idx = ((sim[neg_mask] + 1.0) * scale).long().clamp_(0, nbins - 1)
        hist_pos += torch.bincount(pos_idx, minlength=nbins)
        hist_neg += torch.bincount(neg_idx, minlength=nbins)

        now = time.time()
        if now - last_report > 5 or end == r1:
            last_report = now
            print(f"  [GPU{gpu_id}] 进度 {end - r0}/{total_rows} 行 "
                  f"({(end - r0) / max(total_rows, 1) * 100:.0f}%)  "
                  f"{now - t0:.1f}s", flush=True)
        del sim, valid, same, pos_mask, neg_mask, pos_idx, neg_idx
        start = end

    t_compute = time.time() - t0
    # 结果一律转 numpy 再入队（原因见模块 docstring）
    result = dict(rank=rank, gpu=gpu_id, rows=(r0, r1), t_compute=t_compute,
                  hist_pos=hist_pos.cpu().numpy(), hist_neg=hist_neg.cpu().numpy())
    queue.put(result)
    print(f"  [GPU{gpu_id}] 完成: 行[{r0},{r1}) 共{total_rows}行 耗时 {t_compute:.2f}s",
          flush=True)


# ---------------------------------------------------------------- 统一入口
def compute(feats, ids, gpus, workdir):
    """契约 compute(feats, ids, gpus, workdir) -> (pos_hist, neg_hist, meta)。

    qwen v2 最终版提炼：fork 每卡一进程，按行工作量前缀和静态划分行带，
    对角块列跳过 + 自适应块高分块矩阵乘，GPU bincount 分正/负累积直方图。
    """
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    G = len(gpus)
    nbins = common.BINS
    if G == 0:
        raise ValueError('gpus 为空，至少需要一张卡')
    t_all = time.perf_counter()

    # 理论正样本对数（合并后自检打印用；框架随后仍会严格校验）
    _, counts = np.unique(ids, return_counts=True)
    expect_pos = int((counts.astype(np.int64) * (counts - 1) // 2).sum())

    # 1) 按行工作量负载均衡划分行带
    splits = balanced_row_splits(N, G)
    works = [_prefix_work(splits[k + 1], N) - _prefix_work(splits[k], N)
             for k in range(G)]
    print(f"[qwen] N={N} gpus={list(gpus)} 行带划分: "
          f"{list(zip(splits[:-1], splits[1:]))}", flush=True)
    print(f"[qwen] 各卡样本对数: {[f'{w:,}' for w in works]} "
          f"(最大/平均={max(works) / (sum(works) / G):.3f})", flush=True)

    # 2) 父进程建共享内存张量（fork 直接继承 / spawn 按共享引用还原），父进程不碰 CUDA
    feats_sh = torch.from_numpy(feats).share_memory_()
    ids_sh = torch.from_numpy(ids).share_memory_()
    del feats, ids

    # 3) fork 启动（免重复 import torch）；平台不支持 fork 时降级 spawn
    try:
        ctx = mp.get_context('fork')
        mp_method = 'fork'
    except ValueError:
        ctx = mp.get_context('spawn')
        mp_method = 'spawn'
    queue = ctx.Queue()
    procs = []
    for rank, gid in enumerate(gpus):
        p = ctx.Process(target=worker,
                        args=(rank, gid, feats_sh, ids_sh, splits[rank],
                              splits[rank + 1], nbins, queue))
        p.start()
        procs.append(p)

    results = [queue.get() for _ in range(G)]       # 先取完结果再 join，避免管道死锁
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t_all

    for r in results:
        if 'error' in r:
            raise RuntimeError(f"GPU{r['gpu']} worker 失败:\n{r['error']}")

    # 4) 按 rank 合并直方图（统一网格 200K bins，int64）
    results.sort(key=lambda r: r['rank'])
    hist_pos = np.zeros(nbins, dtype=np.int64)
    hist_neg = np.zeros(nbins, dtype=np.int64)
    t_computes = []
    for r in results:
        hist_pos += r['hist_pos']
        hist_neg += r['hist_neg']
        t_computes.append(r['t_compute'])
    n_pos, n_neg = int(hist_pos.sum()), int(hist_neg.sum())
    total = N * (N - 1) // 2
    print(f"[qwen] 合并: 正 {n_pos:,}（理论 {expect_pos:,}，"
          f"{'一致' if n_pos == expect_pos else '不一致!'}）"
          f" 负 {n_neg:,} 总 {n_pos + n_neg:,}（理论 {total:,}）", flush=True)
    print(f"[qwen] worker 墙钟 avg={sum(t_computes) / G:.2f}s "
          f"max={max(t_computes):.2f}s", flush=True)

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': (f'{mp_method} 每GPU一进程 + 按行工作量前缀和静态划分行带'
                     f'（无动态队列，各卡样本对数均衡）'),
        'block': CHUNK_MAX,
        'blocking': (f'行块高自适应 {CHUNK_MIN}..{CHUNK_MAX} 行'
                     f'（≈{CHUNK_ELEMS} 元素/块）+ 对角块列跳过 [start+1,N)'),
        'precision': 'fp32 (allow_tf32=False) + GPU bincount(int64)',
        'native_bins': 2_000_000,     # 原实现直方图精度；映射与 bins 成正比，直接落统一 200K 网格
        'splits': [list(s) for s in zip(splits[:-1], splits[1:])],
        'pairs_per_gpu': works,
        'core_s': round(core_s, 3),
        'worker_s_avg': round(float(sum(t_computes) / G), 3),
        'worker_s_max': round(float(max(t_computes)), 3),
    }
    return hist_pos, hist_neg, meta
