#!/usr/bin/env python3
"""诊断v2: 无boolean索引的新扫描逻辑 + sync_debug_mode定位隐式同步点 + 每块CPU耗时"""
import time
import warnings

import numpy as np
import torch

from step3_multi_gpu import load_and_sort, build_equal_area_bounds

PKL = 's4_0618_enhance.pkl'
B, C, nb, cap = 4096, 32768, 262144, 3000000
gpus = [0, 1, 2, 3, 4, 5, 6]

feats_s, ids_c, counts, starts, ends, order, fps = load_and_sort(PKL)
N = len(ids_c)
bounds = build_equal_area_bounds(N, 7)
F_pin = torch.from_numpy(np.ascontiguousarray(feats_s)).pin_memory()
ids_pin = torch.from_numpy(ids_c).pin_memory()

# 只测单卡(GPU6, 块最多的卡)的扫描: 分解每块CPU时间
g = 6
dev = torch.device(g)
a, b = bounds[g], bounds[g + 1]
stream = torch.cuda.Stream(device=dev)
Fk_cpu = F_pin[a:]

t0 = time.time()
with torch.cuda.device(dev), torch.cuda.stream(stream):
    Fk = Fk_cpu.to(dev, non_blocking=True)
    ids_g = ids_pin.to(dev, non_blocking=True)
    neg_hist = torch.zeros(nb, dtype=torch.int64, device=dev)
stream.synchronize()
print(f"传输+初始化: {time.time()-t0:.3f}s")

# 开启同步警告
torch.cuda.set_sync_debug_mode(1)
block_cpu_times = []
t_total_launch = time.time()
with torch.cuda.device(dev), torch.cuda.stream(stream):
    scale = (nb - 1) / 2.0
    buf_v = buf_i = buf_j = None
    nblk = 0
    for r0 in range(a, b, B):
        r1 = min(r0 + B, b)
        rows = Fk[r0 - a:r1 - a]
        ids_rows = ids_g[r0:r1]
        ri = torch.arange(r0, r1, device=dev, dtype=torch.int32)
        for c0 in range(r0, N, C):
            c1 = min(c0 + C, N)
            tb = time.time()
            S = rows @ Fk[c0 - a:c1 - a].T
            eq = ids_rows[:, None] == ids_g[c0:c1][None, :]
            if c0 < r1:
                ci = torch.arange(c0, c1, device=dev, dtype=torch.int32)
                excl = eq | (ci[None, :] <= ri[:, None])
            else:
                excl = eq
            S.add_(1.0).mul_(scale)
            S.masked_fill_(excl, -1.0)
            idx = S.to(torch.int32)
            idx.clamp_max_(nb).add_(1)
            cnt = torch.bincount(idx.view(-1), minlength=nb + 1)
            neg_hist += cnt[1:]
            k = min(cap, idx.numel())
            v, p = torch.topk(S.view(-1), k)
            wc = c1 - c0
            if buf_v is None:
                buf_v, buf_i, buf_j = v, p // wc + r0, p % wc + c0
            else:
                buf_v = torch.cat([buf_v, v])
                buf_i = torch.cat([buf_i, p // wc + r0])
                buf_j = torch.cat([buf_j, p % wc + c0])
            if buf_v.numel() > 2 * cap:
                sel = torch.topk(buf_v, cap).indices
                buf_v, buf_i, buf_j = buf_v[sel], buf_i[sel], buf_j[sel]
            block_cpu_times.append(time.time() - tb)
            nblk += 1
torch.cuda.set_sync_debug_mode(0)
t_launch = time.time() - t_total_launch
stream.synchronize()
t_wall = time.time() - t_total_launch

bt = np.array(block_cpu_times) * 1000
print(f"\nGPU{g}: 块数={nblk}, CPU launch总时间={t_launch:.2f}s, 墙钟(含kernel)={t_wall:.2f}s")
print(f"每块CPU耗时: 均值={bt.mean():.1f}ms 中位={np.median(bt):.1f}ms 最大={bt.max():.1f}ms")
print(f"CPU耗时分解: 前5块={bt[:5].round(1)}, 后5块={bt[-5:].round(1)}")
