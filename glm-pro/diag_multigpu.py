#!/usr/bin/env python3
"""诊断多GPU性能瓶颈: 分解传输/正样本/扫描的CPU launch时间与GPU执行时间"""
import time

import numpy as np
import torch

from step3_multi_gpu import load_and_sort, build_equal_area_bounds

PKL = 's4_0618_enhance.pkl'
B, C, nb, cap = 4096, 32768, 262144, 3000000
gpus = list(range(7))

feats_s, ids_c, counts, starts, ends, order, fps = load_and_sort(PKL)
N = len(ids_c)
bounds = build_equal_area_bounds(N, 7)
F_pin = torch.from_numpy(np.ascontiguousarray(feats_s)).pin_memory()
ids_pin = torch.from_numpy(ids_c).pin_memory()
g_assign = np.clip(np.searchsorted(bounds, starts, side='right') - 1, 0, 6)

t_all = time.time()
per_gpu_cpu, streams, evs = [], [], []
for gi, g in enumerate(gpus):
    dev = torch.device(g)
    a, b = bounds[gi], bounds[gi + 1]
    stream = torch.cuda.Stream(device=dev)
    streams.append(stream)
    t0 = time.time()
    nblk = 0
    with torch.cuda.device(dev), torch.cuda.stream(stream):
        e0 = torch.cuda.Event(enable_timing=True); e0.record()
        Fk = F_pin[a:].to(dev, non_blocking=True)
        ids_g = ids_pin.to(dev, non_blocking=True)
        e1 = torch.cuda.Event(enable_timing=True); e1.record()
        # 正样本
        gsel = np.where(g_assign == gi)[0]
        gs, ge = starts[gsel], ends[gsel]
        sizes = ge - gs
        ps = []
        for c in np.unique(sizes):
            if c < 2:
                continue
            gsel2 = np.where(sizes == c)[0]
            X = torch.stack([Fk[gs[k] - a:ge[k] - a] for k in gsel2])
            Gm = torch.bmm(X, X.transpose(1, 2))
            tri = torch.triu_indices(int(c), int(c), offset=1, device=dev)
            ps.append(Gm[:, tri[0], tri[1]].reshape(-1))
        pos_all = torch.cat(ps) if ps else None
        e2 = torch.cuda.Event(enable_timing=True); e2.record()
        # 扫描
        scale = (nb - 1) / 2
        neg_hist = torch.zeros(nb, dtype=torch.int64, device=dev)
        buf_v = buf_i = buf_j = None
        for r0 in range(a, b, B):
            r1 = min(r0 + B, b)
            rows = Fk[r0 - a:r1 - a]
            ids_rows = ids_g[r0:r1]
            ri = torch.arange(r0, r1, device=dev, dtype=torch.int32)
            for c0 in range(r0, N, C):
                c1 = min(c0 + C, N)
                nblk += 1
                S = rows @ Fk[c0 - a:c1 - a].T
                eq = ids_rows[:, None] == ids_g[c0:c1][None, :]
                if c0 < r1:
                    ci = torch.arange(c0, c1, device=dev, dtype=torch.int32)
                    excl = eq | (ci[None, :] <= ri[:, None])
                else:
                    excl = eq
                S.add_(1.0).mul_(scale)
                flat = S.to(torch.int32).clamp_(0, nb - 1).view(-1)
                all_cnt = torch.bincount(flat, minlength=nb)
                ex = flat[excl.view(-1)]
                excl_cnt = (torch.bincount(ex, minlength=nb) if ex.numel()
                            else torch.zeros(nb, dtype=torch.int64, device=dev))
                neg_hist += all_cnt - excl_cnt
                k = min(cap, flat.numel())
                v, p = torch.topk(S.view(-1), k)
                wc = c1 - c0
                bi = r0 + p // wc
                bj = c0 + p % wc
                keep = (bi < bj) & (ids_g[bi] != ids_g[bj])
                v, bi, bj = v[keep], bi[keep].long(), bj[keep].long()
                if v.numel() > 0:
                    if buf_v is None:
                        buf_v, buf_i, buf_j = v, bi, bj
                    else:
                        buf_v = torch.cat([buf_v, v])
                        buf_i = torch.cat([buf_i, bi])
                        buf_j = torch.cat([buf_j, bj])
                        if buf_v.numel() > 2 * cap:
                            sel = torch.topk(buf_v, cap).indices
                            buf_v, buf_i, buf_j = buf_v[sel], buf_i[sel], buf_j[sel]
        e3 = torch.cuda.Event(enable_timing=True); e3.record()
        evs.append((e0, e1, e2, e3))
    per_gpu_cpu.append((time.time() - t0, nblk))
t_launch_total = time.time() - t_all
for s in streams:
    s.synchronize()
t_wall = time.time() - t_all

print(f"\nlaunch循环CPU总时间(不含等待): {t_launch_total:.2f}s | 总墙钟: {t_wall:.2f}s")
for gi, (cpu_t, nblk) in enumerate(per_gpu_cpu):
    e0, e1, e2, e3 = evs[gi]
    print(f"GPU{gi}: 块数={nblk:3d} | CPU launch={cpu_t*1000:6.0f}ms | "
          f"传输={e0.elapsed_time(e1):6.0f}ms | 正样本={e1.elapsed_time(e2):5.0f}ms | "
          f"扫描={e2.elapsed_time(e3):6.0f}ms")
