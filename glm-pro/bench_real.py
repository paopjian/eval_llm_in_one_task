#!/usr/bin/env python3
"""用真实特征数据测histc 65536/scatter_add/单块全流程耗时(排除bench randn钳位热点干扰)"""
import time
import warnings

import numpy as np
import torch

dev = torch.device(0)
B, C = 8192, 65536

import pickle
with open('s4_0618_enhance.pkl', 'rb') as f:
    feats, _, ids, _ = pickle.load(f)
feats = np.asarray(feats, dtype=np.float32)
ids = np.asarray(ids)
order = np.argsort(ids, kind='stable')
F = torch.from_numpy(np.ascontiguousarray(feats[order])).to(dev)
ids_c = torch.from_numpy(np.unique(ids[order], return_inverse=True)[1].astype(np.int32)).to(dev)

# 真实中间块: 行[100000, 100000+B), 列[100000, 100000+C)
r0 = 100000
S = F[r0:r0 + B] @ F[r0:r0 + C].T
print(f"真实S: {S.shape}, 值域[{S.min().item():.3f}, {S.max().item():.3f}], "
      f"sim>=0.2占比 {float((S >= 0.2).float().mean()):.1%}")

def bench(fn, name, repeat=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / repeat * 1000
    print(f"  {name}: {dt:.2f} ms")
    return dt

bench(lambda: F[r0:r0 + B] @ F[r0:r0 + C].T, "matmul 8192x65536")
bench(lambda: torch.histc(S, bins=4096, min=-1.0, max=1.0), "histc 4096 (真实)")
bench(lambda: torch.histc(S, bins=65536, min=0.2, max=1.0), "histc 65536 (真实)")

def scatter_ver():
    idx = ((S.view(-1) - 0.2) * 81920.0).long()
    idx.clamp_(-1, 65536).add_(1)                    # [-1溢出, 0..65536有效+, 65537溢出]
    tmp = torch.zeros(65538, dtype=torch.int64, device=dev)
    tmp.scatter_add_(0, idx, torch.ones(idx.shape[0], dtype=torch.int64, device=dev))
    return tmp
bench(scatter_ver, "scatter_add 65536 (含量化)")

# 验证scatter_add与histc计数一致
h = torch.histc(S, bins=65536, min=0.2, max=1.0).long()
sv = scatter_ver()[1:65537]
print(f"  histc vs scatter_add 最大计数差: {(h - sv).abs().max().item()} (应为0)")
print(f"  总数: histc={h.sum().item():,} scatter={sv.sum().item():,}")

# sync check
torch.cuda.set_sync_debug_mode(1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    scatter_ver()
    torch.cuda.synchronize()
    print(f"  scatter_add警告: {[str(x.message)[:60] for x in w]}")
torch.cuda.set_sync_debug_mode(0)

# 单块全流程(当前实现): matmul+eq+excl+clamp+fill+2×histc+topk
def full_block():
    rows = F[r0:r0 + B]
    ids_rows = ids_c[r0:r0 + B]
    tri_tpl = torch.ones(B, C, dtype=torch.bool, device=dev).tril_()
    Sx = rows @ F[r0:r0 + C].T
    eq = ids_rows[:, None] == ids_c[r0:r0 + C][None, :]
    excl = eq | tri_tpl
    Sx.clamp_(-1.0, 1.0).masked_fill_(excl, -2.0)
    flatS = Sx.view(-1)
    nc = torch.histc(flatS, bins=4096, min=-1.0, max=1.0).long()
    nf = torch.histc(flatS, bins=65536, min=0.2, max=1.0).long()
    v, p = torch.topk(flatS, 1000000)
    return nc, nf, v, p
bench(full_block, "单块全流程(当前实现)")
