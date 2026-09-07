#!/usr/bin/env python3
"""测试histc/scatter_add是否异步 + 速度对比 + 大块matmul效率"""
import time
import numpy as np
import torch

dev = torch.device(0)
N, D, B, C = 203234, 512, 4096, 32768
torch.manual_seed(0)
F_gpu = torch.randn(N, D, device=dev)
F_gpu = F_gpu / F_gpu.norm(dim=1, keepdim=True)
S = (F_gpu[:B] @ F_gpu[:C].T)  # (B,C) fp32
S.add_(1.0).mul_(0.5)  # 量化到[0,1]域
print(f"S: {S.shape}, 值域[{S.min().item():.3f}, {S.max().item():.3f}]")

def bench(fn, name, repeat=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / repeat * 1000
    print(f"  {name}: {dt:.2f} ms")
    return dt

# --- 同步性检测 ---
import sys
print("\n== 同步性检测 (sync_debug_mode=warn) ==")
torch.cuda.set_sync_debug_mode(1)
for name, fn in [
    ("histc 4096 bins", lambda: torch.histc(S, bins=4096, min=0.0, max=1.0)),
    ("histc 65536 bins", lambda: torch.histc(S, bins=65536, min=0.6, max=1.0)),
    ("scatter_add_ int64", lambda: torch.zeros(4097, dtype=torch.int64, device=dev).scatter_add_(
        0, (S * 4096).long().clamp(0, 4096).view(-1), torch.ones(S.numel(), dtype=torch.int64, device=dev))),
]:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
        torch.cuda.synchronize()
        sync_warned = any('synchron' in str(x.message).lower() for x in w)
    print(f"  {name}: {'同步!' if sync_warned else '异步 ✓'}")
torch.cuda.set_sync_debug_mode(0)

# --- 速度 ---
print("\n== 速度对比 (B=4096, C=32768, 1.34e8元素) ==")
bench(lambda: torch.histc(S, bins=4096, min=0.0, max=1.0), "histc 4096 bins (全值域)")
bench(lambda: torch.histc(S, bins=65536, min=0.6, max=1.0), "histc 65536 bins (尾部[0.2,1]sim域)")

def scatter_ver():
    idx = (S * 65535).long().clamp(0, 65535).view(-1)
    out = torch.zeros(65536, dtype=torch.int64, device=dev)
    out.scatter_add_(0, idx, torch.ones_like(idx))
    return out
bench(scatter_ver, "scatter_add_ 65536 bins (含long转换+ones)")

# 两次histc vs 一次bincount参考
idx32 = (S * 65535).to(torch.int32)
bench(lambda: torch.bincount(idx32.view(-1), minlength=65536), "bincount 65536 (同步, 仅参考)")

# --- histc精度验证: 计数与bincount一致? ---
h = torch.histc(S, bins=4096, min=0.0, max=1.0)
bc = torch.bincount((S * 4096).long().clamp(0, 4095).view(-1), minlength=4096)
diff = (h.long() - bc).abs().max().item()
print(f"\n  histc vs bincount 最大计数差: {diff} (应为0)")

# --- 大块matmul ---
print("\n== 大块matmul效率 (fp32 strict) ==")
for BB, CC in [(4096, 32768), (4096, 65536), (8192, 32768), (8192, 65536), (6144, 49152)]:
    A = F_gpu[:BB]
    Bm = F_gpu[:CC].T.contiguous()
    dt = bench(lambda: A @ Bm, f"matmul {BB}x{CC} ({BB*CC*4/1e9:.2f}GB)", repeat=5)
    print(f"    -> {BB*CC*512*2/1e12/dt*1000:.1f} TFLOPS")

# --- 精度: fp32块内计数上限 (histc用fp32累计, 需<2^24) ---
print(f"\nfp32精确计数上限 2^24 = {2**24:,} | 单块元素 {B*C:,} | 块内单bin计数远小于上限 ✓")
