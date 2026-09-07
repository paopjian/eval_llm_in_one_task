#!/usr/bin/env python3
"""GPU kernel基准测试：确定分块大小、直方图、topk方案"""
import time
import torch

dev = 'cuda:0'
torch.backends.cuda.matmul.allow_tf32 = False

N, D = 203234, 512
F_gpu = torch.randn(N, D, device=dev)
F_gpu = F_gpu / F_gpu.norm(dim=1, keepdim=True)

def bench(fn, name, repeat=5):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / repeat * 1000
    print(f"  {name}: {dt:.2f} ms")
    return dt

print("== 1. matmul 分块速度 (fp32 strict) ==")
for B, C in [(2048, 16384), (4096, 16384), (4096, 32768), (8192, 16384)]:
    A = F_gpu[:B]
    Bm = F_gpu[:C].T.contiguous()
    dt = bench(lambda: A @ Bm, f"matmul {B}x{C} ({B*C*512*2/1e12:.2f} TFLOP)")
    print(f"    -> {B*C*512*2/1e12/dt*1000:.1f} TFLOPS")

print("\n== 2. bincount 支持的dtype ==")
x = torch.randint(0, 262144, (10,), device=dev, dtype=torch.int32)
try:
    r = torch.bincount(x, minlength=262144)
    print("  int32: OK")
except Exception as e:
    print(f"  int32: FAIL {e}")

print("\n== 3. 量化+bincount (负样本直方图核心) ==")
NB = 262144
scale = NB / 2.0
for B, C in [(4096, 32768)]:
    S = (F_gpu[:B] @ F_gpu[:C].T).clamp_(-1, 1)  # (B,C) fp32
    def hist_int64():
        idx = ((S + 1) * scale).long().clamp_(0, NB - 1)
        return torch.bincount(idx.view(-1), minlength=NB)
    def hist_int32():
        idx = ((S + 1) * scale).to(torch.int32).clamp_(0, NB - 1)
        return torch.bincount(idx.view(-1), minlength=NB)
    bench(hist_int64, f"quantize fp32->int64 + bincount {B}x{C}")
    try:
        bench(hist_int32, f"quantize fp32->int32 + bincount {B}x{C}")
    except Exception as e:
        print(f"    int32 bincount FAIL: {e}")

print("\n== 4. masked_select / 比较开销 ==")
B, C = 4096, 32768
S = (F_gpu[:B] @ F_gpu[:C].T)
ids = torch.randint(0, 9940, (N,), device=dev, dtype=torch.int32)
bench(lambda: ids[:B][:, None] == ids[:C][None, :], f"eq mask {B}x{C}")
m = (ids[:B][:, None] == ids[:C][None, :])
bench(lambda: S[m], f"masked_select {B}x{C} (全选)")
bench(lambda: S > 0.5, f"compare {B}x{C}")

print("\n== 5. topk 速度 (提取用) ==")
flat = S.view(-1)
for k in [100_000, 1_000_000]:
    bench(lambda: torch.topk(flat, k), f"topk k={k:,} n={flat.numel():,}")

print("\n== 6. 三角mask开销 ==")
ri = torch.arange(B, device=dev)
ci = torch.arange(C, device=dev)
bench(lambda: ci[None, :] > ri[:, None], f"tri mask {B}x{C}")

print("\n== 7. 全流程单块模拟 (matmul+mask+bincount) ==")
B, C = 4096, 32768
rows = F_gpu[:B]
ids_rows = ids[:B]
def full_block():
    S = rows @ F_gpu[:C].T
    eq = ids_rows[:, None] == ids[:C][None, :]
    neg = ~eq
    S = S.clamp_(-1, 1)
    idx = ((S + 1) * scale).to(torch.int32)
    idx.clamp_(0, NB - 1)
    cnt = torch.bincount(idx.view(-1)[neg.view(-1)], minlength=NB)
    return cnt
bench(full_block, "full block (matmul+eq+bincount via masked_select)")

def full_block_sub():
    S = rows @ F_gpu[:C].T
    eq = ids_rows[:, None] == ids[:C][None, :]
    S = S.clamp_(-1, 1)
    idx = ((S + 1) * scale).to(torch.int32)
    idx.clamp_(0, NB - 1)
    all_cnt = torch.bincount(idx.view(-1), minlength=NB)
    excl_cnt = torch.bincount(idx.view(-1)[eq.view(-1)], minlength=NB)
    return all_cnt - excl_cnt
bench(full_block_sub, "full block (bincount all - bincount eq)")

print("\n== 8. bmm 正样本按组批算 ==")
G, c = 450, 22
X = torch.randn(G, c, D, device=dev)
X = X / X.norm(dim=2, keepdim=True)
bench(lambda: torch.bmm(X, X.transpose(1, 2)), f"bmm {G}x{c}x{c}")

print("\n== 9. tf32对比 ==")
torch.backends.cuda.matmul.allow_tf32 = True
A = F_gpu[:4096]
Bm = F_gpu[:32768].T.contiguous()
bench(lambda: A @ Bm, "matmul 4096x32768 tf32")
torch.backends.cuda.matmul.allow_tf32 = False
