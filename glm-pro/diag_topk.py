#!/usr/bin/env python3
"""测试topk/cat/matmul等每块op的异步性: launch N个op后看CPU是否需要等待"""
import time
import warnings

import numpy as np
import torch

dev = torch.device(0)
B, C = 8192, 65536
S = torch.randn(B, C, device=dev).clamp_(-1, 1)

def test_async(name, fn, n=10):
    """launch n次fn, 测CPU时间(不含synchronize) vs 总时间(含)"""
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    t_cpu = time.time() - t0          # CPU launch时间(异步则很小)
    torch.cuda.synchronize()
    t_wall = time.time() - t0         # 含GPU执行
    print(f"  {name}: CPU launch={t_cpu/n*1000:7.2f}ms/op, 含执行={t_wall/n*1000:7.2f}ms/op "
          f"{'[异步✓]' if t_cpu/n < t_wall/n*0.5 else '[疑似同步!]'}")

print("== 每块核心op的异步性 (B=8192, C=65536) ==")
test_async("matmul", lambda: torch.mm(S, S[:512, :512].T if False else S[:C//16, :512].T.contiguous()) if False else (S[:B//16] @ S[:512, :].T))
test_async("eq比较", lambda: (S[:1, :].long() == S[:1, :].long()))
test_async("clamp_+masked_fill_", lambda: S[:B//4].clamp_(-1, 1).masked_fill_(S[:B//4] > 2, -2.0))
test_async("histc 4096", lambda: torch.histc(S, bins=4096, min=-1.0, max=1.0))
test_async("histc 65536", lambda: torch.histc(S, bins=65536, min=0.2, max=1.0))
test_async("topk k=1e6", lambda: torch.topk(S.view(-1), 1000000))
test_async("topk k=1e5", lambda: torch.topk(S.view(-1), 100000))
test_async("cat 2e6", lambda: torch.cat([S.view(-1)[:1000000], S.view(-1)[:1000000]]))

# sync_debug模式交叉验证topk
print("\n== sync_debug_mode验证 ==")
torch.cuda.set_sync_debug_mode(1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    v, p = torch.topk(S.view(-1), 1000000)
    torch.cuda.synchronize()
    print(f"  topk警告: {[str(x.message)[:80] for x in w]}")
torch.cuda.set_sync_debug_mode(0)
