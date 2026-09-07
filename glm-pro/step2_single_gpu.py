#!/usr/bin/env python3
"""
第二步：单GPU分块上三角相似度计算 + TPIR@FPIR指标

核心方法（应对206亿样本对，不保存全量相似度）:
- 按id稳定排序 → 同身份样本连续，正样本位置可用 O(B×C) 广播比较排除
- 行块×列块 matmul，列块从行块起点开始 → 只计算上三角区域
- 负样本(206亿对): 用直方图统计分布(262144 bins, [-1,1], bin宽7.6e-6)
                    + top-N 高相似度对提取(带索引, 用于错误分析)
- 正样本(218万对): 按身份组"同组大小分桶"批量bmm, 全量精确保存
- TPIR@FPIR: 负样本直方图/精确top-N求阈值, 正样本精确值求TPIR
"""
import argparse
import pickle
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description='单卡人脸特征相似度评估')
    p.add_argument('--pkl', default='s4_0618_enhance.pkl')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--block-rows', type=int, default=4096, help='行块大小')
    p.add_argument('--block-cols', type=int, default=32768, help='列块大小')
    p.add_argument('--hist-bins', type=int, default=262144, help='负样本直方图bin数(覆盖[-1,1])')
    p.add_argument('--extract-cap', type=int, default=1000000, help='提取的最高相似度负样本对数量上限')
    p.add_argument('--save', default='result_step2.npz', help='中间统计结果保存路径')
    return p.parse_args()


def load_and_sort(path):
    """读取pkl并按id稳定排序(同身份连续), 返回排序后特征/紧凑id/原始索引映射"""
    with open(path, 'rb') as f:
        feats, _, ids, file_paths = pickle.load(f)
    feats = np.asarray(feats, dtype=np.float32)
    ids = np.asarray(ids)

    order = np.argsort(ids, kind='stable')
    feats_s = feats[order]
    ids_s = ids[order]
    # 紧凑重映射id -> 0..G-1 (排序后天然分组连续)
    _, ids_c = np.unique(ids_s, return_inverse=True)
    counts = np.bincount(ids_c)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ends = np.cumsum(counts).astype(np.int64)
    return feats_s, ids_c.astype(np.int32), counts, starts, ends, order, file_paths


def compute_positives(F_cpu, starts, ends, dev):
    """正样本对: 按组大小分桶批量bmm, 返回 (sims, i, j) 排序后索引"""
    sizes = ends - starts
    sim_list, i_list, j_list = [], [], []
    for c in np.unique(sizes):
        if c < 2:
            continue
        g = np.where(sizes == c)[0]
        X = torch.stack([F_cpu[starts[k]:ends[k]] for k in g]).to(dev)  # (G,c,512)
        Gm = torch.bmm(X, X.transpose(1, 2))                              # (G,c,c)
        tri = torch.triu_indices(int(c), int(c), offset=1, device=dev)    # (2,P)
        vals = Gm[:, tri[0], tri[1]]                                      # (G,P)
        base = torch.from_numpy(starts[g]).to(dev)                        # (G,)
        i_list.append((base[:, None] + tri[0][None, :]).reshape(-1).cpu())
        j_list.append((base[:, None] + tri[1][None, :]).reshape(-1).cpu())
        sim_list.append(vals.reshape(-1).cpu())
    return (torch.cat(sim_list).numpy().astype(np.float32),
            torch.cat(i_list).numpy().astype(np.int64),
            torch.cat(j_list).numpy().astype(np.int64))


def sweep_negatives(F_cpu, ids_c, N, dev, B, C, nb, cap, verbose=True):
    """负样本对扫描: 分块上三角matmul + 直方图 + top-N提取"""
    scale = (nb - 1) / 2.0
    F_gpu = F_cpu.to(dev)
    ids_gpu = torch.from_numpy(ids_c).to(dev)
    neg_hist = torch.zeros(nb, dtype=torch.int64, device=dev)

    # top-N running buffer (在GPU上增量合并)
    buf_v = buf_i = buf_j = None
    t_sweep = 0.0
    n_blocks = 0
    t0 = time.time()
    for r0 in range(0, N, B):
        r1 = min(r0 + B, N)
        rows = F_gpu[r0:r1]
        ids_rows = ids_gpu[r0:r1]
        ri = torch.arange(r0, r1, device=dev, dtype=torch.int32)
        for c0 in range(r0, N, C):
            c1 = min(c0 + C, N)
            S = rows @ F_gpu[c0:c1].T                       # (b,c) fp32, [-1,1]
            if c0 < r1:  # 列块与行块重叠, 需排除下三角(含对角线)
                ci = torch.arange(c0, c1, device=dev, dtype=torch.int32)
                excl = (ids_rows[:, None] == ids_gpu[c0:c1][None, :]) | (ci[None, :] <= ri[:, None])
            else:
                excl = ids_rows[:, None] == ids_gpu[c0:c1][None, :]
            # 就地量化: S -> [0, nb-1] 的float, 与原相似度单调等价(用于topk)
            S.add_(1.0).mul_(scale)
            idx = S.to(torch.int32).clamp_(0, nb - 1)
            flat = idx.view(-1)
            all_cnt = torch.bincount(flat, minlength=nb)
            excl_cnt = torch.bincount(flat[excl.view(-1)], minlength=nb)
            neg_hist += all_cnt - excl_cnt

            # top-N: 块内topk(量化值与原值单调等价, 排名一致)
            # 注意: topk结果中混有正样本对和下三角位置, 需过滤出真正的负样本对(i<j且id不同)
            k = min(cap, flat.numel())
            v, p = torch.topk(S.view(-1), k)
            wc = c1 - c0
            bi = r0 + p // wc
            bj = c0 + p % wc
            keep = (bi < bj) & (ids_gpu[bi] != ids_gpu[bj])
            v, bi, bj = v[keep], bi[keep], bj[keep]
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
            n_blocks += 1
            if verbose and n_blocks % 20 == 0:
                torch.cuda.synchronize()
                print(f"    已完成 {n_blocks} 块, 耗时 {time.time()-t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    t_sweep = time.time() - t0

    if buf_v is None or buf_v.numel() == 0:
        return neg_hist.cpu().numpy(), np.zeros(0, np.float32), np.zeros(0, np.int64), \
               np.zeros(0, np.int64), t_sweep, n_blocks
    sel = torch.topk(buf_v, min(cap, buf_v.numel())).indices
    top_v = (buf_v[sel] / scale - 1.0).cpu().numpy().astype(np.float32)  # 反量化
    top_i = buf_i[sel].cpu().numpy().astype(np.int64)
    top_j = buf_j[sel].cpu().numpy().astype(np.int64)
    return neg_hist.cpu().numpy(), top_v, top_i, top_j, t_sweep, n_blocks


def tpir_at_fpir(pos_sims, neg_hist, nb, targets):
    """TPIR@FPIR: 返回 {fpir: (tpir, threshold, 方法)}"""
    pos_sorted = np.sort(pos_sims)          # 升序
    n_pos, n_neg = len(pos_sorted), int(neg_hist.sum())
    bin_left = 2.0 * np.arange(nb) / (nb - 1) - 1.0
    cum_above = np.cumsum(neg_hist[::-1])[::-1]  # cum_above[b] = #{sim >= bin_left[b]}

    out = {}
    for f in targets:
        # 直方图法: cum_above递减, 找最大b使 cum_above[b] >= f*n_neg (阈值取bin左边界)
        below = cum_above < f * n_neg
        if below.any():
            b = int(np.argmax(below)) - 1  # cum_above[b] >= target > cum_above[b+1]
        else:
            b = nb - 1
        t = bin_left[b]
        # TPIR = #{pos > t} / n_pos
        tpir = 1.0 - np.searchsorted(pos_sorted, t, side='right') / n_pos
        out[f] = (tpir, t, 'hist')
    return out


def main():
    args = parse_args()
    t_total = time.time()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False  # 严格fp32保证精度

    print("=" * 80)
    print("第二步: 单GPU分块相似度计算 + TPIR@FPIR")
    print("=" * 80)

    # ---------- 1. 加载与排序 ----------
    t0 = time.time()
    feats_s, ids_c, counts, starts, ends, order, file_paths = load_and_sort(args.pkl)
    N = len(ids_c)
    n_pos = int(sum(int(c) * (c - 1) // 2 for c in counts))
    n_neg = N * (N - 1) // 2 - n_pos
    print(f"\n[1] 加载+排序: {time.time()-t0:.2f}s | N={N}, 身份数={len(counts)}")
    print(f"    正样本对: {n_pos:,} | 负样本对: {n_neg:,}")

    # ---------- 2. 正样本(精确) ----------
    t0 = time.time()
    F_cpu = torch.from_numpy(np.ascontiguousarray(feats_s))
    pos_sims, pos_i, pos_j = compute_positives(F_cpu, starts, ends, dev)
    torch.cuda.synchronize()
    print(f"\n[2] 正样本计算(bmm分桶): {time.time()-t0:.2f}s, 共{len(pos_sims):,}对")
    print(f"    sim范围: [{pos_sims.min():.4f}, {pos_sims.max():.4f}], 均值{pos_sims.mean():.4f}")
    assert len(pos_sims) == n_pos, f"正样本对数不符: {len(pos_sims)} != {n_pos}"

    # ---------- 3. 负样本扫描 ----------
    print(f"\n[3] 负样本扫描: 行块={args.block_rows}, 列块={args.block_cols}, "
          f"直方图bins={args.hist_bins}, 提取上限={args.extract_cap:,}")
    t0 = time.time()
    neg_hist, top_v, top_i, top_j, t_sweep, n_blocks = sweep_negatives(
        F_cpu, ids_c, N, dev, args.block_rows, args.block_cols,
        args.hist_bins, args.extract_cap)
    print(f"    扫描完成: {n_blocks}块, 纯扫描耗时{t_sweep:.2f}s (含数据搬运总{time.time()-t0:.2f}s)")
    print(f"    吞吐: {n_neg / t_sweep / 1e9:.2f} G对/s")
    assert int(neg_hist.sum()) == n_neg, \
        f"负样本对数不符: {int(neg_hist.sum())} != {n_neg}"

    # ---------- 4. TPIR@FPIR ----------
    print(f"\n[4] TPIR@FPIR (直方图法, bin宽{2.0/(args.hist_bins-1):.2e}):")
    targets = [1e-2, 1e-3, 1e-4, 1e-5]
    res = tpir_at_fpir(pos_sims, neg_hist, args.hist_bins, targets)
    print(f"    {'FPIR':>10} | {'TPIR':>8} | {'阈值':>8} | 直方图bin计数")
    for f in targets:
        tpir, t, _ = res[f]
        print(f"    {f:>10.0e} | {tpir*100:>7.3f}% | {t:>8.4f} | {int(f*n_neg):,}")

    # 精确top-N交叉验证 (覆盖 FPIR >= cap/n_neg)
    f_cover = args.extract_cap / n_neg
    print(f"\n    交叉验证: top-N精确值覆盖 FPIR >= {f_cover:.1e}")
    pos_sorted = np.sort(pos_sims)
    top_sorted = np.sort(top_v)[::-1]
    for f in targets:
        r = int(round(f * n_neg))
        if r <= len(top_sorted) and r >= 1:
            t_exact = top_sorted[r - 1]
            tpir_exact = 1.0 - np.searchsorted(pos_sorted, t_exact, side='right') / len(pos_sorted)
            tpir_hist = res[f][0]
            print(f"    FPIR={f:.0e}: 直方图TPIR={tpir_hist*100:.3f}% vs 精确TPIR={tpir_exact*100:.3f}%"
                  f"  (Δ={abs(tpir_hist-tpir_exact)*100:.4f}%)")

    # ---------- 5. 保存 ----------
    np.savez_compressed(
        args.save,
        neg_hist=neg_hist, pos_sims=pos_sims, pos_i=pos_i, pos_j=pos_j,
        top_v=top_v, top_i=top_i, top_j=top_j,
        order=order, n_neg=n_neg, n_pos=n_pos)
    print(f"\n[5] 统计结果已保存: {args.save}")

    print(f"\n总耗时: {time.time()-t_total:.2f}s")
    print("=" * 80)
    print("预期范围对照: 1e-5: 60-65% | 1e-4: 82-85% | 1e-3: 90-93% | 1e-2: 95-97%")
    for f in targets:
        tpir = res[f][0] * 100
        print(f"  TPIR@FPIR={f:.0e}: {tpir:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
