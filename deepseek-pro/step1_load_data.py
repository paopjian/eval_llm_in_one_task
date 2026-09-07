#!/usr/bin/env python3
"""第一步：读取数据并分析基本统计信息"""
import pickle
import time
import numpy as np

t0 = time.time()
with open('s4_0618_enhance.pkl', 'rb') as f:
    query_feats_list, query_feats_list_flip, query_ids, file_paths = pickle.load(f)
t_load = time.time() - t0

print(f"加载耗时: {t_load:.2f}s")
print(f"特征矩阵形状: {query_feats_list.shape}, dtype={query_feats_list.dtype}")
print(f"flip 列表长度: {len(query_feats_list_flip)}")
print(f"query_ids 形状: {query_ids.shape}, dtype={query_ids.dtype}")
print(f"file_paths 长度: {len(file_paths)}")
print(f"file_paths[0]: {file_paths[0]}")

N = len(query_ids)
total_pairs = N * (N - 1) // 2
unique_ids, counts = np.unique(query_ids, return_counts=True)
total_pos_pairs = int(sum(c * (c - 1) // 2 for c in counts))
total_neg_pairs = total_pairs - total_pos_pairs

print(f"\n样本总数 N = {N:,}")
print(f"身份总数 = {len(unique_ids):,}")
print(f"每个身份样本数: min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}, median={np.median(counts):.1f}")
print(f"总样本对数 = {total_pairs:,}")
print(f"正样本对数 = {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.2f}%)")
print(f"负样本对数 = {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.2f}%)")

# 检查特征是否归一化
norms = np.linalg.norm(query_feats_list, axis=1)
print(f"\nL2范数: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}, std={norms.std():.6f}")
print(f"特征值范围: min={query_feats_list.min():.6f}, max={query_feats_list.max():.6f}")

# 保存轻量级中间数据（float32 压缩 + ids + paths）
np.save('feats_f32.npy', query_feats_list.astype(np.float32))
np.save('ids.npy', query_ids.astype(np.int64))
with open('paths.txt', 'w') as f:
    for p in file_paths:
        f.write(str(p) + '\n')
print("\n已保存 feats_f32.npy / ids.npy / paths.txt")
print(f"总耗时: {time.time()-t0:.2f}s")
