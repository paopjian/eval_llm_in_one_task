#!/usr/bin/env python3
"""
大规模人脸特征相似度评估。

主要特性：
* 分块计算上三角余弦相似度，不构造 NxN 矩阵；
* 流式累计正/负样本相似度直方图，内存复杂度为 O(N * D + bins)；
* 可选保存阈值两侧的样本对到 Parquet；
* 有 CUDA 时支持多卡进程并行；无 CUDA 时自动使用 CPU；
* 输出 TPIR@FPIR 曲线、相似度分布图和 JSON 摘要。

示例：
    python face_similarity_eval.py --input s4_0618_enhance.pkl --output-dir results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


# matplotlib 在部分 cvlface 镜像中会优先加载系统 libstdc++，导致 CXXABI 版本错误。
# 在导入 pyplot 前将环境内的运行库加载为全局符号，失败时仍允许核心评估继续运行。
def _prepare_matplotlib_runtime() -> None:
    try:
        import ctypes

        env_lib = Path(sys_prefix()) / "lib" / "libstdc++.so.6"
        if env_lib.exists():
            ctypes.CDLL(str(env_lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def sys_prefix() -> str:
    return os.environ.get("CONDA_PREFIX", sys.prefix)


@dataclass
class EvaluationResult:
    """评估结果及可选的原始分数。"""

    total_samples: int
    identity_count: int
    total_pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_hist: np.ndarray
    negative_hist: np.ndarray
    bin_edges: np.ndarray
    elapsed_seconds: float
    backend: str
    device_count: int
    precision: str = "fp32"
    positive_scores: np.ndarray | None = None
    negative_scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bin_centers(self) -> np.ndarray:
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def _threshold_from_fpir(self, fpir: float) -> float:
        """用负样本经验分布求阈值，保证 FPIR 不超过目标值。"""
        if not 0.0 <= fpir <= 1.0:
            raise ValueError(f"FPIR 必须位于 [0, 1]，收到 {fpir}")
        if self.negative_pairs == 0:
            # 没有负样本时 FPIR 无法定义；相似度范围为 [-1, 1]，用 1.0
            # 作为不会误报的有限哨兵阈值，避免把 NaN/∞ 写入 JSON 摘要。
            return 1.0
        target_false = fpir * self.negative_pairs
        # 直方图第 i 个区间为 [edge[i], edge[i+1])（最后一个区间包含右端点）。
        # 在 edge[t] 处用区间 t 及以后近似 ``score > threshold``，选择最低的
        # 满足尾部计数不超过目标的边界，避免把 FPIR 低估得过于乐观。
        tail_from_bin = np.r_[np.cumsum(self.negative_hist[::-1], dtype=np.int64)[::-1], 0]
        candidates = np.flatnonzero(tail_from_bin <= math.floor(target_false))
        edge_index = int(candidates[0]) if candidates.size else len(self.bin_edges) - 1
        threshold = float(self.bin_edges[edge_index])
        # 直方图边界可能带来少量量化误差；有原始分数时改用精确分位点。
        if self.negative_scores is not None and len(self.negative_scores):
            if fpir == 0:
                threshold = float(np.max(self.negative_scores))
            elif fpir >= 1:
                threshold = float(np.nextafter(np.min(self.negative_scores), -np.inf))
            else:
                order = max(0, int(math.ceil((1.0 - fpir) * len(self.negative_scores))) - 1)
                threshold = float(np.partition(self.negative_scores, order)[order])
        return threshold

    def metrics_at_fpir(self, fpirs: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2)) -> list[dict[str, float]]:
        """返回指定 FPIR 点的阈值、FPIR 和 TPIR。"""
        rows: list[dict[str, float]] = []
        for fpir in fpirs:
            threshold = self._threshold_from_fpir(float(fpir))
            if self.positive_scores is not None:
                tp = int(np.count_nonzero(self.positive_scores > threshold))
            else:
                index = int(np.searchsorted(self.bin_edges, threshold, side="left"))
                index = int(np.clip(index, 0, len(self.positive_hist)))
                tp = int(np.sum(self.positive_hist[index:], dtype=np.int64))
            if self.negative_scores is not None:
                fp = int(np.count_nonzero(self.negative_scores > threshold))
            else:
                index = int(np.searchsorted(self.bin_edges, threshold, side="left"))
                index = int(np.clip(index, 0, len(self.negative_hist)))
                fp = int(np.sum(self.negative_hist[index:], dtype=np.int64))
            rows.append(
                {
                    "fpir_target": float(fpir),
                    "threshold": threshold,
                    "fpir": fp / self.negative_pairs if self.negative_pairs else 0.0,
                    "tpir": tp / self.positive_pairs if self.positive_pairs else 0.0,
                    "false_positives": float(fp),
                    "true_positives": float(tp),
                }
            )
        return rows

    def curve(self, max_points: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回阈值、FPIR、TPIR 曲线（从高阈值到低阈值）。

        原始分数较多时只保留均匀采样的曲线点，避免绘图阶段构造 O(K²)
        的逐阈值统计；指标计算仍使用精确的 ``metrics_at_fpir``。
        """
        if max_points < 2:
            raise ValueError("max_points 至少为 2")
        if self.positive_scores is not None and self.negative_scores is not None:
            scores = np.unique(np.concatenate([self.positive_scores, self.negative_scores]))
            if len(scores) > max_points:
                keep = np.unique(np.linspace(0, len(scores) - 1, max_points, dtype=np.int64))
                scores = scores[keep]
            thresholds = np.r_[np.inf, scores[::-1], -np.inf]
            negative_sorted = np.sort(self.negative_scores)
            positive_sorted = np.sort(self.positive_scores)
            fp_count = self.negative_pairs - np.searchsorted(negative_sorted, thresholds, side="right")
            tp_count = self.positive_pairs - np.searchsorted(positive_sorted, thresholds, side="right")
            fpir = fp_count / max(self.negative_pairs, 1)
            tpir = tp_count / max(self.positive_pairs, 1)
            return thresholds, fpir, tpir
        thresholds = self.bin_edges[:0:-1]
        false_tail_ascending = np.r_[np.cumsum(self.negative_hist[::-1], dtype=np.int64)[::-1][1:], 0]
        true_tail_ascending = np.r_[np.cumsum(self.positive_hist[::-1], dtype=np.int64)[::-1][1:], 0]
        false_tail = false_tail_ascending[::-1]
        true_tail = true_tail_ascending[::-1]
        return thresholds, false_tail / max(self.negative_pairs, 1), true_tail / max(self.positive_pairs, 1)

    def to_dict(self, fpirs: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2)) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "identity_count": self.identity_count,
            "total_pairs": self.total_pairs,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "elapsed_seconds": self.elapsed_seconds,
            "backend": self.backend,
            "device_count": self.device_count,
            "precision": self.precision,
            "metrics": self.metrics_at_fpir(fpirs),
            "metadata": self.metadata,
        }


def load_features(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """读取任务约定的 pkl，并校验特征、标签、路径长度。"""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 4:
        raise ValueError("特征文件应包含 (features, features_flip, ids, file_paths) 四项")
    features, _features_flip, ids, file_paths = payload
    features = np.asarray(features)
    ids = np.asarray(ids)
    if features.ndim != 2:
        raise ValueError(f"特征必须是二维数组，收到 shape={features.shape}")
    if ids.ndim != 1 or len(ids) != len(features):
        raise ValueError("身份标签必须是一维数组且与特征数量一致")
    if len(file_paths) != len(features):
        raise ValueError("file_paths 长度必须与特征数量一致")
    if not np.issubdtype(features.dtype, np.floating):
        features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("特征中包含 NaN 或无穷值")
    return np.ascontiguousarray(features), np.ascontiguousarray(ids), list(file_paths)


def pair_counts(ids: np.ndarray) -> tuple[int, int, int]:
    n = int(len(ids))
    total = n * (n - 1) // 2
    _, counts = np.unique(ids, return_counts=True)
    positive = int(np.sum(counts.astype(np.int64) * (counts - 1) // 2, dtype=np.int64))
    return total, positive, total - positive


def _normalise_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32, order="C")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if norms.size and np.all(np.abs(norms - 1.0) <= 1e-5):
        # 任务数据已经归一化时复用原数组，避免额外分配数百 MB。
        return features
    # 避免零向量导致 NaN；零向量与任意向量的相似度定义为 0。
    return (features / np.maximum(norms, np.finfo(features.dtype).eps)).astype(np.float32, copy=False)


@dataclass
class _ChunkStats:
    pos_hist: np.ndarray
    neg_hist: np.ndarray
    pos_count: int = 0
    neg_count: int = 0
    pos_scores: list[np.ndarray] = field(default_factory=list)
    neg_scores: list[np.ndarray] = field(default_factory=list)

    def merge(self, other: "_ChunkStats") -> None:
        self.pos_hist += other.pos_hist
        self.neg_hist += other.neg_hist
        self.pos_count += other.pos_count
        self.neg_count += other.neg_count
        self.pos_scores.extend(other.pos_scores)
        self.neg_scores.extend(other.neg_scores)


def _empty_stats(bins: int) -> _ChunkStats:
    return _ChunkStats(np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64))


def _score_to_hist(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.int64)
    # 相似度理论范围 [-1, 1]；clip 可吸收浮点误差。
    indices = np.floor((np.clip(scores, edges[0], edges[-1]) - edges[0]) / (edges[-1] - edges[0]) * (len(edges) - 1)).astype(np.int64)
    indices = np.clip(indices, 0, len(edges) - 2)
    return np.bincount(indices, minlength=len(edges) - 1).astype(np.int64, copy=False)


def _score_to_torch_hist(scores: Any, bins: int) -> Any:
    """在当前 Torch 设备上生成与 ``_score_to_hist`` 一致的直方图。"""
    import torch

    if scores.numel() == 0:
        return torch.zeros(bins, dtype=torch.int64, device=scores.device)
    indices = ((scores.clamp(-1.0, 1.0) + 1.0) * (bins / 2.0)).to(torch.int64)
    indices.clamp_(0, bins - 1)
    return torch.bincount(indices, minlength=bins)


def _iter_row_ranges(start: int, stop: int, block_size: int) -> Iterator[tuple[int, int]]:
    for row_start in range(start, stop, block_size):
        yield row_start, min(row_start + block_size, stop)


def _compute_ranges(
    features: np.ndarray,
    ids: np.ndarray,
    row_ranges: Sequence[tuple[int, int]],
    block_size: int,
    bins: int,
    collect_scores: bool,
    device: str,
    precision: str = "fp32",
) -> _ChunkStats:
    """计算若干行块的上三角分数。

    一个 worker 一次性把特征搬到自己的设备，避免按块重复传输整份特征。
    """
    stats = _empty_stats(bins)
    edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float32)
    use_torch = device.startswith("cuda")
    if use_torch:
        import torch

        torch_device = torch.device(device)
        feature_tensor = torch.from_numpy(features).to(torch_device)
        if precision == "fp16":
            feature_tensor = feature_tensor.half()
        id_tensor = torch.from_numpy(ids).to(torch_device)
        pos_hist_device = torch.zeros(bins, dtype=torch.int64, device=torch_device)
        neg_hist_device = torch.zeros(bins, dtype=torch.int64, device=torch_device)
    else:
        feature_tensor = features
    torch_context = None
    if use_torch:
        import torch

        torch_context = torch.inference_mode()
        torch_context.__enter__()
    try:
        for row_start, row_stop in row_ranges:
            for i0, i1 in _iter_row_ranges(row_start, row_stop, block_size):
                # 从当前块开始计算：当前块只保留严格上三角，后续块全部有效。
                # 这样既不会漏掉块内样本对，也不会重复计算反向对。
                for j0_block, j1_block in _iter_row_ranges(i0, len(features), block_size):
                    if use_torch:
                        score_matrix = torch.matmul(feature_tensor[i0:i1], feature_tensor[j0_block:j1_block].T).float()
                        same_ids = id_tensor[i0:i1, None] == id_tensor[None, j0_block:j1_block]
                        if j0_block == i0:
                            valid = torch.triu(torch.ones_like(score_matrix, dtype=torch.bool), diagonal=1)
                            valid_scores = score_matrix[valid]
                            pos = score_matrix[same_ids & valid]
                            valid_count = int(valid.sum().item())
                        else:
                            valid_scores = score_matrix.reshape(-1)
                            pos = score_matrix[same_ids]
                            valid_count = score_matrix.numel()
                        full_hist = _score_to_torch_hist(valid_scores, bins)
                        pos_hist = _score_to_torch_hist(pos, bins)
                        stats.pos_count += int(pos.numel())
                        stats.neg_count += int(valid_count - pos.numel())
                        pos_hist_device += pos_hist
                        neg_hist_device += full_hist - pos_hist
                        if collect_scores:
                            neg = score_matrix[~same_ids & (valid if j0_block == i0 else True)]
                            if pos.numel():
                                stats.pos_scores.append(pos.cpu().numpy().astype(np.float32, copy=False))
                            if neg.numel():
                                stats.neg_scores.append(neg.cpu().numpy().astype(np.float32, copy=False))
                    else:
                        score_matrix = np.matmul(features[i0:i1], features[j0_block:j1_block].T, dtype=np.float32)
                        same_ids = ids[i0:i1, None] == ids[None, j0_block:j1_block]
                        if j0_block == i0:
                            # 对角块只保留严格上三角；其余块天然满足 j > i。
                            valid = np.triu(np.ones(score_matrix.shape, dtype=bool), k=1)
                            valid_scores = score_matrix[valid]
                            pos = score_matrix[same_ids & valid]
                            valid_count = int(valid.sum())
                        else:
                            valid_scores = score_matrix.reshape(-1)
                            pos = score_matrix[same_ids]
                            valid_count = score_matrix.size
                        full_hist = _score_to_hist(valid_scores, edges)
                        pos_hist = _score_to_hist(pos, edges)
                        stats.pos_count += int(pos.size)
                        stats.neg_count += valid_count - int(pos.size)
                        stats.pos_hist += pos_hist
                        stats.neg_hist += full_hist - pos_hist
                        if collect_scores:
                            neg = score_matrix[~same_ids & (valid if j0_block == i0 else True)]
                            if pos.size:
                                stats.pos_scores.append(pos.astype(np.float32, copy=False))
                            if neg.size:
                                stats.neg_scores.append(neg.astype(np.float32, copy=False))
    finally:
        if torch_context is not None:
            torch_context.__exit__(None, None, None)
    if use_torch:
        stats.pos_hist = pos_hist_device.cpu().numpy()
        stats.neg_hist = neg_hist_device.cpu().numpy()
    return stats


def _compute_range(
    features: np.ndarray,
    ids: np.ndarray,
    row_start: int,
    row_stop: int,
    block_size: int,
    bins: int,
    collect_scores: bool,
    device: str,
    precision: str = "fp32",
) -> _ChunkStats:
    """兼容单行块调用的内部函数。"""
    return _compute_ranges(features, ids, [(row_start, row_stop)], block_size, bins, collect_scores, device, precision)


def _worker_compute_group(args: tuple[Any, ...]) -> _ChunkStats:
    return _compute_ranges(*args)


def _partition_ranges(ranges: Sequence[tuple[int, int]], device_count: int, sample_count: int) -> list[list[tuple[int, int]]]:
    """按剩余行数估算工作量，贪心分配行块以平衡多卡负载。"""
    groups: list[list[tuple[int, int]]] = [[] for _ in range(device_count)]
    loads = [0] * device_count
    weighted = sorted(ranges, key=lambda r: sample_count - r[0], reverse=True)
    for item in weighted:
        target = min(range(device_count), key=loads.__getitem__)
        groups[target].append(item)
        loads[target] += sample_count - item[0]
    return groups


def evaluate_similarity(
    features: np.ndarray,
    ids: np.ndarray,
    *,
    block_size: int = 2048,
    bins: int = 4096,
    devices: Sequence[str] | None = None,
    workers: int | None = None,
    collect_scores: bool | None = None,
    normalize: bool = True,
    precision: str = "auto",
) -> EvaluationResult:
    """计算上三角正负样本相似度统计。

    ``collect_scores`` 默认仅在总样本对不超过 5 百万时启用，避免大数据集内存爆炸。
    """
    if block_size <= 0 or bins < 2:
        raise ValueError("block_size 必须为正数，bins 至少为 2")
    if workers is not None and workers <= 0:
        raise ValueError("workers 必须为正数")
    if precision not in {"auto", "fp32", "fp16"}:
        raise ValueError("precision 必须是 auto、fp32 或 fp16")
    features = np.asarray(features)
    ids = np.asarray(ids)
    if features.ndim != 2 or ids.ndim != 1:
        raise ValueError("features 必须是二维数组，ids 必须是一维数组")
    if len(features) != len(ids):
        raise ValueError("features 与 ids 长度不一致")
    if not np.all(np.isfinite(features)):
        raise ValueError("特征中包含 NaN 或无穷值")
    if normalize:
        features = _normalise_features(features)
    else:
        features = np.asarray(features, dtype=np.float32, order="C")
    unique_ids, encoded_ids = np.unique(ids, return_inverse=True)
    encoded_ids = np.ascontiguousarray(encoded_ids, dtype=np.int64)
    total_pairs, positive_pairs, negative_pairs = pair_counts(encoded_ids)
    if collect_scores is None:
        collect_scores = total_pairs <= 5_000_000
    if devices is None:
        try:
            import torch

            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        except Exception:
            devices = ["cpu"]
    devices = list(devices) or ["cpu"]
    if any(str(d).startswith("cuda") for d in devices):
        try:
            import torch

            if not torch.cuda.is_available():
                devices = ["cpu"]
        except Exception:
            devices = ["cpu"]
    has_cuda = any(str(d).startswith("cuda") for d in devices)
    if precision == "auto":
        effective_precision = "fp16" if has_cuda else "fp32"
    elif precision == "fp16" and not has_cuda:
        effective_precision = "fp32"
    else:
        effective_precision = precision
    start_time = time.perf_counter()
    # 按行范围切分，每个范围只负责其起始行对应的上三角，避免重复。
    ranges = list(_iter_row_ranges(0, len(features), block_size))
    merged = _empty_stats(bins)
    groups = _partition_ranges(ranges, len(devices), len(features))
    group_args = [
        (features, encoded_ids, group, block_size, bins, bool(collect_scores), devices[index], effective_precision)
        for index, group in enumerate(groups)
        if group
    ]
    # CPU 多进程需要序列化特征，通常不划算；GPU 场景每个任务只传输一次特征。
    use_parallel = len(group_args) > 1 and all(str(d).startswith("cuda") for d in devices)
    if use_parallel:
        process_count = min(workers or len(group_args), len(group_args))
        ctx = get_context("spawn")
        with ctx.Pool(processes=process_count) as pool:
            for partial in pool.imap(_worker_compute_group, group_args, chunksize=1):
                merged.merge(partial)
    else:
        for args in group_args:
            merged.merge(_worker_compute_group(args))
    positive_scores = np.concatenate(merged.pos_scores) if collect_scores and merged.pos_scores else None
    negative_scores = np.concatenate(merged.neg_scores) if collect_scores and merged.neg_scores else None
    elapsed = time.perf_counter() - start_time
    if merged.pos_count != positive_pairs or merged.neg_count != negative_pairs:
        raise RuntimeError(
            "上三角统计数量异常："
            f"期望正/负={positive_pairs}/{negative_pairs}，"
            f"实际正/负={merged.pos_count}/{merged.neg_count}"
        )
    return EvaluationResult(
        total_samples=len(features),
        identity_count=len(unique_ids),
        total_pairs=total_pairs,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        positive_hist=merged.pos_hist,
        negative_hist=merged.neg_hist,
        bin_edges=np.linspace(-1.0, 1.0, bins + 1, dtype=np.float32),
        elapsed_seconds=elapsed,
        backend="cuda" if any(str(d).startswith("cuda") for d in devices) else "cpu",
        device_count=len(devices),
        precision=effective_precision,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    )


def extract_pairs(
    features: np.ndarray,
    ids: np.ndarray,
    file_paths: Sequence[str],
    *,
    threshold: float,
    mode: str = "above",
    block_size: int = 2048,
    limit: int = 10000,
    pair_type: str = "all",
) -> list[dict[str, Any]]:
    """提取阈值以上/以下的样本对，返回最多 ``limit`` 条。"""
    if mode not in {"above", "below"}:
        raise ValueError("mode 必须是 above 或 below")
    if pair_type not in {"all", "positive", "negative"}:
        raise ValueError("pair_type 必须是 all、positive 或 negative")
    if block_size <= 0:
        raise ValueError("block_size 必须为正数")
    if limit <= 0:
        return []
    features = _normalise_features(np.asarray(features))
    ids = np.asarray(ids)
    if features.ndim != 2 or ids.ndim != 1 or len(features) != len(ids):
        raise ValueError("features 必须为二维、ids 必须为一维且长度一致")
    if len(file_paths) != len(features):
        raise ValueError("file_paths 长度必须与 features 一致")
    result: list[dict[str, Any]] = []
    n = len(features)
    for i0, i1 in _iter_row_ranges(0, n, block_size):
        for j0, j1 in _iter_row_ranges(i0, n, block_size):
            scores = np.matmul(features[i0:i1], features[j0:j1].T, dtype=np.float32)
            same = ids[i0:i1, None] == ids[None, j0:j1]
            mask = scores > threshold if mode == "above" else scores < threshold
            if j0 == i0:
                mask &= np.triu(np.ones(scores.shape, dtype=bool), k=1)
            if pair_type == "positive":
                mask &= same
            elif pair_type == "negative":
                mask &= ~same
            rows, cols = np.nonzero(mask)
            for r, c in zip(rows.tolist(), cols.tolist()):
                result.append(
                    {
                        "index_i": i0 + r,
                        "index_j": j0 + c,
                        "id_i": ids[i0 + r].item() if hasattr(ids[i0 + r], "item") else ids[i0 + r],
                        "id_j": ids[j0 + c].item() if hasattr(ids[j0 + c], "item") else ids[j0 + c],
                        "similarity": float(scores[r, c]),
                        "path_i": str(file_paths[i0 + r]),
                        "path_j": str(file_paths[j0 + c]),
                        "is_positive": bool(same[r, c]),
                    }
                )
                if len(result) >= limit:
                    return result
    return result


def save_pairs(rows: Sequence[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    """优先使用 Polars 保存 Parquet；若不可用则明确报错。"""
    try:
        import polars as pl

        pl.DataFrame(rows).write_parquet(path)
    except ImportError as exc:
        raise RuntimeError("保存样本对需要环境中的 polars") from exc


def _configure_matplotlib() -> Any:
    _prepare_matplotlib_runtime()
    if not os.environ.get("MPLCONFIGDIR"):
        cache_dir = Path(tempfile.gettempdir()) / "face_similarity_eval_mpl"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_results(
    result: EvaluationResult,
    output_dir: str | os.PathLike[str],
    font_path: str | os.PathLike[str] | None = None,
    fpirs: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2),
) -> tuple[Path, Path]:
    """绘制正负分布和 TPIR@FPIR 曲线。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib()
    if font_path and Path(font_path).exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    distribution_path = output / "similarity_distribution.png"
    curve_path = output / "tpir_fpir_curve.png"
    centers = result.bin_centers
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pos_density = result.positive_hist / max(result.positive_pairs, 1)
    neg_density = result.negative_hist / max(result.negative_pairs, 1)
    ax.plot(centers, pos_density, label="正样本", color="#d62728", linewidth=1.4)
    ax.plot(centers, neg_density, label="负样本", color="#1f77b4", linewidth=1.4)
    ax.set_xlabel("余弦相似度")
    ax.set_ylabel("比例（每个直方图区间）")
    ax.set_title("正负样本相似度分布")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=150)
    plt.close(fig)
    _thresholds, fpir, tpir = result.curve()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    valid = fpir > 0
    ax.plot(fpir[valid], tpir[valid], color="#2ca02c", linewidth=1.6)
    ax.set_xscale("log")
    if np.any(valid):
        min_fpir = float(np.min(fpir[valid]))
        left_fpir = min(max(min_fpir * 0.5, 1e-12), 0.5)
    else:
        left_fpir = 1e-8
    ax.set_xlim(left=left_fpir, right=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("FPIR")
    ax.set_ylabel("TPIR")
    ax.set_title("TPIR@FPIR 曲线")
    ax.grid(which="both", alpha=0.25)
    for row in result.metrics_at_fpir(fpirs):
        ax.scatter([max(row["fpir"], 1e-12)], [row["tpir"]], s=25)
        ax.annotate(f"{row['fpir_target']:.0e}", (max(row["fpir"], 1e-12), row["tpir"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    return distribution_path, curve_path


def _parse_devices(value: str) -> list[str]:
    value = value.strip().lower()
    if value in {"auto", ""}:
        return []
    if value == "cpu":
        return ["cpu"]
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("devices 不能为空")
    devices: list[str] = []
    for part in parts:
        device = part if part.startswith("cuda:") else f"cuda:{part}"
        index = device.removeprefix("cuda:")
        if not index.isdigit():
            raise ValueError(f"无效设备: {part!r}，应使用 cpu、GPU 编号或 cuda:N")
        devices.append(device)
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="大规模人脸特征相似度与 TPIR@FPIR 评估")
    parser.add_argument("--input", default="s4_0618_enhance.pkl", help="特征 pkl 文件")
    parser.add_argument("--output-dir", default="results", help="输出目录")
    parser.add_argument("--block-size", type=int, default=2048, help="矩阵分块边长")
    parser.add_argument("--bins", type=int, default=4096, help="相似度直方图区间数")
    parser.add_argument("--devices", default="auto", help="auto、cpu 或逗号分隔的 GPU 编号，如 0,1,2")
    parser.add_argument("--workers", type=int, default=None, help="GPU worker 进程数")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto", help="GPU 矩阵乘法精度；auto 在 CUDA 上使用 FP16")
    parser.add_argument("--fpirs", default="1e-5,1e-4,1e-3,1e-2", help="输出的 FPIR 点，逗号分隔")
    parser.add_argument("--no-normalize", action="store_true", help="跳过 L2 归一化（仅在已确认归一化时使用）")
    parser.add_argument("--collect-scores", action="store_true", help="强制保存全部原始分数；大数据集不建议")
    parser.add_argument("--extract-threshold", type=float, default=None, help="提取样本对的相似度阈值")
    parser.add_argument("--extract-mode", choices=("above", "below"), default="above")
    parser.add_argument("--extract-type", choices=("all", "positive", "negative"), default="all")
    parser.add_argument("--extract-limit", type=int, default=10000)
    parser.add_argument("--font", default="font/SourceHanSansSC-Normal.otf", help="中文字体路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("大规模人脸特征相似度评估")
    print("=" * 72)
    print(f"读取特征: {args.input}")
    features, ids, file_paths = load_features(args.input)
    total, positive, negative = pair_counts(ids)
    _, identity_counts = np.unique(ids, return_counts=True)
    norms = np.linalg.norm(features, axis=1)
    print(f"样本数: {len(features):,}，特征维度: {features.shape[1]}，身份数: {len(np.unique(ids)):,}")
    print(f"样本对: {total:,}（正 {positive:,}，负 {negative:,}）")
    if identity_counts.size:
        print(f"每身份样本数: 均值 {identity_counts.mean():.2f}，范围 {identity_counts.min()}–{identity_counts.max()}")
    print(f"L2 范数: 均值 {norms.mean():.6f}，标准差 {norms.std():.6f}")
    devices = _parse_devices(args.devices)
    try:
        fpirs = tuple(float(value.strip()) for value in args.fpirs.split(",") if value.strip())
    except ValueError as exc:
        raise SystemExit(f"--fpirs 参数无效: {args.fpirs}") from exc
    if not fpirs or any(value < 0.0 or value > 1.0 for value in fpirs):
        raise SystemExit("--fpirs 中的每个值必须位于 [0, 1]")
    result = evaluate_similarity(
        features,
        ids,
        block_size=args.block_size,
        bins=args.bins,
        devices=devices or None,
        workers=args.workers,
        collect_scores=True if args.collect_scores else None,
        normalize=not args.no_normalize,
        precision=args.precision,
    )
    if devices and any(device.startswith("cuda") for device in devices) and result.backend != "cuda":
        print("警告：请求的 CUDA 设备当前不可用，已自动退化为 CPU。")
    result.metadata.update(
        {
            "input": str(args.input),
            "block_size": args.block_size,
            "bins": args.bins,
            "normalize": not args.no_normalize,
            "precision": result.precision,
            "collect_scores": result.positive_scores is not None or result.negative_scores is not None,
            "devices": devices or ["auto"],
        }
    )
    print(f"计算完成: {result.elapsed_seconds:.2f} 秒，后端 {result.backend}（{result.device_count} 个设备，{result.precision}）")
    print("\nTPIR@FPIR:")
    for row in result.metrics_at_fpir(fpirs):
        print(f"  FPIR={row['fpir_target']:.0e}: threshold={row['threshold']:.6f}, actual FPIR={row['fpir']:.6g}, TPIR={row['tpir']:.4%}")
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(fpirs), handle, ensure_ascii=False, indent=2)
    np.savez_compressed(output_dir / "similarity_histograms.npz", positive_hist=result.positive_hist, negative_hist=result.negative_hist, bin_edges=result.bin_edges)
    try:
        paths = plot_results(result, output_dir, args.font, fpirs)
        print(f"图表: {paths[0]}，{paths[1]}")
    except Exception as exc:
        print(f"警告：绘图失败（核心评估结果已保存）: {exc}")
    if args.extract_threshold is not None:
        rows = extract_pairs(features, ids, file_paths, threshold=args.extract_threshold, mode=args.extract_mode, block_size=args.block_size, limit=args.extract_limit, pair_type=args.extract_type)
        pair_path = output_dir / f"pairs_{args.extract_mode}_{args.extract_threshold:g}.parquet"
        save_pairs(rows, pair_path)
        print(f"已提取 {len(rows):,} 条样本对: {pair_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
