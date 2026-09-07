#!/usr/bin/env python3
"""大规模人脸特征相似度评估。

设计要点：
1. 只计算上三角块，避免重复的 NxN 相似度；
2. 正样本仅占极小比例，预先建立正样本坐标索引。每个块先统计全部相似度，
   再减去正样本直方图得到负样本统计，避免构造巨大的身份比较掩码；
3. 全程仅保留固定大小的直方图，因此可处理数十亿级样本对；
4. Linux 下使用 fork 让多个 GPU 进程共享只读的 CPU 特征数组，避免重复加载 pkl。

默认使用直方图近似 TPIR@FPIR。hist-bins=200000 时阈值分辨率为 1e-5，
适合本任务的指标精度和大规模吞吐需求。
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import queue
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


# fork 后由子进程以只读方式复用；父进程在创建子进程前不得初始化 CUDA。
_SHARED_FEATURES: np.ndarray | None = None
_SHARED_ID_CODES: np.ndarray | None = None
_SHARED_POSITIVE_PAIRS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None


@dataclass(frozen=True)
class BlockTask:
    """一个上三角块乘法任务。"""

    row_block: int
    col_block: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def pair_count(self) -> int:
        rows = self.row_end - self.row_start
        cols = self.col_end - self.col_start
        if self.row_block == self.col_block:
            return rows * (rows - 1) // 2
        return rows * cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分块多卡人脸特征相似度评估（TPIR@FPIR）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="s4_0618_enhance.pkl", help="特征 pkl 路径")
    parser.add_argument("--output-dir", default="evaluation_output", help="结果输出目录")
    parser.add_argument(
        "--devices",
        default="auto",
        help="GPU 逻辑编号，逗号分隔；auto 自动发现（最多 7 张）；本程序要求 CUDA",
    )
    parser.add_argument("--max-gpus", type=int, default=7, help="auto 模式最多使用的 GPU 数")
    parser.add_argument("--block-size", type=int, default=8192, help="相似度块边长")
    parser.add_argument("--hist-bins", type=int, default=200_000, help="[-1, 1] 区间直方图桶数")
    parser.add_argument(
        "--fpir-targets",
        default="1e-5,1e-4,1e-3,1e-2",
        help="逗号分隔的 FPIR 目标值",
    )
    parser.add_argument(
        "--precision",
        choices=("tf32", "fp32"),
        default="tf32",
        help="tf32 更快；fp32 适合需要更严格数值复现的场景",
    )
    parser.add_argument(
        "--gpu-feature-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help="是否将完整特征缓存至每张 GPU；auto 在特征占显存不超过 25%% 时启用",
    )
    parser.add_argument(
        "--extract-above",
        type=float,
        default=None,
        help="提取相似度不低于该阈值的负样本对（潜在误接受）",
    )
    parser.add_argument(
        "--extract-below",
        type=float,
        default=None,
        help="提取相似度不高于该阈值的正样本对（潜在误拒绝）",
    )
    parser.add_argument("--max-extracted", type=int, default=100, help="每类最多写出的错误样本对数")
    parser.add_argument(
        "--font-path",
        default="font/SourceHanSansSC-Normal.otf",
        help="中文字体路径；不存在时自动使用 matplotlib 默认字体",
    )
    parser.add_argument("--no-plots", action="store_true", help="仅生成统计和 Parquet，不绘图")
    parser.add_argument("--dry-run", action="store_true", help="仅加载并输出执行计划，不计算相似度")
    args = parser.parse_args()

    if args.block_size < 2:
        parser.error("--block-size 必须不小于 2")
    if args.hist_bins < 100:
        parser.error("--hist-bins 必须不小于 100")
    if args.max_gpus < 1:
        parser.error("--max-gpus 必须不小于 1")
    if args.max_extracted < 1:
        parser.error("--max-extracted 必须不小于 1")
    for threshold in (args.extract_above, args.extract_below):
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            parser.error("提取阈值必须位于 [-1, 1]")
    return args


def parse_target_fpirs(text: str) -> list[float]:
    try:
        targets = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("--fpir-targets 必须是逗号分隔的浮点数") from error
    if not targets or any(not 0.0 < value < 1.0 for value in targets):
        raise ValueError("--fpir-targets 中每个值必须位于 (0, 1)")
    return targets


def load_features(input_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """加载 pkl，并将特征转换为连续 float32 数组。"""
    with input_path.open("rb") as handle:
        features, _flipped_features, identity_ids, file_paths = pickle.load(handle)

    features_array = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    identities_array = np.asarray(identity_ids).reshape(-1)
    paths = [str(path) for path in file_paths]

    if features_array.ndim != 2:
        raise ValueError(f"特征必须是二维数组，实际为 {features_array.shape}")
    if len(features_array) != len(identities_array) or len(features_array) != len(paths):
        raise ValueError("特征、身份标签、文件路径的长度不一致")
    if len(features_array) < 2:
        raise ValueError("至少需要两条特征才能计算样本对")
    if not np.isfinite(features_array).all():
        raise ValueError("特征包含 NaN 或 Inf")

    norms = np.linalg.norm(features_array, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-3:
        nonzero = norms > 0
        if not bool(np.all(nonzero)):
            raise ValueError("特征包含零向量，无法计算余弦相似度")
        features_array /= norms[:, None]
        print(f"检测到未归一化特征（最大范数误差 {max_norm_error:.6g}），已在内存中归一化。")

    return features_array, identities_array, paths


def make_tasks(sample_count: int, block_size: int) -> list[BlockTask]:
    """生成上三角块任务。"""
    block_count = math.ceil(sample_count / block_size)
    tasks: list[BlockTask] = []
    for row_block in range(block_count):
        row_start = row_block * block_size
        row_end = min(row_start + block_size, sample_count)
        for col_block in range(row_block, block_count):
            col_start = col_block * block_size
            col_end = min(col_start + block_size, sample_count)
            tasks.append(
                BlockTask(
                    row_block=row_block,
                    col_block=col_block,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                )
            )
    return tasks


def balance_tasks(tasks: list[BlockTask], worker_count: int) -> list[list[BlockTask]]:
    """按块乘法元素数贪心均衡分配任务。"""
    assignments: list[list[BlockTask]] = [[] for _ in range(worker_count)]
    assigned_pairs = [0] * worker_count
    for task in sorted(tasks, key=lambda item: item.pair_count, reverse=True):
        worker_index = min(range(worker_count), key=assigned_pairs.__getitem__)
        assignments[worker_index].append(task)
        assigned_pairs[worker_index] += task.pair_count
    return assignments


def build_positive_pair_index(
    id_codes: np.ndarray,
    block_size: int,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """建立每个块任务中的正样本局部坐标。

    正样本总量远小于全部样本对。预先索引后，GPU 只对这部分坐标做身份相关的
    统计；负样本直方图由全部样本对直方图减去正样本直方图得到。
    """
    parts: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    order = np.argsort(id_codes, kind="stable")
    sorted_codes = id_codes[order]
    # 不使用 np.diff，字符串或其他可比较标签同样可以作为身份 ID。
    group_starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1, len(order)]

    for begin, end in zip(group_starts[:-1], group_starts[1:]):
        indices = order[begin:end]
        if len(indices) < 2:
            continue
        block_numbers = indices // block_size
        unique_blocks, starts = np.unique(block_numbers, return_index=True)
        starts = np.r_[starts, len(indices)]
        block_positions: list[tuple[int, np.ndarray]] = []
        for position, block_number in enumerate(unique_blocks):
            local_indices = (indices[starts[position] : starts[position + 1]] - block_number * block_size).astype(
                np.int32,
                copy=False,
            )
            block_positions.append((int(block_number), local_indices))

        for left_position, (left_block, left_indices) in enumerate(block_positions):
            if len(left_indices) >= 2:
                row_offsets, col_offsets = np.triu_indices(len(left_indices), k=1)
                parts[(left_block, left_block)].append(
                    (left_indices[row_offsets], left_indices[col_offsets])
                )
            for right_block, right_indices in block_positions[left_position + 1 :]:
                rows = np.repeat(left_indices, len(right_indices))
                cols = np.tile(right_indices, len(left_indices))
                parts[(left_block, right_block)].append((rows, cols))

    positive_pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for task_key, coordinate_parts in parts.items():
        positive_pairs[task_key] = (
            np.concatenate([rows for rows, _ in coordinate_parts]).astype(np.int32, copy=False),
            np.concatenate([cols for _, cols in coordinate_parts]).astype(np.int32, copy=False),
        )
    return positive_pairs


def discover_devices(specification: str, max_gpus: int) -> list[str]:
    """确定 CUDA worker 设备；GPU 不可用时直接报错，不回退 CPU。"""
    if specification.strip().lower() == "cpu":
        raise ValueError("本任务要求使用 GPU，不接受 --devices cpu")
    if specification.strip().lower() != "auto":
        try:
            device_ids = [int(item.strip()) for item in specification.split(",") if item.strip()]
        except ValueError as error:
            raise ValueError("--devices 必须为 auto 或 GPU 编号列表") from error
        if not device_ids or any(index < 0 for index in device_ids):
            raise ValueError("--devices 中的 GPU 编号必须为非负整数")
        return [f"cuda:{index}" for index in device_ids]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1，当前进程没有可见 GPU。")
    if visible and visible.strip() != "":
        visible_count = len([item for item in visible.split(",") if item.strip()])
        return [f"cuda:{index}" for index in range(min(visible_count, max_gpus))]

    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        gpu_count = sum(1 for line in completed.stdout.splitlines() if line.startswith("GPU "))
        if gpu_count:
            return [f"cuda:{index}" for index in range(min(gpu_count, max_gpus))]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    # 某些容器没有 nvidia-smi 时，使用 PyTorch 的设备计数作为次级探测。
    try:
        device_count = torch.cuda.device_count()
        if device_count:
            return [f"cuda:{index}" for index in range(min(device_count, max_gpus))]
    except Exception:
        pass
    raise RuntimeError("未检测到可用 CUDA GPU；本任务禁止回退到 CPU。")


def configure_precision(precision: str) -> None:
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def should_cache_features(
    mode: str,
    feature_bytes: int,
    device: torch.device,
) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    total_memory = torch.cuda.get_device_properties(device).total_memory
    return feature_bytes <= total_memory // 4


def _histogram(scores: torch.Tensor, bins: int) -> torch.Tensor:
    """对分数计算整数直方图。

    ``torch.histc`` 的输出是 float32，在一个桶超过 2^24 个样本时不能精确
    表示整数。这里分块使用 ``bincount``，既保持 int64 精确计数，也限制了
    临时索引张量的大小。
    """
    flat_scores = scores.reshape(-1)
    histogram = torch.zeros(bins, dtype=torch.int64, device=scores.device)
    chunk_size = 4_000_000
    scale = bins / 2.0
    for chunk_start in range(0, flat_scores.numel(), chunk_size):
        chunk = flat_scores[chunk_start : chunk_start + chunk_size]
        valid = (chunk >= -1.0) & (chunk <= 1.0)
        bin_indices = torch.floor((chunk[valid] + 1.0) * scale).to(torch.int64)
        bin_indices.clamp_(min=0, max=bins - 1)
        histogram.add_(torch.bincount(bin_indices, minlength=bins))
    return histogram


def _top_negative_above(
    scores: torch.Tensor,
    task: BlockTask,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    threshold: float,
    limit: int,
) -> list[tuple[float, int, int]]:
    """从一个块中选择得分最高且达到阈值的负样本。"""
    # 仅将同身份位置屏蔽后取 top-k，保证正样本很多时也不会漏掉负样本。
    same_identity = row_ids[:, None] == col_ids[None, :]
    negative_scores = scores.masked_fill(same_identity, -2.0)
    candidate_count = min(negative_scores.numel(), limit)
    values, flat_indices = torch.topk(
        negative_scores.reshape(-1),
        k=candidate_count,
        largest=True,
        sorted=True,
    )
    local_rows = torch.div(flat_indices, scores.shape[1], rounding_mode="floor")
    local_cols = torch.remainder(flat_indices, scores.shape[1])
    valid = values >= threshold
    if not bool(valid.any()):
        return []
    selected_values = values[valid][:limit].detach().cpu().numpy()
    selected_rows = local_rows[valid][:limit].detach().cpu().numpy()
    selected_cols = local_cols[valid][:limit].detach().cpu().numpy()
    return [
        (float(score), task.row_start + int(row), task.col_start + int(col))
        for score, row, col in zip(selected_values, selected_rows, selected_cols)
    ]


def _top_positive_below(
    positive_scores: torch.Tensor,
    positive_rows: torch.Tensor,
    positive_cols: torch.Tensor,
    task: BlockTask,
    threshold: float,
    limit: int,
) -> list[tuple[float, int, int]]:
    """从一个块中选择得分最低的正样本。"""
    if positive_scores.numel() == 0:
        return []
    candidate_count = min(positive_scores.numel(), limit)
    values, selected = torch.topk(positive_scores, k=candidate_count, largest=False, sorted=True)
    valid = values <= threshold
    if not bool(valid.any()):
        return []
    selected_values = values[valid][:limit].detach().cpu().numpy()
    selected_indices = selected[valid][:limit].detach().cpu().numpy()
    rows = positive_rows[selected_indices].detach().cpu().numpy()
    cols = positive_cols[selected_indices].detach().cpu().numpy()
    return [
        (float(score), task.row_start + int(row), task.col_start + int(col))
        for score, row, col in zip(selected_values, rows, cols)
    ]


def evaluate_worker(
    device_name: str,
    tasks: list[BlockTask],
    histogram_bins: int,
    precision: str,
    feature_cache_mode: str,
    extract_above: float | None,
    extract_below: float | None,
    max_extracted: int,
) -> dict[str, Any]:
    """在单个设备上执行已分配的块任务。"""
    if _SHARED_FEATURES is None or _SHARED_ID_CODES is None or _SHARED_POSITIVE_PAIRS is None:
        raise RuntimeError("共享数据未初始化")

    started_at = time.perf_counter()
    if not device_name.startswith("cuda:"):
        raise RuntimeError("本任务要求使用 CUDA，禁止 CPU worker")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA 不可用，无法使用 {device_name}")
    configure_precision(precision)

    feature_cpu = torch.from_numpy(_SHARED_FEATURES)
    id_cpu = torch.from_numpy(_SHARED_ID_CODES)
    cache_enabled = should_cache_features(
        feature_cache_mode,
        _SHARED_FEATURES.nbytes,
        device,
    )
    if cache_enabled:
        feature_source = feature_cpu.to(device, non_blocking=False)
        id_source = id_cpu.to(device, non_blocking=False)
    else:
        feature_source = feature_cpu
        id_source = id_cpu

    positive_histogram = torch.zeros(histogram_bins, dtype=torch.int64, device=device)
    negative_histogram = torch.zeros(histogram_bins, dtype=torch.int64, device=device)
    positive_pair_count = 0
    extracted: dict[str, list[tuple[float, int, int]]] = {
        "negative_above": [],
        "positive_below": [],
    }
    per_task_limit = min(max_extracted, 256)

    for task in tasks:
        if cache_enabled:
            row_features = feature_source[task.row_start : task.row_end]
            col_features = feature_source[task.col_start : task.col_end]
            row_ids = id_source[task.row_start : task.row_end]
            col_ids = id_source[task.col_start : task.col_end]
        else:
            row_features = feature_source[task.row_start : task.row_end].to(device, non_blocking=False)
            col_features = feature_source[task.col_start : task.col_end].to(device, non_blocking=False)
            row_ids = id_source[task.row_start : task.row_end].to(device, non_blocking=False)
            col_ids = id_source[task.col_start : task.col_end].to(device, non_blocking=False)

        scores = torch.mm(row_features, col_features.T)
        scores.clamp_(min=-1.0, max=1.0)
        is_diagonal = task.row_block == task.col_block
        if is_diagonal:
            diagonal_indices = torch.arange(scores.shape[0], device=device)
            invalid = diagonal_indices[:, None] >= diagonal_indices[None, :]
            scores.masked_fill_(invalid, -2.0)

        all_histogram = _histogram(scores, histogram_bins)
        pair_coordinates = _SHARED_POSITIVE_PAIRS.get((task.row_block, task.col_block))
        if pair_coordinates is None:
            positive_scores = torch.empty(0, dtype=scores.dtype, device=device)
            positive_rows = torch.empty(0, dtype=torch.long, device=device)
            positive_cols = torch.empty(0, dtype=torch.long, device=device)
        else:
            positive_rows = torch.as_tensor(pair_coordinates[0], dtype=torch.long, device=device)
            positive_cols = torch.as_tensor(pair_coordinates[1], dtype=torch.long, device=device)
            positive_scores = scores[positive_rows, positive_cols]
            positive_histogram_for_task = _histogram(positive_scores, histogram_bins)
            positive_histogram.add_(positive_histogram_for_task)
            negative_histogram.sub_(positive_histogram_for_task)
            positive_pair_count += int(positive_scores.numel())

        negative_histogram.add_(all_histogram)

        if extract_above is not None:
            extracted["negative_above"].extend(
                _top_negative_above(
                    scores,
                    task,
                    row_ids,
                    col_ids,
                    extract_above,
                    per_task_limit,
                )
            )
        if extract_below is not None:
            extracted["positive_below"].extend(
                _top_positive_below(
                    positive_scores,
                    positive_rows,
                    positive_cols,
                    task,
                    extract_below,
                    per_task_limit,
                )
            )

        del scores

    torch.cuda.synchronize(device)

    # 仅保留每一类最有代表性的样本，避免 IPC 和主进程内存随任务数增长。
    extracted["negative_above"] = sorted(extracted["negative_above"], reverse=True)[:max_extracted]
    extracted["positive_below"] = sorted(extracted["positive_below"])[:max_extracted]
    return {
        "ok": True,
        "device": device_name,
        "task_count": len(tasks),
        "pair_count": sum(task.pair_count for task in tasks),
        "positive_pair_count": positive_pair_count,
        "positive_histogram": positive_histogram.cpu().numpy(),
        "negative_histogram": negative_histogram.cpu().numpy(),
        "extracted": extracted,
        "cache_enabled": cache_enabled,
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def worker_entry(
    result_queue: mp.queues.Queue,
    device_name: str,
    tasks: list[BlockTask],
    worker_config: dict[str, Any],
) -> None:
    """将子进程异常安全地返回给主进程。"""
    try:
        result = evaluate_worker(device_name=device_name, tasks=tasks, **worker_config)
    except BaseException:
        result = {"ok": False, "device": device_name, "error": traceback.format_exc()}
    result_queue.put(result)


def run_workers(
    devices: list[str],
    assignments: list[list[BlockTask]],
    worker_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """并行启动多个 GPU worker，并持续报告完成情况。"""
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise RuntimeError("本任务要求使用一个或多个 CUDA GPU，不接受 CPU worker。")
    if sys.platform != "linux":
        raise RuntimeError("多 GPU 共享特征实现依赖 Linux fork；请在 Linux GPU 节点运行。")

    context = mp.get_context("fork")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker_entry,
            args=(result_queue, device_name, tasks, worker_config),
            daemon=False,
        )
        for device_name, tasks in zip(devices, assignments)
    ]
    for process in processes:
        process.start()

    results: list[dict[str, Any]] = []
    while len(results) < len(processes):
        try:
            result = result_queue.get(timeout=1)
        except queue.Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                failed_codes = ", ".join(str(process.exitcode) for process in failed)
                raise RuntimeError(f"GPU worker 异常退出，退出码：{failed_codes}")
            continue
        results.append(result)
        if result.get("ok"):
            print(
                f"  {result['device']} 完成：{result['task_count']} 个块任务，"
                f"{result['pair_count']:,} 对，耗时 {result['elapsed_seconds']:.1f} 秒。",
                flush=True,
            )
        else:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(f"{result['device']} 执行失败：\n{result['error']}")

    for process in processes:
        process.join()
    return results


def total_pair_counts(id_codes: np.ndarray) -> tuple[int, int, int]:
    sample_count = len(id_codes)
    total_pairs = sample_count * (sample_count - 1) // 2
    _, counts = np.unique(id_codes, return_counts=True)
    positive_pairs = int(np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2))
    return total_pairs, positive_pairs, total_pairs - positive_pairs


def compute_tpir_at_fpir(
    positive_histogram: np.ndarray,
    negative_histogram: np.ndarray,
    targets: Iterable[float],
) -> list[dict[str, float]]:
    """根据直方图插值估计给定 FPIR 下的阈值和 TPIR。"""
    positive_total = int(positive_histogram.sum())
    negative_total = int(negative_histogram.sum())
    bin_count = len(negative_histogram)
    width = 2.0 / bin_count
    if negative_total == 0:
        return [
            {
                "target_fpir": target,
                "threshold": float("nan"),
                "tpir": float("nan"),
                "estimated_false_accepts": 0.0,
                "bin_width": width,
            }
            for target in targets
        ]
    negative_tail_from_high = np.cumsum(negative_histogram[::-1], dtype=np.int64)
    positive_tail_from_high = np.cumsum(positive_histogram[::-1], dtype=np.int64)
    records: list[dict[str, float]] = []

    for target in targets:
        desired_false_accepts = target * negative_total
        high_index = int(np.searchsorted(negative_tail_from_high, desired_false_accepts, side="left"))
        high_index = min(max(high_index, 0), bin_count - 1)
        bin_index = bin_count - 1 - high_index
        before_current_bin = int(negative_tail_from_high[high_index - 1]) if high_index else 0
        current_bin_count = int(negative_histogram[bin_index])
        fraction_above = (
            (desired_false_accepts - before_current_bin) / current_bin_count if current_bin_count else 0.0
        )
        fraction_above = min(max(fraction_above, 0.0), 1.0)
        bin_low = -1.0 + bin_index * width
        threshold = bin_low + (1.0 - fraction_above) * width

        positive_before_current_bin = int(positive_tail_from_high[high_index - 1]) if high_index else 0
        estimated_true_accepts = positive_before_current_bin + positive_histogram[bin_index] * fraction_above
        tpir_value = float(estimated_true_accepts / positive_total) if positive_total else float("nan")
        records.append(
            {
                "target_fpir": target,
                "threshold": float(threshold),
                "tpir": tpir_value,
                "estimated_false_accepts": float(desired_false_accepts),
                "bin_width": width,
            }
        )
    return records


def write_metrics_parquet(
    output_dir: Path,
    metric_records: list[dict[str, float]],
) -> None:
    import polars as pl

    pl.DataFrame(metric_records).write_parquet(output_dir / "tpir_at_fpir.parquet")


def write_error_pairs(
    output_dir: Path,
    extracted: dict[str, list[tuple[float, int, int]]],
    original_ids: np.ndarray,
    file_paths: list[str],
) -> None:
    """将提取的错误样本写成 Parquet，便于后续 Polars 分析。"""
    import polars as pl

    records: list[dict[str, Any]] = []
    for category, pairs in extracted.items():
        for score, left_index, right_index in pairs:
            records.append(
                {
                    "category": category,
                    "similarity": score,
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_id": str(original_ids[left_index]),
                    "right_id": str(original_ids[right_index]),
                    "left_path": file_paths[left_index],
                    "right_path": file_paths[right_index],
                }
            )
    if records:
        error_frame = pl.DataFrame(records)
    else:
        error_frame = pl.DataFrame(
            schema={
                "category": pl.String,
                "similarity": pl.Float64,
                "left_index": pl.Int64,
                "right_index": pl.Int64,
                "left_id": pl.String,
                "right_id": pl.String,
                "left_path": pl.String,
                "right_path": pl.String,
            }
        )
    error_frame.write_parquet(output_dir / "error_pairs.parquet")


def draw_plots(
    output_dir: Path,
    positive_histogram: np.ndarray,
    negative_histogram: np.ndarray,
    font_path: Path,
) -> None:
    """生成相似度分布和 TPIR-FPIR 曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    chinese_labels = False
    if font_path.is_file():
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            chinese_labels = True
        except (OSError, RuntimeError):
            pass
    if not chinese_labels:
        print(f"未找到可用中文字体：{font_path}，图表将使用英文标签。", file=sys.stderr)
    plt.rcParams["axes.unicode_minus"] = False

    labels = (
        {
            "negative": "负样本",
            "positive": "正样本",
            "similarity": "余弦相似度",
            "density": "概率密度（对数坐标）",
            "distribution_title": "正负样本相似度分布",
            "fpir": "FPIR（对数坐标）",
            "tpir": "TPIR",
            "curve_title": "TPIR@FPIR 曲线",
        }
        if chinese_labels
        else {
            "negative": "negative samples",
            "positive": "positive samples",
            "similarity": "cosine similarity",
            "density": "probability density (log scale)",
            "distribution_title": "Similarity distribution",
            "fpir": "FPIR (log scale)",
            "tpir": "TPIR",
            "curve_title": "TPIR@FPIR curve",
        }
    )

    bin_count = len(positive_histogram)
    width = 2.0 / bin_count
    centers = np.linspace(-1.0 + width / 2.0, 1.0 - width / 2.0, bin_count)
    positive_total = positive_histogram.sum()
    negative_total = negative_histogram.sum()
    positive_density = positive_histogram / positive_total / width if positive_total else np.zeros_like(positive_histogram, dtype=float)
    negative_density = negative_histogram / negative_total / width if negative_total else np.zeros_like(negative_histogram, dtype=float)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positive_mask = positive_density > 0
    negative_mask = negative_density > 0
    axis.semilogy(centers[negative_mask], negative_density[negative_mask], label=labels["negative"], linewidth=1.2)
    axis.semilogy(centers[positive_mask], positive_density[positive_mask], label=labels["positive"], linewidth=1.2)
    axis.set_xlabel(labels["similarity"])
    axis.set_ylabel(labels["density"])
    axis.set_title(labels["distribution_title"])
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "similarity_distribution.png", dpi=180)
    plt.close(figure)

    # 用各桶上边界对应的严格大于阈值生存率绘制曲线，并抽样控制渲染开销。
    fpir = (
        np.cumsum(negative_histogram[::-1], dtype=np.float64)[::-1] / negative_total
        if negative_total
        else np.zeros_like(negative_histogram, dtype=float)
    )
    tpir = (
        np.cumsum(positive_histogram[::-1], dtype=np.float64)[::-1] / positive_total
        if positive_total
        else np.zeros_like(positive_histogram, dtype=float)
    )
    step = max(1, bin_count // 20_000)
    valid = fpir[::step] > 0
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.semilogx(fpir[::step][valid], tpir[::step][valid], linewidth=1.5)
    axis.set_xlim(1e-7, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(labels["fpir"])
    axis.set_ylabel(labels["tpir"])
    axis.set_title(labels["curve_title"])
    axis.grid(True, which="both", alpha=0.25)
    figure.savefig(output_dir / "tpir_fpir_curve.png", dpi=180)
    plt.close(figure)


def main() -> int:
    global _SHARED_FEATURES, _SHARED_ID_CODES, _SHARED_POSITIVE_PAIRS

    args = parse_args()
    try:
        targets = parse_target_fpirs(args.fpir_targets)
    except ValueError as error:
        print(f"参数错误：{error}", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        print(f"输入文件不存在：{input_path}", file=sys.stderr)
        return 2

    print("=" * 80)
    print("大规模人脸特征相似度评估")
    print("=" * 80)
    load_started_at = time.perf_counter()
    features, original_ids, file_paths = load_features(input_path)
    _, id_codes = np.unique(original_ids, return_inverse=True)
    id_codes = np.ascontiguousarray(id_codes.astype(np.int64, copy=False))
    norms = np.linalg.norm(features, axis=1)
    total_pairs, expected_positive_pairs, expected_negative_pairs = total_pair_counts(id_codes)
    tasks = make_tasks(len(features), args.block_size)
    device_error: RuntimeError | ValueError | None = None
    try:
        devices = discover_devices(args.devices, args.max_gpus)
    except (RuntimeError, ValueError) as error:
        device_error = error
        devices = []
    if isinstance(device_error, ValueError) or (device_error is not None and not args.dry_run):
        print(f"设备配置错误：{device_error}", file=sys.stderr)
        return 2
    if devices and any(not device.startswith("cuda:") for device in devices):
        print("设备配置错误：必须使用一个或多个 CUDA GPU。", file=sys.stderr)
        return 2
    assignments = balance_tasks(tasks, len(devices)) if devices else []

    print(f"样本数：{len(features):,}，特征维度：{features.shape[1]}")
    print(f"身份数：{len(np.unique(id_codes)):,}")
    print(f"L2 范数：均值 {norms.mean():.6f}，标准差 {norms.std():.6f}")
    print(f"正样本对：{expected_positive_pairs:,}；负样本对：{expected_negative_pairs:,}")
    device_summary = ", ".join(devices) if devices else "CUDA 不可用（dry-run）"
    print(f"块大小：{args.block_size:,}，上三角块任务：{len(tasks):,}，设备：{device_summary}")
    print(f"直方图桶数：{args.hist_bins:,}，阈值分辨率：{2.0 / args.hist_bins:.1e}")
    print(f"数据准备耗时：{time.perf_counter() - load_started_at:.2f} 秒")

    if args.dry_run:
        print("dry-run 完成，未执行相似度计算。")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始分块相似度计算…", flush=True)
    _SHARED_FEATURES = features
    _SHARED_ID_CODES = id_codes
    index_started_at = time.perf_counter()
    _SHARED_POSITIVE_PAIRS = build_positive_pair_index(id_codes, args.block_size)
    indexed_positive_pairs = sum(len(rows) for rows, _ in _SHARED_POSITIVE_PAIRS.values())
    if indexed_positive_pairs != expected_positive_pairs:
        raise RuntimeError(
            f"正样本索引校验失败：{indexed_positive_pairs:,} != {expected_positive_pairs:,}"
        )
    print(
        f"正样本稀疏索引完成：{len(_SHARED_POSITIVE_PAIRS):,} 个非空块，"
        f"耗时 {time.perf_counter() - index_started_at:.2f} 秒。"
    )

    worker_config = {
        "histogram_bins": args.hist_bins,
        "precision": args.precision,
        "feature_cache_mode": args.gpu_feature_cache,
        "extract_above": args.extract_above,
        "extract_below": args.extract_below,
        "max_extracted": args.max_extracted,
    }
    evaluation_started_at = time.perf_counter()
    results = run_workers(devices, assignments, worker_config)
    evaluation_seconds = time.perf_counter() - evaluation_started_at

    positive_histogram = np.sum([result["positive_histogram"] for result in results], axis=0, dtype=np.int64)
    negative_histogram = np.sum([result["negative_histogram"] for result in results], axis=0, dtype=np.int64)
    actual_positive_pairs = int(positive_histogram.sum())
    actual_negative_pairs = int(negative_histogram.sum())
    if actual_positive_pairs != expected_positive_pairs or actual_negative_pairs != expected_negative_pairs:
        raise RuntimeError(
            "样本对统计校验失败："
            f"正样本 {actual_positive_pairs:,}/{expected_positive_pairs:,}，"
            f"负样本 {actual_negative_pairs:,}/{expected_negative_pairs:,}"
        )

    metric_records = compute_tpir_at_fpir(positive_histogram, negative_histogram, targets)
    print("\nTPIR@FPIR（基于直方图插值）：")
    for record in metric_records:
        print(
            f"  FPIR={record['target_fpir']:.0e}: TPIR={record['tpir'] * 100:.3f}%"
            f"，阈值≈{record['threshold']:.6f}"
        )

    merged_extracted: dict[str, list[tuple[float, int, int]]] = {
        "negative_above": [],
        "positive_below": [],
    }
    for result in results:
        for category, pairs in result["extracted"].items():
            merged_extracted[category].extend(pairs)
    merged_extracted["negative_above"] = sorted(merged_extracted["negative_above"], reverse=True)[: args.max_extracted]
    merged_extracted["positive_below"] = sorted(merged_extracted["positive_below"])[: args.max_extracted]

    write_metrics_parquet(output_dir, metric_records)
    if args.extract_above is not None or args.extract_below is not None:
        write_error_pairs(output_dir, merged_extracted, original_ids, file_paths)
        print(
            f"错误样本对：negative_above={len(merged_extracted['negative_above'])}，"
            f"positive_below={len(merged_extracted['positive_below'])}，"
            f"已写入 {output_dir / 'error_pairs.parquet'}"
        )
    if not args.no_plots:
        draw_plots(output_dir, positive_histogram, negative_histogram, Path(args.font_path))

    print(f"\n统计结果：{output_dir / 'tpir_at_fpir.parquet'}")
    if not args.no_plots:
        print(f"图表结果：{output_dir / 'similarity_distribution.png'}、{output_dir / 'tpir_fpir_curve.png'}")
    print(f"全量相似度评估耗时：{evaluation_seconds:.2f} 秒")
    print("评估完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
