#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基准 core: cluster_utils（仓库外参考实现；查找顺序：环境变量 CLUSTER_UTILS_PATH
指定路径 -> 仓库上级目录 cluster_utils.py）。

默认调用 get_sim_matrix_large_scale_v5（v4 的升级版：masked_fill+histc 替代布尔索引、
neg_hist=full_hist-pos_hist、可选 fp16 Tensor Core、显存更低；原 scripts/test_baseline
_cluster_utils.py 用的是 v4，可经环境变量切回复现旧口径）。
可用环境变量覆盖（实验/复现用）:
    CLUSTER_UTILS_VER=v4        # 回到 v4（历史基准口径）
    CLUSTER_UTILS_PREC=fp16     # v5 的 precision='fp16'
本 core 保留其内部 20M bins / 动态取块调度，仅把返回的 20M-bin 直方图按 factor=100
聚合到统一网格 200K bins。
"""
import os
import time

import numpy as np

from .. import common

MODEL_NAME = 'baseline'
MODEL_DESC = 'cluster_utils 基准方法（v5 动态调度大块矩阵乘 + 20M bins 直方图）'
ORIGIN = ('cluster_utils.py:get_sim_matrix_large_scale_v5（v4 升级版：masked_fill+histc、'
          'neg=full-pos、可选 fp16；默认 v5，可用环境变量切回 v4）')

_FACTOR = 20_000_000 // common.BINS


def _find_cluster_utils():
    cands = []
    env = os.environ.get('CLUSTER_UTILS_PATH')
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))          # benchmark/cores
    cands.append(os.path.join(here, '..', '..', '..', 'cluster_utils.py'))  # 仓库上级目录
    for p in cands:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError('找不到 cluster_utils.py；请设置环境变量 CLUSTER_UTILS_PATH')


def compute(feats, ids, gpus, workdir):
    import importlib.util
    cu_path = _find_cluster_utils()
    spec = importlib.util.spec_from_file_location('cluster_utils', cu_path)
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)

    # 版本/精度开关（实验 A/B 用；默认 v5 + fp32）
    #   CLUSTER_UTILS_VER=v4       使用 get_sim_matrix_large_scale_v4（历史口径）
    #   CLUSTER_UTILS_PREC=fp16    v5 的 precision 参数（fp16 Tensor Core）
    ver = os.environ.get('CLUSTER_UTILS_VER', 'v5')
    prec = os.environ.get('CLUSTER_UTILS_PREC', 'fp32')
    fn = getattr(cu, f'get_sim_matrix_large_scale_{ver}')
    kwargs = dict(precision=prec) if ver == 'v5' else {}

    t0 = time.perf_counter()
    pos20, neg20 = fn(
        query_feats_list=feats,
        query_ids=ids,
        num_gpus=len(gpus),
        block_size=2048 * 5,
        hist_bins=20_000_000,
        hist_range=(-1.0, 1.0),
        collect_pairs_config=None,
        memory_mode='low_memory',
        show_progress=False,
        **kwargs,
    )
    core_s = time.perf_counter() - t0

    pos_hist = common.rebin(np.asarray(pos20, dtype=np.int64), _FACTOR)
    neg_hist = common.rebin(np.asarray(neg20, dtype=np.int64), _FACTOR)
    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'native_bins': 20_000_000, 'rebin_factor': _FACTOR,
        'parallel': '动态共享任务池(spawn) 7卡',
        'cluster_utils_version': ver, 'precision': prec,
        'core_s': core_s,
    }
    return pos_hist, neg_hist, meta
