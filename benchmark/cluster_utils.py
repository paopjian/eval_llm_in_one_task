import os
import cv2

import pandas as pd
import pickle
from PIL import Image
from tqdm.auto import tqdm
import shutil

import torch
import numpy as np
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import math
import threading
import multiprocessing as mp
from collections import defaultdict, Counter
import heapq

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import onnxruntime as ort
from torchvision.transforms import Compose, ToTensor, Normalize

from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool



def get_all_files(root, extension_list=['.jpg']):
    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files




def get_matrix_plot(neg_socres, min_val=0.4, max_val=1.0, bins=50):
    import matplotlib.pyplot as plt
    from fast_histogram import histogram1d
    import numpy as np

    # 参数
    bins = 50
    min_val = 0.4
    max_val = 1.0

    # 计算直方图
    counts = histogram1d(neg_socres, bins=bins, range=[min_val, max_val])
    counts = counts.astype('int')
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (max_val - min_val) / bins

    # 创建多子图对比
    fig, axes = plt.subplots(1,1, figsize=(15, 10))

    # 子图1: 柱状图
    axes.bar(bin_centers, counts, width=bin_width, color='blue', alpha=0.7, edgecolor='black')
    axes.set_title('Bar Plot')
    axes.set_xlabel('Cosine Similarity')
    axes.set_ylabel('Frequency')
    axes.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"数据点总数: {counts.sum():,}")
    print(f"最大频率: {counts.max():,} (在区间 [{bin_edges[counts.argmax()]:.3f}, {bin_edges[counts.argmax()+1]:.3f}))")

# HDBSACN聚类
def hdbscan_cluster(image_list, image_index_dict, feat_ndarray, gpu_id=0):
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # gpu_id = 0
    import cupy as cp
    cp.cuda.Device(gpu_id % 8).use()
    device = 'cuda:'+ str(gpu_id % 8)
    from cuml.cluster import HDBSCAN
    def get_clustered_images(labels, image_list):
        # 分离正常簇和噪声点（label == -1 表示噪声）
        clustered_images = defaultdict(list)
        outliers = []

        for label, img_name in zip(labels, image_list):
            if label == -1:
                outliers.append(img_name)
            else:
                clustered_images[label].append(img_name)

        # 转为普通 dict
        clustered_images = dict(clustered_images)

        return clustered_images, outliers
    
    feat_indices = [image_index_dict[img] for img in image_list if img in image_index_dict]
    if len(feat_indices) <= 5:
        return [], []
    try:
        feats = [feat_ndarray[i] for i in feat_indices]
        feats = np.asarray(feats)
        feats = torch.from_numpy(feats).to(device)
        clusterer = HDBSCAN(min_cluster_size=5, 
                            min_samples=5,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            output_type='numpy')
        cluster_labels = clusterer.fit_predict(feats)

        clustered_images, outliers = get_clustered_images(cluster_labels, image_list)
    except Exception as e:
        print(f"HDBSCAN聚类报错 {gpu_id}: {e}")
        raise e
        return [], []
    return clustered_images, outliers

# 使用中心点进行噪音匹配, 选择与匹配图片数量最多且总相似度最高的最为主图, 高于阈值0.48视为相似, 
def get_cluster_by_center(image_list_to_process, image_index_dict, feat_ndarray, device):
    image_list = [img for img in image_list_to_process if img in image_index_dict]
    
    feats = [feat_ndarray[image_index_dict[img]] for img in image_list]
    feats = np.asarray(feats)
    feats_torch = torch.from_numpy(feats).to(device)
    sim_matrix = feats_torch @ feats_torch.t()

    threshold = 0.48
    # 使用 torch 进行后续计算,避免 CPU-GPU 来回传输
    sim_matrix_mask = (sim_matrix >= threshold)
    sim_counts = torch.sum(sim_matrix_mask, dim=1)

    # 找 sim_counts 中最大的, 可能有多个, 再比较 sim_sums 选最大的
    max_sim_count_indices = torch.where(sim_counts == sim_counts.max())[0]
    # 如果只有一个最大值,直接使用;否则比较 sim_sums
    if len(max_sim_count_indices) == 1:
        center_index = max_sim_count_indices[0]
    else:
        sim_sums = torch.sum(sim_matrix * sim_matrix_mask, dim=1)
        center_index = max_sim_count_indices[torch.argmax(sim_sums[max_sim_count_indices])]

    # 最后需要用到索引时再转回 CPU
    cluster_indices = torch.where(sim_matrix_mask[center_index])[0].cpu().numpy()
    cluster_image_list = [image_list[index] for index in cluster_indices]
    cluster_images = {0: cluster_image_list}
    outliers = set(image_list) - set(cluster_image_list)
    return cluster_images, list(outliers)


# 如果有多个簇, 选择图片数量最多的簇作为主簇, 
# 主簇平均特征与其他噪音图片匹配, 阈值设定0.48, 高于就视为相似图片
def get_main_cluster_and_match_noise(clustered_images, outliers, image_index_dict, feat_ndarray, device):
    if len(clustered_images) > 1:
        main_cluster_id = max(clustered_images, key=lambda k: len(clustered_images[k]))
        main_cluster_images = clustered_images[main_cluster_id]
        main_cluster_feats = [feat_ndarray[image_index_dict[img]] for img in main_cluster_images if img in image_index_dict]
        main_cluster_feats = np.asarray(main_cluster_feats)
        main_cluster_feats_mean = np.mean(main_cluster_feats, axis=0, keepdims=True)
        main_cluster_feats_mean = main_cluster_feats_mean / np.linalg.norm(main_cluster_feats_mean, axis=1, keepdims=True)
        main_cluster_feats_torch = torch.from_numpy(main_cluster_feats_mean).to(device)
        
        noise_images = outliers
        noise_images.extend([img for cid, imgs in clustered_images.items() if cid != main_cluster_id for img in imgs])
        noise_feats = [feat_ndarray[image_index_dict[img]] for img in noise_images if img in image_index_dict]
        noise_feats = np.asarray(noise_feats)
        noise_feats_torch = torch.from_numpy(noise_feats).to(device)
        
        sim_matrix = noise_feats_torch @ main_cluster_feats_torch.t()
        sim_matrix = sim_matrix.cpu().numpy()
        threshold = 0.48
        matched_noise_images = set()
        for i in range(sim_matrix.shape[0]):
            if sim_matrix[i][0] >= threshold:
                matched_noise_images.add((noise_images[i], sim_matrix[i][0]))  # 记录匹配的噪音图片和主簇中的一张图片
        if len(matched_noise_images) > 0:
            main_cluster_images.extend([img for img, sim in matched_noise_images])
    return main_cluster_images



class FolderMapper:
    """文件夹映射管理类，用于管理文件夹层级关系的缓存"""
    
    def __init__(self, base_dir: str = '/data/zkj-data/dataset_v3/all_aligned',
                 cache_path: str = '/dev/shm/v3_folder_mapping.pkl'):
        """
        初始化文件夹映射管理器
        
        Args:
            base_dir: 基础目录路径
            cache_path: 缓存文件路径
        """
        self.base_dir = base_dir
        self.cache_path = cache_path
        self._mapping = None
    
    def build_mapping(self) -> Dict[str, str]:
        """
        构建子文件夹到父文件夹的映射
        
        Returns:
            字典，key为子文件夹名，value为父文件夹名
        """
        folder_mapping = {}
        
        if not os.path.exists(self.base_dir):
            print(f"警告: 目录 {self.base_dir} 不存在")
            return folder_mapping
        
        # 遍历第一层文件夹（A文件夹）
        for parent_folder in os.listdir(self.base_dir):
            parent_path = os.path.join(self.base_dir, parent_folder)
            
            # 确保是文件夹
            if not os.path.isdir(parent_path):
                continue
                
            # 遍历第二层文件夹（B文件夹）
            for child_folder in os.listdir(parent_path):
                child_path = os.path.join(parent_path, child_folder)
                
                # 确保是文件夹
                if os.path.isdir(child_path):
                    folder_mapping[child_folder] = parent_folder
        
        return folder_mapping
    
    def save_mapping(self, folder_mapping: Optional[Dict[str, str]] = None):
        """
        保存文件夹映射到缓存文件
        
        Args:
            folder_mapping: 文件夹映射字典，如果为None则保存当前加载的映射
        """
        if folder_mapping is None:
            folder_mapping = self._mapping
        
        if folder_mapping is None:
            print("警告: 没有可保存的映射")
            return
        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(folder_mapping, f)
        print(f"文件夹映射已保存到 {self.cache_path}")
    
    def load_mapping(self, force_rebuild: bool = False) -> Dict[str, str]:
        """
        加载文件夹映射，如果缓存不存在则重新构建
        
        Args:
            force_rebuild: 是否强制重新构建
            
        Returns:
            文件夹映射字典
        """
        if not force_rebuild and os.path.exists(self.cache_path):
            print(f"从缓存加载文件夹映射: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self._mapping = pickle.load(f)
            print(f"共加载 {len(self._mapping)} 个子文件夹映射")
        else:
            print(f"构建新的文件夹映射...")
            self._mapping = self.build_mapping()
            self.save_mapping()
            print(f"共构建 {len(self._mapping)} 个子文件夹映射")
        
        return self._mapping
    
    def get_parent(self, child_folder: str) -> Optional[str]:
        """
        查询子文件夹对应的父文件夹
        
        Args:
            child_folder: 子文件夹名
            
        Returns:
            父文件夹名，如果未找到返回 None
        """
        if self._mapping is None:
            self.load_mapping()
        
        return self._mapping.get(child_folder)
    
    @property
    def mapping(self) -> Dict[str, str]:
        """获取映射字典，如果未加载则自动加载"""
        if self._mapping is None:
            self.load_mapping()
        return self._mapping
    
    def help():
        # 在其他文件中使用
        print('''
        from cluster_utils import FolderMapper

        # 方式1: 使用默认路径
        mapper = FolderMapper()

        # 查询父文件夹
        parent = mapper.get_parent('doc_name')
        print(f"父文件夹: {parent}")

        # 获取完整映射字典
        all_mapping = mapper.mapping

        # 方式2: 自定义路径
        mapper = FolderMapper(
            base_dir='/custom/path/to/aligned',
            cache_path='/custom/cache/path.pkl'
        )

        # 强制重建缓存
        mapper.load_mapping(force_rebuild=True)

        # 方式3: 批量查询
        doc_names = ['doc1', 'doc2', 'doc3']
        for doc in doc_names:
            parent = mapper.get_parent(doc)
            if parent:
                print(f"{doc} -> {parent}") ''')
        
# 单显卡获取相似度矩阵, 并区分正负样本
def get_similarity_matrix(feats_t, ids_t=None, threshold=0.48):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if type(feats_t) is np.ndarray:
        feats_t = torch.from_numpy(feats_t).to(device)
    if type(feats_t) is torch.Tensor:
        feats_t = feats_t.to(device)
    
    block_size = 2048*5
    N = feats_t.shape[0]
    neg_scores = []
    pos_scores = []
    high_sim_pairs = []  # 记录高相似度对 (i, j, score)
    # ids_t 是对应组的标签, 默认是全部不同, 这样输出的就是全部负样本
    if ids_t is None:
        ids_t = torch.arange(N, device=feats_t.device)
    
    for i in tqdm(range(0, N, block_size), desc='Processing blocks'):
        block1 = feats_t[i:i+block_size]
        ids1 = ids_t[i:i+block_size]

        for j in range(i, N, block_size):
            block2 = feats_t[j:j+block_size]
            ids2 = ids_t[j:j+block_size]

            # 相似度计算
            sim_block = block1 @ block2.T

            # 标签匹配
            label_eq = (ids1[:, None] == ids2[None, :])

            # 上三角 mask
            if i == j:
                mask = torch.triu(torch.ones_like(sim_block, dtype=torch.bool), diagonal=1)
            else:
                mask = torch.ones_like(sim_block, dtype=torch.bool)

            # 提取有效的相似度和标签
            flat_sim = sim_block[mask]
            flat_labels = label_eq[mask]

            # 正样本分数直接收集
            pos_sim = flat_sim[flat_labels]
            if len(pos_sim) > 0:
                pos_scores.append(pos_sim.half().cpu())
            neg_sim = flat_sim[~flat_labels]
            if len(neg_sim) > 0:
                neg_scores.append(neg_sim.half().cpu())
                
            # 找出高于阈值的负样本对
            high_sim_mask = (flat_sim > threshold) & (~flat_labels)
            if high_sim_mask.any():
                # 获取原始索引
                row_indices, col_indices = torch.where(mask)
                high_indices = torch.where(high_sim_mask)[0]
                
                for idx in high_indices:
                    local_i = row_indices[idx].item()
                    local_j = col_indices[idx].item()
                    global_i = i + local_i
                    global_j = j + local_j
                    score = flat_sim[idx].item()
                    high_sim_pairs.append((global_i, global_j, score))
                    
    return pos_scores, neg_scores, high_sim_pairs


class TriangularBlockScheduler:
    """上三角矩阵的块调度器,实现负载均衡的GPU分配"""
    
    def __init__(self, total_size: int, num_gpus: int, block_size: int = 128, print_info: bool = False):
        self.total_size = total_size
        self.num_gpus = num_gpus
        self.block_size = block_size
        self.num_blocks = math.ceil(total_size / block_size)
        self.print_info = print_info
        
        # 计算所有需要计算的块(上三角)
        self.blocks = self._generate_upper_triangular_blocks()
        
        # 为每个GPU分配块
        self.gpu_assignments = self._assign_blocks_to_gpus()
        
        
    def _generate_upper_triangular_blocks(self) -> List[Tuple[int, int]]:
        """生成所有上三角块的坐标 (row_block, col_block)"""
        blocks = []
        for row_block in range(self.num_blocks):
            for col_block in range(row_block, self.num_blocks):
                blocks.append((row_block, col_block))
        return blocks
    
    def _calculate_block_workload(self, row_block: int, col_block: int) -> int:
        """计算单个块的工作量(元素数量)"""
        row_start = row_block * self.block_size
        row_end = min((row_block + 1) * self.block_size, self.total_size)
        col_start = col_block * self.block_size
        col_end = min((col_block + 1) * self.block_size, self.total_size)
        
        # 修复: 无论是否是对角线块，底层 torch.matmul 都会计算完整的矩形区域
        # 因此实际的计算开销(FLOPs)是完整的面积
        return (row_end - row_start) * (col_end - col_start)
    
    def _assign_blocks_to_gpus(self) -> List[List[Tuple[int, int]]]:
        """使用贪心算法将块均衡分配给各GPU"""
        # 计算每个块的工作量
        block_workloads = [(block, self._calculate_block_workload(*block)) 
                          for block in self.blocks]
        
        # 按工作量降序排序(先分配大块)
        block_workloads.sort(key=lambda x: x[1], reverse=True)
        
        # 初始化每个GPU的分配
        gpu_assignments = [[] for _ in range(self.num_gpus)]
        gpu_workloads = [0.0] * self.num_gpus
        
        # 贪心分配:每次将块分配给当前工作量最小的GPU
        for block, workload in block_workloads:
            min_gpu = gpu_workloads.index(min(gpu_workloads))

            gpu_assignments[min_gpu].append(block)
            gpu_workloads[min_gpu] += workload
    
        
        # 打印负载均衡信息
        self._print_balance_info(gpu_workloads)
        
        return gpu_assignments
    
    
    def _print_balance_info(self, gpu_workloads: List[int]):
        if not self.print_info:
            return
        """打印负载均衡信息"""
        total_workload = sum(gpu_workloads)
        print(f"\n总块数: {len(self.blocks)} (矩阵大小: {self.num_blocks}x{self.num_blocks})")
        print(f"总工作量: {total_workload:,}")
        print("\nGPU负载分配:")
        for i, workload in enumerate(gpu_workloads):
            percentage = (workload / total_workload * 100) if total_workload > 0 else 0
            num_blocks = len(self.gpu_assignments[i]) if hasattr(self, 'gpu_assignments') else 0
            print(f"  GPU {i}: {workload:,} ({percentage:.2f}%) - {num_blocks} 块")
        
        if len(gpu_workloads) > 1:
            max_workload = max(gpu_workloads)
            min_workload = min(gpu_workloads)
            imbalance = (max_workload - min_workload) / max_workload * 100 if max_workload > 0 else 0
            print(f"\n负载不均衡度: {imbalance:.2f}%")
    
    def get_gpu_blocks(self, gpu_id: int) -> List[Tuple[int, int, int, int]]:
        """获取指定GPU需要计算的所有块的实际坐标范围
        
        Returns:
            List of (row_start, row_end, col_start, col_end)
        """
        blocks = []
        for row_block, col_block in self.gpu_assignments[gpu_id]:
            row_start = row_block * self.block_size
            row_end = min((row_block + 1) * self.block_size, self.total_size)
            col_start = col_block * self.block_size
            col_end = min((col_block + 1) * self.block_size, self.total_size)
            blocks.append((row_start, row_end, col_start, col_end))
        return blocks

    def example_usage(self):
        # 示例1: 2x2块(总共3个上三角块)
        print("=" * 60)
        print("示例1: 矩阵256, 块大小128, 2个GPU")
        scheduler = TriangularBlockScheduler(total_size=256, num_gpus=2, block_size=128)
        
        for gpu_id in range(scheduler.num_gpus):
            print(f"\nGPU {gpu_id} 的计算块:")
            for row_start, row_end, col_start, col_end in scheduler.get_gpu_blocks(gpu_id):
                print(f"  [{row_start}:{row_end}, {col_start}:{col_end}]")
        
        # 示例2: 8x8块(总共36个上三角块)
        print("\n" + "=" * 60)
        print("示例2: 矩阵1024, 块大小128, 4个GPU")
        scheduler = TriangularBlockScheduler(total_size=1024, num_gpus=4, block_size=128)
        
        for gpu_id in range(scheduler.num_gpus):
            blocks = scheduler.get_gpu_blocks(gpu_id)
            print(f"\nGPU {gpu_id}: {len(blocks)} 个块")

def get_pos_neg_similarities_core(feats_t, ids_t, blocks, gpu_id, N,
                                  topk=None, threshold=None, 
                                  return_stats_only=False, return_pairs_only=False,
                                  pbar=None, pbar_lock=None):
    """核心计算函数 - 优化 topk 性能"""
    
    if return_stats_only:
        pos_count = 0
        neg_count = 0
        high_sim_count = 0
    elif return_pairs_only and threshold is not None:
        pos_count = 0
        neg_count = 0
        high_sim_pairs = []
    else:
        pos_scores = []
        neg_scores = []
        high_sim_pairs = [] if threshold is not None else None
        
        if topk is not None:
            # 在 GPU 上维护当前的 topk 负样本
            current_topk_neg = None
    
    # 遍历该GPU分配到的所有块
    for row_start, row_end, col_start, col_end in blocks:
        block1 = feats_t[row_start:row_end]
        ids1 = ids_t[row_start:row_end]
        
        block2 = feats_t[col_start:col_end]
        ids2 = ids_t[col_start:col_end]
        
        sim_block = block1 @ block2.T
        label_eq = (ids1[:, None] == ids2[None, :])
        
        if row_start == col_start:
            mask = torch.triu(torch.ones_like(sim_block, dtype=torch.bool), diagonal=1)
        else:
            mask = torch.ones_like(sim_block, dtype=torch.bool)
        
        flat_sim = sim_block[mask]
        flat_labels = label_eq[mask]
        
        # 正样本处理
        pos_sim = flat_sim[flat_labels]
        if return_stats_only or (return_pairs_only and threshold is not None):
            pos_count += len(pos_sim)
        else:
            if len(pos_sim) > 0:
                pos_scores.append(pos_sim.half().cpu())
        
        # 负样本处理
        neg_sim = flat_sim[~flat_labels]
        
        if len(neg_sim) > 0:
            if return_stats_only:
                neg_count += len(neg_sim)
                if threshold is not None:
                    high_sim_count += (neg_sim > threshold).sum().item()
            elif return_pairs_only and threshold is not None:
                neg_count += len(neg_sim)
                high_sim_mask = neg_sim > threshold
                if high_sim_mask.any():
                    # 一次性获取所有需要的索引(在GPU上完成)
                    row_indices, col_indices = torch.where(mask)
                    neg_mask_positions = torch.where(~flat_labels)[0]
                    high_mask_positions = torch.where(high_sim_mask)[0]
                    
                    # 通过索引映射获取原始位置
                    selected_mask_positions = neg_mask_positions[high_mask_positions]
                    local_i = row_indices[selected_mask_positions]
                    local_j = col_indices[selected_mask_positions]
                    
                    # 计算全局索引
                    global_i = local_i + row_start
                    global_j = local_j + col_start
                    scores = neg_sim[high_mask_positions]
                    
                    # 一次性转移到CPU并构造列表
                    global_i_cpu = global_i.cpu().numpy()
                    global_j_cpu = global_j.cpu().numpy()
                    scores_cpu = scores.cpu().numpy()
                    
                    # 使用列表推导式批量添加(比循环append快)
                    high_sim_pairs.extend([
                        (int(i), int(j), float(s)) 
                        for i, j, s in zip(global_i_cpu, global_j_cpu, scores_cpu)
                    ])
            else:
                if topk is not None:
                    # 增量式维护 topk: 每次与当前 topk 合并后再取 topk
                    if current_topk_neg is None:
                        # 第一次: 直接取当前块的 topk
                        if len(neg_sim) > topk:
                            current_topk_neg = neg_sim.topk(topk).values
                        else:
                            current_topk_neg = neg_sim
                    else:
                        # 合并当前块的负样本与已有的 topk
                        if len(neg_sim) > topk * 2:
                            # 当前块负样本过多,先粗筛
                            topk_local = min(topk * 2, len(neg_sim))
                            neg_sim = neg_sim.topk(topk_local).values
                        
                        # 合并后取 topk
                        merged = torch.cat([current_topk_neg, neg_sim])
                        if len(merged) > topk:
                            current_topk_neg = merged.topk(topk).values
                        else:
                            current_topk_neg = merged
                    
                elif threshold is not None:
                    high_sim_mask = neg_sim > threshold
                    if high_sim_mask.any():
                        # 一次性获取所有需要的索引(在GPU上完成)
                        row_indices, col_indices = torch.where(mask)
                        neg_mask_positions = torch.where(~flat_labels)[0]
                        high_mask_positions = torch.where(high_sim_mask)[0]
                        
                        # 通过索引映射获取原始位置
                        selected_mask_positions = neg_mask_positions[high_mask_positions]
                        local_i = row_indices[selected_mask_positions]
                        local_j = col_indices[selected_mask_positions]
                        
                        # 计算全局索引
                        global_i = local_i + row_start
                        global_j = local_j + col_start
                        scores = neg_sim[high_mask_positions]
                        
                        # 一次性转移到CPU并构造列表
                        global_i_cpu = global_i.cpu().numpy()
                        global_j_cpu = global_j.cpu().numpy()
                        scores_cpu = scores.cpu().numpy()
                        
                        # 使用列表推导式批量添加(比循环append快)
                        high_sim_pairs.extend([
                            (int(i), int(j), float(s)) 
                            for i, j, s in zip(global_i_cpu, global_j_cpu, scores_cpu)
                        ])
                    neg_scores.append(neg_sim.half().cpu())
                else:
                    neg_scores.append(neg_sim.half().cpu())
        
        # 更新进度条
        if pbar is not None and pbar_lock is not None:
            with pbar_lock:
                pbar.update(1)
                pbar.set_postfix({'GPU': gpu_id})
    
    if not return_stats_only and not return_pairs_only:
        pos_scores = torch.cat(pos_scores).numpy() if pos_scores else np.array([])
        
        if topk is not None:
            # 最终结果已经是 topk,直接转换
            if current_topk_neg is not None:
                neg_scores = current_topk_neg.half().cpu().numpy()
                # 降序排列
                neg_scores = np.sort(neg_scores)[::-1]
            else:
                neg_scores = np.array([])
            return pos_scores, neg_scores
        elif threshold is not None:
            neg_scores = torch.cat(neg_scores).numpy() if neg_scores else np.array([])
            return pos_scores, neg_scores, high_sim_pairs
        else:
            neg_scores = torch.cat(neg_scores).numpy() if neg_scores else np.array([])
            return pos_scores, neg_scores
    
    # 其他模式的返回
    if return_stats_only:
        return pos_count, neg_count, high_sim_count
    elif return_pairs_only and threshold is not None:
        return pos_count, neg_count, high_sim_pairs
    

def get_sim_matrix_batch_balanced(query_feats_list, query_ids=None, num_gpus=7, 
                                  block_size=2048*5, topk=None, threshold=None, 
                                  show_progress=True, return_stats_only=True, return_pairs_only=False):
    """
    使用块调度的负载均衡方式计算所有正负样本
    """
    if topk is not None and threshold is not None:
        raise ValueError("topk和threshold参数互斥,不能同时设置")
    
    if return_pairs_only and threshold is None:
        raise ValueError("return_pairs_only=True时必须设置threshold参数")
    
    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))
    
    N = len(query_ids)
    total_pairs = N * (N - 1) // 2
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs
    print(f"总对数: {total_pairs}, 正样本对数: {total_pos_pairs}, 负样本对数: {total_neg_pairs}")
    
    if topk is not None:
        print(f"TOPK模式: 每个GPU维护top-{topk}负样本")
    elif threshold is not None:
        print(f"THRESHOLD模式: 记录相似度 > {threshold} 的负样本对")
    
    if return_stats_only:
        print("统计模式: 只返回数据量,不收集实际数据")
    elif return_pairs_only:
        print("PAIRS-ONLY模式: 只返回high_sim_pairs,pos和neg仅返回统计数量")
    
    scheduler = TriangularBlockScheduler(total_size=N, num_gpus=num_gpus, block_size=block_size)
    total_blocks = len(scheduler.blocks)
    
    # ===== 关键优化1: 预先将数据转换为tensor,避免重复转换 =====
    print("正在准备数据...")
    prep_start = time.time()
    if type(query_feats_list) is list:
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif type(query_feats_list) is np.ndarray:
        query_feats_tensor = torch.from_numpy(query_feats_list)
    elif type(query_feats_list) is torch.Tensor:
        query_feats_tensor = query_feats_list
    else:
        raise ValueError("不支持的特征数据类型")
    
    # 将tensor放在CPU,pin_memory可以加速GPU传输
    query_feats_tensor = query_feats_tensor.cpu().pin_memory()
    print(f"数据准备耗时: {time.time() - prep_start:.2f} 秒")
    
    start = time.time()
    results = []
    
    # 创建共享的进度条和锁
    pbar = tqdm(total=total_blocks, desc="总体进度", disable=not show_progress)
    pbar_lock = threading.Lock()
    
    # ===== 关键优化2: 添加初始化进度提示 =====
    init_pbar = tqdm(total=num_gpus, desc="初始化GPU", disable=not show_progress)
    
    def get_pos_neg_with_init(query_feats_tensor, query_ids, blocks, gpu_id, N, 
                              topk, threshold, return_stats_only, return_pairs_only, 
                              pbar, pbar_lock, init_pbar):
        """包装函数,添加初始化进度反馈"""
        # 数据传输到GPU
        device = f'cuda:{gpu_id}'
        feats_t = query_feats_tensor.to(device, non_blocking=True)
        ids_t = torch.tensor(query_ids).to(device)
        
        # 标记该GPU初始化完成
        if init_pbar is not None:
            init_pbar.update(1)
            init_pbar.set_postfix({'GPU': gpu_id, 'status': 'ready'})
        
        # 调用原始计算函数(需要修改为接受tensor)
        return get_pos_neg_similarities_core(
            feats_t, ids_t, blocks, gpu_id, N,
            topk, threshold, return_stats_only, return_pairs_only,
            pbar, pbar_lock
        )
    
    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    get_pos_neg_with_init,
                    query_feats_tensor,  # 传入已准备好的tensor
                    query_ids,
                    scheduler.get_gpu_blocks(gpu_id),
                    gpu_id,
                    N,
                    topk,
                    threshold,
                    return_stats_only,
                    return_pairs_only,
                    pbar if show_progress else None,
                    pbar_lock if show_progress else None,
                    init_pbar if show_progress else None
                ): gpu_id for gpu_id in range(num_gpus)
            }
            
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if show_progress:
            init_pbar.close()
        pbar.close()
    
    print(f"计算矩阵耗时: {time.time() - start:.2f} 秒")
    
    # ...existing code...处理返回结果的代码保持不变...
    if return_stats_only:
        pos_count = sum(result[0] for result in results)
        neg_count = sum(result[1] for result in results)
        high_sim_count = sum(result[2] for result in results)
        
        print(f"正样本对数量: {pos_count}, 负样本对数量: {neg_count}")
        if threshold is not None:
            print(f"高相似度负样本对数量: {high_sim_count}")
        
        return pos_count, neg_count, high_sim_count
    
    elif return_pairs_only and threshold is not None:
        pos_count = sum(result[0] for result in results)
        neg_count = sum(result[1] for result in results)
        all_high_sim_pairs = []
        
        for _, _, pairs in results:
            if pairs:
                all_high_sim_pairs.extend(pairs)
        
        all_high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"正样本对数量: {pos_count}, 负样本对数量: {neg_count}")
        print(f"高相似度负样本对数量: {len(all_high_sim_pairs)}")
        
        return pos_count, neg_count, all_high_sim_pairs
    
    elif topk is not None:
        all_pos_scores = []
        all_neg_topk = []
        
        for pos, neg in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_topk.extend(neg)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        
        # if len(all_neg_topk) > topk:
        #     neg_scores = np.array(heapq.nlargest(topk, all_neg_topk))
        if len(all_neg_topk) > topk:
            all_neg_topk = np.array(all_neg_topk)
            idx = np.argpartition(all_neg_topk, -topk)[-topk:]
            neg_scores = all_neg_topk[idx].tolist()
        else:
            neg_scores = np.array(sorted(all_neg_topk, reverse=True))
        
        print(f"正样本对数量: {len(pos_scores)}, Top-{topk} 负样本对数量: {len(neg_scores)}")
        return pos_scores, neg_scores, None
    
    elif threshold is not None:
        all_pos_scores = []
        all_neg_scores = []
        all_high_sim_pairs = []
        
        for pos, neg, pairs in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_scores.append(neg)
            if pairs:
                all_high_sim_pairs.extend(pairs)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        neg_scores = np.concatenate(all_neg_scores) if all_neg_scores else np.array([])
        
        all_high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"正样本对数量: {len(pos_scores)}, 负样本对数量: {len(neg_scores)}")
        print(f"高相似度负样本对数量: {len(all_high_sim_pairs)}")
        return pos_scores, neg_scores, all_high_sim_pairs
    
    else:
        all_pos_scores = []
        all_neg_scores = []
        
        for pos, neg in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_scores.append(neg)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        neg_scores = np.concatenate(all_neg_scores) if all_neg_scores else np.array([])
        
        print(f"正样本对数量: {len(pos_scores)}, 负样本对数量: {len(neg_scores)}")
        return pos_scores, neg_scores, None
    
# 不输出任何信息的版本
def get_sim_matrix_batch_balanced_silent(query_feats_list, query_ids=None, num_gpus=7, 
                                  block_size=2048*5, topk=None, threshold=None, 
                                  show_progress=False, return_stats_only=True, return_pairs_only=False):
    """
    使用块调度的负载均衡方式计算所有正负样本
    """
    if topk is not None and threshold is not None:
        raise ValueError("topk和threshold参数互斥,不能同时设置")
    
    if return_pairs_only and threshold is None:
        raise ValueError("return_pairs_only=True时必须设置threshold参数")
    
    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))
    
    N = len(query_ids)
    total_pairs = N * (N - 1) // 2
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs
    
    scheduler = TriangularBlockScheduler(total_size=N, num_gpus=num_gpus, block_size=block_size, print_info=True)
    total_blocks = len(scheduler.blocks)
    
    # ===== 关键优化1: 预先将数据转换为tensor,避免重复转换 =====
    prep_start = time.time()
    if type(query_feats_list) is list:
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif type(query_feats_list) is np.ndarray:
        query_feats_tensor = torch.from_numpy(query_feats_list)
    elif type(query_feats_list) is torch.Tensor:
        query_feats_tensor = query_feats_list
    else:
        raise ValueError("不支持的特征数据类型")
    
    # 将tensor放在CPU,pin_memory可以加速GPU传输
    query_feats_tensor = query_feats_tensor.cpu().pin_memory()
    # print(f"数据准备耗时: {time.time() - prep_start:.2f} 秒")
    
    start = time.time()
    results = []
    
    # 创建共享的进度条和锁
    pbar = tqdm(total=total_blocks, desc="总体进度", disable=not show_progress)
    pbar_lock = threading.Lock()
        
    # ===== 关键优化2: 添加初始化进度提示 =====
    init_pbar = tqdm(total=num_gpus, desc="初始化GPU", disable=not show_progress)
    
    def get_pos_neg_with_init(query_feats_tensor, query_ids, blocks, gpu_id, N, 
                              topk, threshold, return_stats_only, return_pairs_only, 
                              pbar, pbar_lock, init_pbar):
        """包装函数,添加初始化进度反馈"""
        
        # 数据传输到GPU
        device = f'cuda:{gpu_id}'
        feats_t = query_feats_tensor.to(device, non_blocking=True)
        ids_t = torch.tensor(query_ids).to(device)
        
        # 标记该GPU初始化完成
        if init_pbar is not None:
            init_pbar.update(1)
            init_pbar.set_postfix({'GPU': gpu_id, 'status': 'ready'})
        
        # 调用原始计算函数(需要修改为接受tensor)
        try:
            result = get_pos_neg_similarities_core(
                feats_t, ids_t, blocks, gpu_id, N,
                topk, threshold, return_stats_only, return_pairs_only,
                pbar, pbar_lock
            )
            return result
        finally:
            # 显式删除GPU上的tensor
            del feats_t
            del ids_t
            # 清空该GPU的缓存
            torch.cuda.empty_cache()
        
    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    get_pos_neg_with_init,
                    query_feats_tensor,  # 传入已准备好的tensor
                    query_ids,
                    scheduler.get_gpu_blocks(gpu_id),
                    gpu_id,
                    N,
                    topk,
                    threshold,
                    return_stats_only,
                    return_pairs_only,
                    pbar if show_progress else None,
                    pbar_lock if show_progress else None,
                    init_pbar if show_progress else None
                ): gpu_id for gpu_id in range(num_gpus)
            }
            
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if show_progress:
            init_pbar.close()
        pbar.close()
        
    del query_feats_tensor
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
    
    # ...existing code...处理返回结果的代码保持不变...
    if return_stats_only:
        pos_count = sum(result[0] for result in results)
        neg_count = sum(result[1] for result in results)
        high_sim_count = sum(result[2] for result in results)
                
        return pos_count, neg_count, high_sim_count
    
    elif return_pairs_only and threshold is not None:
        pos_count = sum(result[0] for result in results)
        neg_count = sum(result[1] for result in results)
        all_high_sim_pairs = []
        
        for _, _, pairs in results:
            if pairs:
                all_high_sim_pairs.extend(pairs)
        
        all_high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
        return pos_count, neg_count, all_high_sim_pairs
    
    elif topk is not None:
        all_pos_scores = []
        all_neg_topk = []
        
        for pos, neg in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_topk.extend(neg)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        
        # if len(all_neg_topk) > topk:
        #     neg_scores = np.array(heapq.nlargest(topk, all_neg_topk))
        if len(all_neg_topk) > topk:
            all_neg_topk = np.array(all_neg_topk)
            idx = np.argpartition(all_neg_topk, -topk)[-topk:]
            neg_scores = all_neg_topk[idx].tolist()
        else:
            neg_scores = np.array(sorted(all_neg_topk, reverse=True))
        
        return pos_scores, neg_scores, None
    
    elif threshold is not None:
        all_pos_scores = []
        all_neg_scores = []
        all_high_sim_pairs = []
        
        for pos, neg, pairs in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_scores.append(neg)
            if pairs:
                all_high_sim_pairs.extend(pairs)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        neg_scores = np.concatenate(all_neg_scores) if all_neg_scores else np.array([])
        
        all_high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pos_scores, neg_scores, all_high_sim_pairs
    
    else:
        all_pos_scores = []
        all_neg_scores = []
        
        for pos, neg in results:
            if len(pos) > 0:
                all_pos_scores.append(pos)
            if len(neg) > 0:
                all_neg_scores.append(neg)
        
        pos_scores = np.concatenate(all_pos_scores) if all_pos_scores else np.array([])
        neg_scores = np.concatenate(all_neg_scores) if all_neg_scores else np.array([])
        return pos_scores, neg_scores, None


# ============================================================
# V3: 基于直方图的大规模相似度矩阵计算
# ============================================================

class PairCollector:
    """线程安全的样本对收集器，支持数量限制"""

    def __init__(self, max_pairs: int = -1):
        self.max_pairs = max_pairs
        self.pairs = []
        self.lock = threading.Lock()
        self._is_full = False

    def is_full(self) -> bool:
        if self.max_pairs == -1:
            return False
        return self._is_full

    def add_pairs(self, new_pairs) -> int:
        if self.is_full() or not new_pairs:
            return 0
        with self.lock:
            if self._is_full:
                return 0
            if self.max_pairs == -1:
                self.pairs.extend(new_pairs)
                return len(new_pairs)
            else:
                remaining = self.max_pairs - len(self.pairs)
                if remaining <= 0:
                    self._is_full = True
                    return 0
                to_add = new_pairs[:remaining]
                self.pairs.extend(to_add)
                if len(self.pairs) >= self.max_pairs:
                    self._is_full = True
                return len(to_add)

    def get_pairs(self):
        return self.pairs

    def count(self) -> int:
        return len(self.pairs)


def _collect_pairs(mask, flat_labels, target_mask, sim_values,
                   row_start, col_start, pair_collector, is_positive):
    """辅助函数：从Block中收集符合条件的样本对"""
    if pair_collector.is_full():
        return
    block_row_indices, block_col_indices = torch.where(mask)
    if is_positive:
        sample_indices_in_flat = torch.where(flat_labels)[0]
    else:
        sample_indices_in_flat = torch.where(~flat_labels)[0]
    target_indices_in_sample = torch.where(target_mask)[0]
    target_indices_in_flat = sample_indices_in_flat[target_indices_in_sample]
    local_i = block_row_indices[target_indices_in_flat]
    local_j = block_col_indices[target_indices_in_flat]
    global_i = local_i + row_start
    global_j = local_j + col_start
    scores = sim_values[target_mask]
    g_i_cpu = global_i.cpu().numpy()
    g_j_cpu = global_j.cpu().numpy()
    s_cpu = scores.cpu().numpy()
    current_pairs = list(zip(g_i_cpu, g_j_cpu, s_cpu))
    pair_collector.add_pairs(current_pairs)


def get_pos_neg_similarities_large_scale_core_v3(
    feats, ids, blocks, gpu_id,
    hist_bins=20_000_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None,
    pair_collector=None, pos_pair_collector=None, neg_pair_collector=None,
    memory_mode='low_memory', pbar=None, pbar_lock=None
):
    """计算相似度矩阵的核心函数 v3 (单GPU核心)"""
    with torch.cuda.device(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')

        # 解析收集配置
        do_collect_single = False
        do_collect_dual = False
        if collect_pairs_config is not None:
            if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
                do_collect_dual = True
                pos_cfg = collect_pairs_config.get('pos', None)
                neg_cfg = collect_pairs_config.get('neg', None)
            elif pair_collector is not None:
                do_collect_single = True
                sample_type = collect_pairs_config.get('sample_type', 'neg')
                threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
                threshold = collect_pairs_config.get('threshold', 0.5)

        # 准备 ids
        if torch.is_tensor(ids):
            ids_full = ids.to(device)
        else:
            ids_full = torch.tensor(ids, device=device)

        # 准备 feats
        use_gpu_feats = False
        feats_source = feats
        if memory_mode == 'high_performance':
            try:
                feats_source = feats.to(device)
                use_gpu_feats = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    feats_source = feats
                else:
                    raise e

        pos_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)
        neg_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)

        for row_start, row_end, col_start, col_end in blocks:
            if do_collect_single:
                collector_full = pair_collector.is_full()
            elif do_collect_dual:
                pos_collector_full = pos_pair_collector is None or pos_pair_collector.is_full() if pos_cfg else True
                neg_collector_full = neg_pair_collector is None or neg_pair_collector.is_full() if neg_cfg else True
            else:
                collector_full = True

            if use_gpu_feats:
                block1 = feats_source[row_start:row_end]
                block2 = feats_source[col_start:col_end]
            else:
                block1 = feats_source[row_start:row_end].to(device, non_blocking=True)
                block2 = feats_source[col_start:col_end].to(device, non_blocking=True)

            ids1 = ids_full[row_start:row_end]
            ids2 = ids_full[col_start:col_end]

            sim_block = torch.matmul(block1, block2.T)
            label_eq = (ids1[:, None] == ids2[None, :])

            if row_start == col_start:
                mask = torch.triu(torch.ones_like(sim_block, dtype=torch.bool), diagonal=1)
            else:
                mask = torch.ones_like(sim_block, dtype=torch.bool)

            flat_sim = sim_block[mask]
            flat_labels = label_eq[mask]

            # 正样本
            pos_sim = flat_sim[flat_labels]
            if pos_sim.numel() > 0:
                hist_counts = torch.histc(pos_sim, bins=hist_bins, min=hist_range[0], max=hist_range[1])
                pos_hist += hist_counts.long()
                if do_collect_single and not collector_full and sample_type == 'pos':
                    target_mask = pos_sim > threshold if threshold_mode == 'above' else pos_sim < threshold
                    if target_mask.any():
                        _collect_pairs(mask, flat_labels, target_mask, pos_sim, row_start, col_start, pair_collector, True)
                if do_collect_dual and pos_cfg and not pos_collector_full:
                    pos_tmode = pos_cfg.get('threshold_mode', 'below')
                    pos_thresh = pos_cfg.get('threshold', 0.25)
                    target_mask = pos_sim > pos_thresh if pos_tmode == 'above' else pos_sim < pos_thresh
                    if target_mask.any():
                        _collect_pairs(mask, flat_labels, target_mask, pos_sim, row_start, col_start, pos_pair_collector, True)

            # 负样本
            neg_sim = flat_sim[~flat_labels]
            if neg_sim.numel() > 0:
                hist_counts = torch.histc(neg_sim, bins=hist_bins, min=hist_range[0], max=hist_range[1])
                neg_hist += hist_counts.long()
                if do_collect_single and not collector_full and sample_type == 'neg':
                    target_mask = neg_sim > threshold if threshold_mode == 'above' else neg_sim < threshold
                    if target_mask.any():
                        _collect_pairs(mask, flat_labels, target_mask, neg_sim, row_start, col_start, pair_collector, False)
                if do_collect_dual and neg_cfg and not neg_collector_full:
                    neg_tmode = neg_cfg.get('threshold_mode', 'above')
                    neg_thresh = neg_cfg.get('threshold', 0.5)
                    target_mask = neg_sim > neg_thresh if neg_tmode == 'above' else neg_sim < neg_thresh
                    if target_mask.any():
                        _collect_pairs(mask, flat_labels, target_mask, neg_sim, row_start, col_start, neg_pair_collector, False)

            del block1, block2, sim_block, mask, flat_sim, flat_labels
            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

    return pos_hist.cpu(), neg_hist.cpu()


def get_sim_matrix_large_scale_v3(
    query_feats_list, query_ids=None, num_gpus=7, block_size=2048*5,
    hist_bins=20_000_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None, memory_mode='low_memory', show_progress=True
):
    """
    基于直方图的大规模相似度矩阵计算 v3 (多GPU并行)

    Returns:
        单收集模式: (pos_hist, neg_hist, collected_pairs)
        双收集模式: (pos_hist, neg_hist, pos_collected, neg_collected)
        无收集: (pos_hist, neg_hist)
    """
    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))

    N = len(query_ids)
    scheduler = TriangularBlockScheduler(total_size=N, num_gpus=num_gpus, block_size=block_size)
    total_blocks = len(scheduler.blocks)

    print(f"正在准备数据 (Mode: {memory_mode})...")
    prep_start = time.time()
    if isinstance(query_feats_list, list):
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif isinstance(query_feats_list, np.ndarray):
        query_feats_tensor = torch.from_numpy(query_feats_list)
    else:
        query_feats_tensor = query_feats_list

    if memory_mode == 'low_memory':
        if query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.cpu()
        query_feats_tensor = query_feats_tensor.float().contiguous().pin_memory()
    else:
        if not query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.float().pin_memory()
    print(f"数据准备耗时: {time.time() - prep_start:.2f} 秒")

    # 收集器准备
    pair_collector = None
    pos_pair_collector = None
    neg_pair_collector = None
    is_dual_mode = False

    if collect_pairs_config is not None:
        if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
            is_dual_mode = True
            if 'pos' in collect_pairs_config:
                pos_max = collect_pairs_config['pos'].get('max_pairs', -1)
                pos_pair_collector = PairCollector(max_pairs=pos_max)
            if 'neg' in collect_pairs_config:
                neg_max = collect_pairs_config['neg'].get('max_pairs', -1)
                neg_pair_collector = PairCollector(max_pairs=neg_max)
        else:
            max_pairs = collect_pairs_config.get('max_pairs', -1)
            pair_collector = PairCollector(max_pairs=max_pairs)

    start = time.time()
    results = []
    pbar = tqdm(total=total_blocks, desc="Matrix Cal v3", disable=not show_progress)
    pbar_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for gpu_id in range(num_gpus):
                gpu_blocks = scheduler.get_gpu_blocks(gpu_id)
                if not gpu_blocks:
                    continue
                futures[executor.submit(
                    get_pos_neg_similarities_large_scale_core_v3,
                    query_feats_tensor, query_ids, gpu_blocks, gpu_id,
                    hist_bins, hist_range, collect_pairs_config,
                    pair_collector, pos_pair_collector, neg_pair_collector,
                    memory_mode,
                    pbar if show_progress else None,
                    pbar_lock if show_progress else None
                )] = gpu_id
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Task failed on GPU {futures[future]}: {e}")
                    raise e
    finally:
        pbar.close()

    print(f"计算总耗时: {time.time() - start:.2f} 秒")

    # 结果聚合
    total_pos_hist = torch.zeros(hist_bins, dtype=torch.long)
    total_neg_hist = torch.zeros(hist_bins, dtype=torch.long)
    for p_hist, n_hist in results:
        total_pos_hist += p_hist
        total_neg_hist += n_hist

    print(f"Total Pos Pairs: {total_pos_hist.sum().item()}")
    print(f"Total Neg Pairs: {total_neg_hist.sum().item()}")

    if collect_pairs_config is not None:
        if is_dual_mode:
            pos_collected = pos_pair_collector.get_pairs() if pos_pair_collector else []
            neg_collected = neg_pair_collector.get_pairs() if neg_pair_collector else []
            if pos_collected:
                pos_tmode = collect_pairs_config['pos'].get('threshold_mode', 'below')
                pos_collected.sort(key=lambda x: x[2], reverse=(pos_tmode == 'above'))
                print(f"Collected POS Pairs: {len(pos_collected)}")
            if neg_collected:
                neg_tmode = collect_pairs_config['neg'].get('threshold_mode', 'above')
                neg_collected.sort(key=lambda x: x[2], reverse=(neg_tmode == 'above'))
                print(f"Collected NEG Pairs: {len(neg_collected)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), pos_collected, neg_collected
        else:
            collected_pairs = pair_collector.get_pairs()
            threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
            collected_pairs.sort(key=lambda x: x[2], reverse=(threshold_mode == 'above'))
            print(f"Collected Pairs: {len(collected_pairs)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), collected_pairs

    return total_pos_hist.numpy(), total_neg_hist.numpy()


# ============================================================
# V4: 动态调度的大规模相似度矩阵计算
# ============================================================

class DynamicBlockPool:
    """动态任务池 - 运行时按需分配计算块给各GPU，实现动态负载均衡

    与V3的TriangularBlockScheduler不同，不预先将块绑定到GPU，
    而是各GPU运行时从共享池中取块，快的GPU自动多干活。
    块按工作量降序排列，大块优先分配以减少尾部延迟。
    """

    def __init__(self, total_size: int, block_size: int,
                 initial_ratio: float = 0.01, subsequent_ratio: float = 0.005):
        """
        Args:
            total_size: 矩阵总大小
            block_size: 每个块的大小
            initial_ratio: 每个GPU首次取任务的比例(默认1%)
            subsequent_ratio: 后续每次取任务的比例(默认0.5%)
        """
        self.total_size = total_size
        self.block_size = block_size
        num_blocks_dim = math.ceil(total_size / block_size)

        # 生成所有上三角块，按工作量降序排列（大块优先，减少尾部延迟）
        blocks_with_workload = []
        for rb in range(num_blocks_dim):
            for cb in range(rb, num_blocks_dim):
                rs, re = rb * block_size, min((rb + 1) * block_size, total_size)
                cs, ce = cb * block_size, min((cb + 1) * block_size, total_size)
                blocks_with_workload.append(((rs, re, cs, ce), (re - rs) * (ce - cs)))

        blocks_with_workload.sort(key=lambda x: x[1], reverse=True)
        self._blocks = [b for b, _ in blocks_with_workload]
        self._total = len(self._blocks)
        self._initial_batch = max(1, int(self._total * initial_ratio))
        self._subsequent_batch = max(1, int(self._total * subsequent_ratio))
        self._cursor = 0
        self._lock = threading.Lock()
        self._initialized_gpus = set()

    @property
    def total_blocks(self) -> int:
        return self._total

    def get_batch(self, gpu_id: int) -> List[Tuple[int, int, int, int]]:
        """获取下一批计算块。首次取1%，后续取0.5%，池空返回[]"""
        with self._lock:
            if self._cursor >= self._total:
                return []
            if gpu_id not in self._initialized_gpus:
                self._initialized_gpus.add(gpu_id)
                batch_size = self._initial_batch
            else:
                batch_size = self._subsequent_batch
            start = self._cursor
            self._cursor = min(start + batch_size, self._total)
            return self._blocks[start:self._cursor]

    def remaining(self) -> int:
        return self._total - self._cursor


def _v4_gpu_worker(
    feats, ids, pool, gpu_id,
    hist_bins, hist_range,
    collect_pairs_config,
    pair_collector, pos_pair_collector, neg_pair_collector,
    memory_mode, pbar, pbar_lock
):
    """V4 GPU Worker: 从动态任务池中循环取块并计算"""
    with torch.cuda.device(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')

        # ---- 一次性GPU初始化 ----
        ids_full = ids.to(device) if torch.is_tensor(ids) else torch.tensor(ids, device=device)

        use_gpu_feats = False
        feats_source = feats
        if memory_mode == 'high_performance':
            try:
                feats_source = feats.to(device)
                use_gpu_feats = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    feats_source = feats
                else:
                    raise

        # 解析收集配置
        do_collect_single = False
        do_collect_dual = False
        sample_type = threshold_mode = threshold_val = None
        pos_cfg = neg_cfg = None

        if collect_pairs_config is not None:
            if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
                do_collect_dual = True
                pos_cfg = collect_pairs_config.get('pos', None)
                neg_cfg = collect_pairs_config.get('neg', None)
            elif pair_collector is not None:
                do_collect_single = True
                sample_type = collect_pairs_config.get('sample_type', 'neg')
                threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
                threshold_val = collect_pairs_config.get('threshold', 0.5)

        pos_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)
        neg_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)

        # ---- 动态取块循环 ----
        while True:
            batch = pool.get_batch(gpu_id)
            if not batch:
                break

            for row_start, row_end, col_start, col_end in batch:
                if use_gpu_feats:
                    block1 = feats_source[row_start:row_end]
                    block2 = feats_source[col_start:col_end]
                else:
                    block1 = feats_source[row_start:row_end].to(device, non_blocking=True)
                    block2 = feats_source[col_start:col_end].to(device, non_blocking=True)

                ids1 = ids_full[row_start:row_end]
                ids2 = ids_full[col_start:col_end]

                sim_block = torch.matmul(block1, block2.T)
                label_eq = (ids1[:, None] == ids2[None, :])

                if row_start == col_start:
                    mask = torch.triu(torch.ones_like(sim_block, dtype=torch.bool), diagonal=1)
                else:
                    mask = torch.ones_like(sim_block, dtype=torch.bool)

                flat_sim = sim_block[mask]
                flat_labels = label_eq[mask]

                # -- 正样本 --
                pos_sim = flat_sim[flat_labels]
                if pos_sim.numel() > 0:
                    pos_hist += torch.histc(pos_sim, bins=hist_bins,
                                            min=hist_range[0], max=hist_range[1]).long()
                    if do_collect_single and not pair_collector.is_full() and sample_type == 'pos':
                        tmask = pos_sim > threshold_val if threshold_mode == 'above' else pos_sim < threshold_val
                        if tmask.any():
                            _collect_pairs(mask, flat_labels, tmask, pos_sim,
                                           row_start, col_start, pair_collector, True)
                    if do_collect_dual and pos_cfg and pos_pair_collector and not pos_pair_collector.is_full():
                        pt = pos_cfg.get('threshold_mode', 'below')
                        pv = pos_cfg.get('threshold', 0.25)
                        tmask = pos_sim > pv if pt == 'above' else pos_sim < pv
                        if tmask.any():
                            _collect_pairs(mask, flat_labels, tmask, pos_sim,
                                           row_start, col_start, pos_pair_collector, True)

                # -- 负样本 --
                neg_sim = flat_sim[~flat_labels]
                if neg_sim.numel() > 0:
                    neg_hist += torch.histc(neg_sim, bins=hist_bins,
                                            min=hist_range[0], max=hist_range[1]).long()
                    if do_collect_single and not pair_collector.is_full() and sample_type == 'neg':
                        tmask = neg_sim > threshold_val if threshold_mode == 'above' else neg_sim < threshold_val
                        if tmask.any():
                            _collect_pairs(mask, flat_labels, tmask, neg_sim,
                                           row_start, col_start, pair_collector, False)
                    if do_collect_dual and neg_cfg and neg_pair_collector and not neg_pair_collector.is_full():
                        nt = neg_cfg.get('threshold_mode', 'above')
                        nv = neg_cfg.get('threshold', 0.5)
                        tmask = neg_sim > nv if nt == 'above' else neg_sim < nv
                        if tmask.any():
                            _collect_pairs(mask, flat_labels, tmask, neg_sim,
                                           row_start, col_start, neg_pair_collector, False)

                del block1, block2, sim_block, mask, flat_sim, flat_labels
                if pbar is not None:
                    with pbar_lock:
                        pbar.update(1)

    return pos_hist.cpu(), neg_hist.cpu()


def get_sim_matrix_large_scale_v4(
    query_feats_list, query_ids=None, num_gpus=7, block_size=2048*5,
    hist_bins=20_000_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None, memory_mode='low_memory', show_progress=True,
    initial_ratio=0.01, subsequent_ratio=0.005
):
    """
    基于动态调度的大规模相似度矩阵计算 v4

    与v3的区别: 不预先将块分配给GPU，而是各GPU运行时从共享任务池动态取块，
    首次取1%，后续每次取0.5%，实现运行时动态负载均衡。
    块按工作量降序排列，大块优先分配以减少尾部延迟。

    Args:
        initial_ratio: 每个GPU首次取任务的比例(默认0.01即1%)
        subsequent_ratio: 后续每次取任务的比例(默认0.005即0.5%)
        其余参数与v3一致

    Returns:
        与v3一致
    """
    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))

    N = len(query_ids)

    # 创建动态任务池
    pool = DynamicBlockPool(total_size=N, block_size=block_size,
                            initial_ratio=initial_ratio, subsequent_ratio=subsequent_ratio)
    total_blocks = pool.total_blocks
    print(f"动态任务池: 共 {total_blocks} 块, 首批 {pool._initial_batch} 块/GPU, "
          f"后续 {pool._subsequent_batch} 块/次")

    # 准备数据
    print(f"正在准备数据 (Mode: {memory_mode})...")
    prep_start = time.time()
    if isinstance(query_feats_list, list):
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif isinstance(query_feats_list, np.ndarray):
        query_feats_tensor = torch.from_numpy(query_feats_list)
    else:
        query_feats_tensor = query_feats_list

    if memory_mode == 'low_memory':
        if query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.cpu()
        query_feats_tensor = query_feats_tensor.float().contiguous().pin_memory()
    else:
        if not query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.float().pin_memory()
    print(f"数据准备耗时: {time.time() - prep_start:.2f} 秒")

    # 收集器准备
    pair_collector = None
    pos_pair_collector = None
    neg_pair_collector = None
    is_dual_mode = False

    if collect_pairs_config is not None:
        if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
            is_dual_mode = True
            if 'pos' in collect_pairs_config:
                pos_pair_collector = PairCollector(max_pairs=collect_pairs_config['pos'].get('max_pairs', -1))
            if 'neg' in collect_pairs_config:
                neg_pair_collector = PairCollector(max_pairs=collect_pairs_config['neg'].get('max_pairs', -1))
        else:
            pair_collector = PairCollector(max_pairs=collect_pairs_config.get('max_pairs', -1))

    start = time.time()
    results = []
    pbar = tqdm(total=total_blocks, desc="Matrix Cal v4 (dynamic)", disable=not show_progress)
    pbar_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for gpu_id in range(num_gpus):
                futures[executor.submit(
                    _v4_gpu_worker,
                    query_feats_tensor, query_ids, pool, gpu_id,
                    hist_bins, hist_range, collect_pairs_config,
                    pair_collector, pos_pair_collector, neg_pair_collector,
                    memory_mode,
                    pbar if show_progress else None,
                    pbar_lock if show_progress else None
                )] = gpu_id
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"GPU {futures[future]} failed: {e}")
                    raise
    finally:
        pbar.close()

    print(f"计算总耗时: {time.time() - start:.2f} 秒")

    # 释放各 GPU 上的 CUDA 缓存（ids_full / hist 等 worker 残留）
    # 注意: 不调用 gc.collect()，大规模计算后其遍历对象图极慢（可达数百秒）
    for gid in range(num_gpus):
        with torch.cuda.device(gid):
            torch.cuda.empty_cache()

    # 结果聚合
    total_pos_hist = torch.zeros(hist_bins, dtype=torch.long)
    total_neg_hist = torch.zeros(hist_bins, dtype=torch.long)
    for p_hist, n_hist in results:
        total_pos_hist += p_hist
        total_neg_hist += n_hist

    print(f"Total Pos Pairs: {total_pos_hist.sum().item()}")
    print(f"Total Neg Pairs: {total_neg_hist.sum().item()}")

    if collect_pairs_config is not None:
        if is_dual_mode:
            pos_collected = pos_pair_collector.get_pairs() if pos_pair_collector else []
            neg_collected = neg_pair_collector.get_pairs() if neg_pair_collector else []
            if pos_collected:
                pos_tmode = collect_pairs_config['pos'].get('threshold_mode', 'below')
                pos_collected.sort(key=lambda x: x[2], reverse=(pos_tmode == 'above'))
                print(f"Collected POS Pairs: {len(pos_collected)}")
            if neg_collected:
                neg_tmode = collect_pairs_config['neg'].get('threshold_mode', 'above')
                neg_collected.sort(key=lambda x: x[2], reverse=(neg_tmode == 'above'))
                print(f"Collected NEG Pairs: {len(neg_collected)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), pos_collected, neg_collected
        else:
            collected_pairs = pair_collector.get_pairs()
            threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
            collected_pairs.sort(key=lambda x: x[2], reverse=(threshold_mode == 'above'))
            print(f"Collected Pairs: {len(collected_pairs)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), collected_pairs

    return total_pos_hist.numpy(), total_neg_hist.numpy()

# ============================================================
# V5: where(NaN) + masked_fill_ 优化版本
# 消除 boolean indexing 的动态分配, 支持 FP16/FP32 精度控制
# ============================================================

def _v5_collect_pairs(sim, mask_2d, row_start, col_start, pair_collector):
    """V5辅助函数：从2D sim矩阵中直接收集符合条件的样本对"""
    if pair_collector.is_full():
        return
    rows, cols = torch.where(mask_2d)
    if rows.numel() == 0:
        return
    scores = sim[rows, cols]
    global_i = (rows + row_start).cpu().numpy()
    global_j = (cols + col_start).cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    current_pairs = list(zip(global_i, global_j, scores_cpu))
    pair_collector.add_pairs(current_pairs)

# ============================================================
# V5 GPU Worker
# ============================================================

def _v5_gpu_worker(
    feats, ids, pool, gpu_id,
    hist_bins, hist_range,
    collect_pairs_config,
    pair_collector, pos_pair_collector, neg_pair_collector,
    memory_mode, precision, pbar, pbar_lock
):
    """V5 GPU Worker: masked_fill_(NaN)优化, 消除boolean indexing动态分配"""
    with torch.cuda.device(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        NAN = float('nan')

        # ---- 精度设置 ----
        use_fp16 = (precision == 'fp16')
        if use_fp16:
            torch.backends.cuda.matmul.allow_tf32 = True

        # ---- 一次性GPU初始化 ----
        ids_full = ids.to(device) if torch.is_tensor(ids) else torch.tensor(ids, device=device)

        use_gpu_feats = False
        feats_source = feats
        if memory_mode == 'high_performance':
            try:
                feats_source = feats.to(device)
                use_gpu_feats = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    feats_source = feats
                else:
                    raise

        # 解析收集配置
        do_collect_single = False
        do_collect_dual = False
        sample_type = threshold_mode = threshold_val = None
        pos_cfg = neg_cfg = None

        if collect_pairs_config is not None:
            if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
                do_collect_dual = True
                pos_cfg = collect_pairs_config.get('pos', None)
                neg_cfg = collect_pairs_config.get('neg', None)
            elif pair_collector is not None:
                do_collect_single = True
                sample_type = collect_pairs_config.get('sample_type', 'neg')
                threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
                threshold_val = collect_pairs_config.get('threshold', 0.5)

        need_collect = (do_collect_single or do_collect_dual)

        pos_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)
        neg_hist = torch.zeros(hist_bins, device=device, dtype=torch.long)

        # ---- 动态取块循环 ----
        while True:
            batch = pool.get_batch(gpu_id)
            if not batch:
                break

            for row_start, row_end, col_start, col_end in batch:
                if use_gpu_feats:
                    block1 = feats_source[row_start:row_end]
                    block2 = feats_source[col_start:col_end]
                else:
                    block1 = feats_source[row_start:row_end].to(device, non_blocking=True)
                    block2 = feats_source[col_start:col_end].to(device, non_blocking=True)

                # matmul
                if use_fp16:
                    sim = torch.matmul(block1.half(), block2.half().T).float()
                else:
                    sim = torch.matmul(block1, block2.T)
                del block1, block2

                ids1 = ids_full[row_start:row_end]
                ids2 = ids_full[col_start:col_end]
                label_eq = (ids1[:, None] == ids2[None, :])

                # 对角块: in-place遮蔽下三角+对角线
                is_diag = (row_start == col_start)
                if is_diag:
                    triu_mask = torch.triu(
                        torch.ones(row_end - row_start, col_end - col_start,
                                   device=device, dtype=torch.bool),
                        diagonal=1)
                    sim.masked_fill_(~triu_mask, NAN)

                # -- pair collection 需要原始sim值, 在histogram修改前保存引用 --
                if need_collect:
                    # 保存一份sim用于pair collection(histogram会in-place修改sim)
                    sim_for_pairs = sim.clone()

                # -- histogram: full → pos → neg = full - pos --
                full_hist = torch.histc(sim, bins=hist_bins,
                                        min=hist_range[0], max=hist_range[1]).long()
                sim.masked_fill_(~label_eq, NAN)
                pos_block = torch.histc(sim, bins=hist_bins,
                                        min=hist_range[0], max=hist_range[1]).long()
                pos_hist += pos_block
                neg_hist += full_hist - pos_block
                del full_hist, pos_block, sim

                # -- pair collection --
                if need_collect:
                    if is_diag:
                        valid_mask = triu_mask
                    else:
                        valid_mask = torch.ones(row_end - row_start, col_end - col_start,
                                               device=device, dtype=torch.bool)

                    if do_collect_single and not pair_collector.is_full():
                        if sample_type == 'pos':
                            pos_valid = label_eq & valid_mask
                            if threshold_mode == 'above':
                                target = pos_valid & (sim_for_pairs > threshold_val)
                            else:
                                target = pos_valid & (sim_for_pairs < threshold_val)
                            _v5_collect_pairs(sim_for_pairs, target, row_start, col_start, pair_collector)
                            del pos_valid, target
                        elif sample_type == 'neg':
                            neg_valid = (~label_eq) & valid_mask
                            if threshold_mode == 'above':
                                target = neg_valid & (sim_for_pairs > threshold_val)
                            else:
                                target = neg_valid & (sim_for_pairs < threshold_val)
                            _v5_collect_pairs(sim_for_pairs, target, row_start, col_start, pair_collector)
                            del neg_valid, target

                    if do_collect_dual:
                        if pos_cfg and pos_pair_collector and not pos_pair_collector.is_full():
                            pt = pos_cfg.get('threshold_mode', 'below')
                            pv = pos_cfg.get('threshold', 0.25)
                            pos_valid = label_eq & valid_mask
                            if pt == 'above':
                                target = pos_valid & (sim_for_pairs > pv)
                            else:
                                target = pos_valid & (sim_for_pairs < pv)
                            _v5_collect_pairs(sim_for_pairs, target, row_start, col_start, pos_pair_collector)
                            del pos_valid, target

                        if neg_cfg and neg_pair_collector and not neg_pair_collector.is_full():
                            nt = neg_cfg.get('threshold_mode', 'above')
                            nv = neg_cfg.get('threshold', 0.5)
                            neg_valid = (~label_eq) & valid_mask
                            if nt == 'above':
                                target = neg_valid & (sim_for_pairs > nv)
                            else:
                                target = neg_valid & (sim_for_pairs < nv)
                            _v5_collect_pairs(sim_for_pairs, target, row_start, col_start, neg_pair_collector)
                            del neg_valid, target

                    del sim_for_pairs, valid_mask

                if is_diag:
                    del triu_mask
                del label_eq

                if pbar is not None:
                    with pbar_lock:
                        pbar.update(1)

    return pos_hist.cpu(), neg_hist.cpu()


def get_sim_matrix_large_scale_v5(
    query_feats_list, query_ids=None, num_gpus=7, block_size=2048*5,
    hist_bins=20_000_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None, memory_mode='low_memory', show_progress=True,
    initial_ratio=0.01, subsequent_ratio=0.005,
    precision='fp32'
):
    """
    基于动态调度的大规模相似度矩阵计算 v5 (where/masked_fill_ 优化版)

    相比v4的改进:
    - 使用 masked_fill_(NaN) + histc 替代 boolean indexing, 消除动态分配
    - neg_hist = full_hist - pos_hist, 避免负样本的额外遍历
    - 支持 FP16 matmul (precision='fp16'), 利用 Tensor Core 加速
    - 峰值显存更低 (无需分配 flat_sim, flat_labels 等中间张量)

    Args:
        query_feats_list: 特征矩阵 (numpy array, list, 或 torch.Tensor)
        query_ids: 身份标签
        num_gpus: GPU数量
        block_size: 每个块的大小
        hist_bins: 直方图bin数量 (默认20M, FP16建议2000)
        hist_range: 直方图范围
        collect_pairs_config: 样本对收集配置 (与v4兼容)
        memory_mode: 'low_memory' 或 'high_performance'
        show_progress: 是否显示进度条
        initial_ratio: 每个GPU首次取任务的比例
        subsequent_ratio: 后续每次取任务的比例
        precision: 'fp32' 或 'fp16'

    Returns:
        与v4一致
    """
    # 精度建议
    if precision == 'fp16' and hist_bins != 2000:
        print(f"[V5建议] FP16精度下有效分辨率约0.001, 当前hist_bins={hist_bins:,}, "
              f"建议设为2000以获得最佳性能 (当前仍按{hist_bins:,}计算)")

    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))

    N = len(query_ids)

    # 创建动态任务池
    pool = DynamicBlockPool(total_size=N, block_size=block_size,
                            initial_ratio=initial_ratio, subsequent_ratio=subsequent_ratio)
    total_blocks = pool.total_blocks
    print(f"动态任务池 v5: 共 {total_blocks} 块, 首批 {pool._initial_batch} 块/GPU, "
          f"后续 {pool._subsequent_batch} 块/次 | precision={precision}")

    # 准备数据
    print(f"正在准备数据 (Mode: {memory_mode})...")
    prep_start = time.time()
    if isinstance(query_feats_list, list):
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif isinstance(query_feats_list, np.ndarray):
        query_feats_tensor = torch.from_numpy(query_feats_list)
    else:
        query_feats_tensor = query_feats_list

    if memory_mode == 'low_memory':
        if query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.cpu()
        query_feats_tensor = query_feats_tensor.float().contiguous().pin_memory()
    else:
        if not query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.float().pin_memory()
    print(f"数据准备耗时: {time.time() - prep_start:.2f} 秒")

    # 收集器准备
    pair_collector = None
    pos_pair_collector = None
    neg_pair_collector = None
    is_dual_mode = False

    if collect_pairs_config is not None:
        if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
            is_dual_mode = True
            if 'pos' in collect_pairs_config:
                pos_pair_collector = PairCollector(max_pairs=collect_pairs_config['pos'].get('max_pairs', -1))
            if 'neg' in collect_pairs_config:
                neg_pair_collector = PairCollector(max_pairs=collect_pairs_config['neg'].get('max_pairs', -1))
        else:
            pair_collector = PairCollector(max_pairs=collect_pairs_config.get('max_pairs', -1))

    start = time.time()
    results = []
    pbar = tqdm(total=total_blocks, desc="Matrix Cal v5 (dynamic)", disable=not show_progress)
    pbar_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for gpu_id in range(num_gpus):
                futures[executor.submit(
                    _v5_gpu_worker,
                    query_feats_tensor, query_ids, pool, gpu_id,
                    hist_bins, hist_range, collect_pairs_config,
                    pair_collector, pos_pair_collector, neg_pair_collector,
                    memory_mode, precision,
                    pbar if show_progress else None,
                    pbar_lock if show_progress else None
                )] = gpu_id
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"GPU {futures[future]} failed: {e}")
                    raise
    finally:
        pbar.close()

    print(f"计算总耗时: {time.time() - start:.2f} 秒")

    # 释放各 GPU 上的 CUDA 缓存
    for gid in range(num_gpus):
        with torch.cuda.device(gid):
            torch.cuda.empty_cache()

    # 结果聚合
    total_pos_hist = torch.zeros(hist_bins, dtype=torch.long)
    total_neg_hist = torch.zeros(hist_bins, dtype=torch.long)
    for p_hist, n_hist in results:
        total_pos_hist += p_hist
        total_neg_hist += n_hist

    print(f"Total Pos Pairs: {total_pos_hist.sum().item()}")
    print(f"Total Neg Pairs: {total_neg_hist.sum().item()}")

    if collect_pairs_config is not None:
        if is_dual_mode:
            pos_collected = pos_pair_collector.get_pairs() if pos_pair_collector else []
            neg_collected = neg_pair_collector.get_pairs() if neg_pair_collector else []
            if pos_collected:
                pos_tmode = collect_pairs_config['pos'].get('threshold_mode', 'below')
                pos_collected.sort(key=lambda x: x[2], reverse=(pos_tmode == 'above'))
                print(f"Collected POS Pairs: {len(pos_collected)}")
            if neg_collected:
                neg_tmode = collect_pairs_config['neg'].get('threshold_mode', 'above')
                neg_collected.sort(key=lambda x: x[2], reverse=(neg_tmode == 'above'))
                print(f"Collected NEG Pairs: {len(neg_collected)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), pos_collected, neg_collected
        else:
            collected_pairs = pair_collector.get_pairs()
            threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
            collected_pairs.sort(key=lambda x: x[2], reverse=(threshold_mode == 'above'))
            print(f"Collected Pairs: {len(collected_pairs)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), collected_pairs

    return total_pos_hist.numpy(), total_neg_hist.numpy()


# ============================================================
# V6: 吸收 glm 线程版优点 + 保留 v5 通用能力
#   - block_size 默认 16384（更大 GEMM）
#   - 单 tile 动态队列（行块序 + 行内宽优先），尾延迟更小
#   - low_memory 下 row_cache：同一行块只 H2D 一次
#   - sim 先 clamp 到 [LO,HI]（不丢越界对），单趟直方图 neg=full-pos
#   - precision: 'fp32'(关TF32) / 'tf32' / 'fp16'
#   - 保留 low_memory/high_performance、collect_pairs_config、show_progress
# ============================================================

import queue


def _v6_gpu_worker(
    feats, ids, tile_q, gpu_id, block_size, n,
    hist_bins, hist_range,
    collect_pairs_config, pair_collector, pos_pair_collector, neg_pair_collector,
    memory_mode, precision, row_cache, hist_method, pbar, pbar_lock, out, slot
):
    """V6 GPU Worker：单 tile 动态队列 + 行块缓存 + 单趟直方图(neg=full-pos)"""
    with torch.cuda.device(gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        LO, HI = float(hist_range[0]), float(hist_range[1])
        FILL = LO - 2.0                     # 越界填充值，histc 会丢弃
        scale = hist_bins / 2.0             # 量化 scale（bincount 用）
        use_bincount = (hist_method == 'bincount')

        # ---- 精度设置：fp32(关TF32) / tf32 / fp16 ----
        use_fp16 = (precision == 'fp16')
        use_tf32 = (precision == 'tf32')
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

        ids_full = torch.as_tensor(ids, device=device)   # (N,) 常驻显存

        use_gpu_feats = (memory_mode == 'high_performance')
        if use_gpu_feats:
            feats_source = feats.to(device)              # 特征常驻显存
        else:
            feats_source = feats                         # pinned CPU，按需 H2D

        # ---- 解析样本对收集配置（与 v5 同语义）----
        do_collect_single = False
        do_collect_dual = False
        sample_type = threshold_mode = threshold_val = None
        pos_cfg = neg_cfg = None
        if collect_pairs_config is not None:
            if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
                do_collect_dual = True
                pos_cfg = collect_pairs_config.get('pos', None)
                neg_cfg = collect_pairs_config.get('neg', None)
            elif pair_collector is not None:
                do_collect_single = True
                sample_type = collect_pairs_config.get('sample_type', 'neg')
                threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
                threshold_val = collect_pairs_config.get('threshold', 0.5)
        need_collect = (do_collect_single or do_collect_dual)

        pos_hist = torch.zeros(hist_bins, device=device, dtype=torch.int64)
        neg_hist = torch.zeros(hist_bins, device=device, dtype=torch.int64)

        # 行块缓存（low_memory 下避免同一行块重复 H2D）
        cached_bi = -1
        cached_row = None

        while True:
            tile = tile_q.get()
            if tile is None:
                break
            bi, bj, has_pos = tile
            r0, r1 = bi * block_size, min((bi + 1) * block_size, n)
            c0, c1 = bj * block_size, min((bj + 1) * block_size, n)
            is_diag = (bi == bj)

            # 行块：缓存命中复用，否则 H2D（或显存切片）
            if use_gpu_feats:
                block1 = feats_source[r0:r1]
            elif row_cache and bi == cached_bi:
                block1 = cached_row
            else:
                block1 = feats_source[r0:r1].to(device, non_blocking=True)
                if row_cache:
                    cached_bi, cached_row = bi, block1

            # 列块：对角块复用行块，否则 H2D（或显存切片）
            if is_diag:
                block2 = block1
            elif use_gpu_feats:
                block2 = feats_source[c0:c1]
            else:
                block2 = feats_source[c0:c1].to(device, non_blocking=True)

            # matmul
            if use_fp16:
                sim = torch.matmul(block1.half(), block2.half().T).float()
            else:
                sim = torch.matmul(block1, block2.T)     # fp32 / tf32
            sim.clamp_(LO, HI)

            # 身份等值掩码：仅本 tile 含正样本对 或 需要收集样本对 时才计算
            if has_pos or need_collect:
                label_eq = (ids_full[r0:r1, None] == ids_full[None, c0:c1])
            else:
                label_eq = None

            if is_diag:
                triu = torch.triu(
                    torch.ones(r1 - r0, c1 - c0, device=device, dtype=torch.bool),
                    diagonal=1)

            # 单趟直方图前半：full（对角块先把下三角填范围外）
            if use_bincount:
                q = ((sim + 1.0) * scale).floor_().clamp_(0, hist_bins - 1).to(torch.int32)
                if is_diag:
                    full_hist = torch.bincount(q[triu], minlength=hist_bins).to(torch.int64)
                else:
                    full_hist = torch.bincount(q.reshape(-1), minlength=hist_bins).to(torch.int64)
            else:
                if is_diag:
                    sim.masked_fill_(~triu, FILL)
                full_hist = torch.histc(sim, bins=hist_bins, min=LO, max=HI).to(torch.int64)

            # 样本对收集：在 pos 的 in-place masked_fill 之前做，直接用 sim（省掉整份 clone）
            if need_collect:
                if is_diag:
                    valid_mask = triu
                else:
                    valid_mask = torch.ones(r1 - r0, c1 - c0, device=device, dtype=torch.bool)

                if do_collect_single and not pair_collector.is_full():
                    base = label_eq if sample_type == 'pos' else ~label_eq
                    target = (base & valid_mask & (sim > threshold_val)) \
                        if threshold_mode == 'above' else (base & valid_mask & (sim < threshold_val))
                    _v5_collect_pairs(sim, target, r0, c0, pair_collector)
                    del target

                if do_collect_dual:
                    if pos_cfg and pos_pair_collector and not pos_pair_collector.is_full():
                        pt = pos_cfg.get('threshold_mode', 'below')
                        pv = pos_cfg.get('threshold', 0.25)
                        cond = sim > pv if pt == 'above' else sim < pv
                        _v5_collect_pairs(sim, label_eq & valid_mask & cond,
                                          r0, c0, pos_pair_collector)
                    if neg_cfg and neg_pair_collector and not neg_pair_collector.is_full():
                        nt = neg_cfg.get('threshold_mode', 'above')
                        nv = neg_cfg.get('threshold', 0.5)
                        cond = sim > nv if nt == 'above' else sim < nv
                        _v5_collect_pairs(sim, (~label_eq) & valid_mask & cond,
                                          r0, c0, neg_pair_collector)

                del valid_mask

            # 单趟直方图后半：pos + neg = full - pos
            if use_bincount:
                pos_block = None
                if has_pos:
                    sel = (triu & label_eq) if is_diag else label_eq
                    pos_block = torch.bincount(q[sel], minlength=hist_bins).to(torch.int64)
                    del sel
                del q
            else:
                pos_block = None
                if has_pos:
                    sim.masked_fill_(~label_eq, FILL)
                    pos_block = torch.histc(sim, bins=hist_bins, min=LO, max=HI).to(torch.int64)

            if pos_block is not None:
                pos_hist += pos_block
                neg_hist += full_hist - pos_block
                del pos_block
            else:
                neg_hist += full_hist
            del full_hist, sim
            if label_eq is not None:
                del label_eq
            if is_diag:
                del triu

            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

        out[slot] = (pos_hist.cpu(), neg_hist.cpu())


def _v6_pos_tile_flags(ids, block_size, n):
    """静态判定哪些上三角块 (bi,bj) 含 >=1 条同身份样本对（保守：绝不漏标）。

    对角块：块内某身份 >=2 个成员；非对角：某身份成员跨越块 [bf,bl] 时标记其间
    全部 (b1<b2)。热路径据此跳过无正样本 tile 的身份等值比对（正样本对稀疏时
    绝大多数 tile 可免比对）。"""
    nb = math.ceil(n / block_size)
    flags = set()
    if n < 2:
        return flags
    ids = np.asarray(ids)
    for bi in range(nb):
        b0, b1 = bi * block_size, min((bi + 1) * block_size, n)
        _, cnt = np.unique(ids[b0:b1], return_counts=True)
        if cnt.size and int(cnt.max()) >= 2:
            flags.add((bi, bi))
    order = np.argsort(ids, kind='stable')
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1, n]
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < 2:
            continue
        bf = int(order[s]) // block_size
        bl = int(order[e - 1]) // block_size
        if bl > bf:
            for b1 in range(bf, bl):
                for b2 in range(b1 + 1, bl + 1):
                    flags.add((b1, b2))
    return flags


def get_sim_matrix_large_scale_v6(
    query_feats_list, query_ids=None, num_gpus=7, block_size=16384,
    hist_bins=200_000, hist_range=(-1.0, 1.0),
    collect_pairs_config=None, memory_mode='low_memory', show_progress=True,
    precision='tf32', row_cache=True, pin_inplace=True,
    hist_method='histc', skip_pos_check=True,
):
    """
    cluster_utils v6 —— 吸收 glm 线程版 + deepseek(-pro) + codex-sol-2 优点，保留 v5 通用能力。

    相比 v5 的改进:
      - block_size 默认 16384（更大 GEMM，kernel 更少、GPU 利用率更高）
      - 调度从"成批取块(1%/0.5%)"改为"单 tile 动态队列(行块序+行内宽优先)"，尾延迟更小
      - low_memory 下支持 row_cache：同一行块只 H2D 一次（缓存复用），减少 PCIe 传输
      - low_memory 下 pin_inplace=True 用 cudaHostRegister 原地锁定 numpy（零拷贝），
        省掉 pin_memory() 的整份 pinned 副本（约一份 feats 大小）
      - sim 先 clamp 到 [LO,HI]（不丢越界对），单趟直方图 neg = full - pos
      - 精度三档 precision: 'fp32'(关TF32) / 'tf32' / 'fp16'
      - hist_method='histc'（默认，torch 2.12 下实测比量化+bincount 快 ~30%）或
        'bincount'（deepseek 式量化+GPU bincount，int64 精确，本环境实测更慢，保留备选）
      - skip_pos_check=True（deepseek-pro 式）静态预标含正样本 tile，热路径对无正
        样本 tile 免身份等值比对（本数据 96.8% tile 无正样本，2M 实测 +27%）
    相比 v5 保留:
      - low_memory / high_performance 两种显存模式
      - collect_pairs_config 样本对收集（单/双模式）
      - show_progress 进度条
    """
    if precision not in ('fp32', 'tf32', 'fp16'):
        raise ValueError(f"precision 必须是 'fp32'/'tf32'/'fp16'，收到 {precision!r}")

    if precision == 'fp16' and hist_bins > 2000:
        print(f"[V6建议] fp16 有效分辨率约 0.001，当前 hist_bins={hist_bins:,} 偏细，"
              f"如需最佳性能可设 2000（当前仍按 {hist_bins:,} 计算）")

    if query_ids is None:
        query_ids = np.arange(len(query_feats_list))
    N = len(query_ids)

    # ---- 准备数据 ----
    if isinstance(query_feats_list, list):
        query_feats_tensor = torch.from_numpy(np.array(query_feats_list))
    elif isinstance(query_feats_list, np.ndarray):
        query_feats_tensor = torch.from_numpy(query_feats_list)
    else:
        query_feats_tensor = query_feats_list

    registered_ptr = None
    if memory_mode == 'low_memory':
        if query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.cpu()
        if pin_inplace and isinstance(query_feats_list, np.ndarray):
            # 原地锁定 numpy 缓冲区（零拷贝），避免 pin_memory() 的整份 pinned 副本
            arr = np.ascontiguousarray(query_feats_list, dtype=np.float32)
            ptr = arr.ctypes.data
            torch.cuda.cudart().cudaHostRegister(ptr, arr.nbytes, 0)
            registered_ptr = ptr
            query_feats_tensor = torch.from_numpy(arr).float().contiguous()
        else:
            query_feats_tensor = query_feats_tensor.float().contiguous().pin_memory()
    else:
        if not query_feats_tensor.is_cuda:
            query_feats_tensor = query_feats_tensor.float().pin_memory()

    # ---- 静态预标含正样本 tile（deepseek-pro 式，热路径免身份比对）----
    if skip_pos_check:
        flags = _v6_pos_tile_flags(np.asarray(query_ids), block_size, N)
    else:
        flags = None

    # ---- 建 tile 队列：行块序 + 行内宽优先（利于行块缓存 + 大块优先）----
    nb = math.ceil(N / block_size)
    tiles = []
    for bi in range(nb):
        for bj in range(bi, nb):
            has_pos = (flags is None) or ((bi, bj) in flags)
            tiles.append((bi, bj, has_pos))
    tiles.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    tile_q = queue.Queue()
    for t in tiles:
        tile_q.put(t)
    for _ in range(num_gpus):
        tile_q.put(None)
    n_pos_tiles = sum(1 for t in tiles if t[2]) if flags is not None else len(tiles)
    print(f"V6 tile 队列: 共 {len(tiles)} 块(含正样本 {n_pos_tiles}), block={block_size}, "
          f"precision={precision}, hist={hist_method}, row_cache={row_cache}, mode={memory_mode}")

    # ---- 收集器准备（与 v5 同语义）----
    pair_collector = pos_pair_collector = neg_pair_collector = None
    is_dual_mode = False
    if collect_pairs_config is not None:
        if 'pos' in collect_pairs_config or 'neg' in collect_pairs_config:
            is_dual_mode = True
            if 'pos' in collect_pairs_config:
                pos_pair_collector = PairCollector(
                    max_pairs=collect_pairs_config['pos'].get('max_pairs', -1))
            if 'neg' in collect_pairs_config:
                neg_pair_collector = PairCollector(
                    max_pairs=collect_pairs_config['neg'].get('max_pairs', -1))
        else:
            pair_collector = PairCollector(
                max_pairs=collect_pairs_config.get('max_pairs', -1))

    start = time.time()
    pbar = tqdm(total=len(tiles), desc="Matrix Cal v6", disable=not show_progress) \
        if show_progress else None
    pbar_lock = threading.Lock()

    out = [None] * num_gpus
    threads = []
    try:
        for gpu_id in range(num_gpus):
            th = threading.Thread(
                target=_v6_gpu_worker,
                args=(query_feats_tensor, query_ids, tile_q, gpu_id, block_size, N,
                      hist_bins, hist_range, collect_pairs_config,
                      pair_collector, pos_pair_collector, neg_pair_collector,
                      memory_mode, precision, row_cache, hist_method,
                      pbar, pbar_lock, out, gpu_id),
                name=f'v6-gpu-{gpu_id}',
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    finally:
        if registered_ptr is not None:
            torch.cuda.cudart().cudaHostUnregister(registered_ptr)
    if pbar is not None:
        pbar.close()

    results = []
    for r in out:
        if r is None or isinstance(r, Exception):
            raise RuntimeError(f'V6 worker 失败: {r!r}')
        results.append(r)

    print(f"计算总耗时: {time.time() - start:.2f} 秒")

    for gid in range(num_gpus):
        with torch.cuda.device(gid):
            torch.cuda.empty_cache()

    total_pos_hist = torch.zeros(hist_bins, dtype=torch.int64)
    total_neg_hist = torch.zeros(hist_bins, dtype=torch.int64)
    for p_hist, n_hist in results:
        total_pos_hist += p_hist
        total_neg_hist += n_hist

    print(f"Total Pos Pairs: {total_pos_hist.sum().item()}")
    print(f"Total Neg Pairs: {total_neg_hist.sum().item()}")

    if collect_pairs_config is not None:
        if is_dual_mode:
            pos_collected = pos_pair_collector.get_pairs() if pos_pair_collector else []
            neg_collected = neg_pair_collector.get_pairs() if neg_pair_collector else []
            if pos_collected:
                pos_tmode = collect_pairs_config['pos'].get('threshold_mode', 'below')
                pos_collected.sort(key=lambda x: x[2], reverse=(pos_tmode == 'above'))
                print(f"Collected POS Pairs: {len(pos_collected)}")
            if neg_collected:
                neg_tmode = collect_pairs_config['neg'].get('threshold_mode', 'above')
                neg_collected.sort(key=lambda x: x[2], reverse=(neg_tmode == 'above'))
                print(f"Collected NEG Pairs: {len(neg_collected)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), pos_collected, neg_collected
        else:
            collected_pairs = pair_collector.get_pairs()
            threshold_mode = collect_pairs_config.get('threshold_mode', 'above')
            collected_pairs.sort(key=lambda x: x[2], reverse=(threshold_mode == 'above'))
            print(f"Collected Pairs: {len(collected_pairs)}")
            return total_pos_hist.numpy(), total_neg_hist.numpy(), collected_pairs

    return total_pos_hist.numpy(), total_neg_hist.numpy()
