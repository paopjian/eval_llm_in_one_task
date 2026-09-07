# 大规模人脸特征相似度评估

主程序为 [`face_similarity_eval.py`](face_similarity_eval.py)，读取任务约定的
`(query_feats_list, query_feats_list_flip, query_ids, file_paths)` pkl 文件，按块计算
严格上三角样本对，并流式统计正负相似度分布。程序不会构造完整的 `N×N` 矩阵。

实现按任务要求逐层演进：先校验数据和正负对数量，再用分块矩阵乘法替代完整矩阵；
随后以整数直方图替代亿级分数数组，最后在 CUDA 可用时将行块分配到多张 GPU，并在
设备端完成掩码和直方图统计。

## 运行

在 `cvlface` 环境中执行：

```bash
python face_similarity_eval.py \
  --input s4_0618_enhance.pkl \
  --output-dir results \
  --devices auto
```

有多张 GPU 时可显式指定设备，例如 `--devices 0,1,2,3,4,5,6`。没有 CUDA 时会自动
退化为 CPU。`--precision auto` 会在 CUDA 上使用 FP16 矩阵乘法（相似度统计仍以
整数直方图累计）；如需全精度可指定 `--precision fp32`。`--block-size` 控制矩阵块
大小；显存充足时可尝试 4096–10240，显存紧张时可使用 1024 或 2048。

常用选项：

```bash
# 自定义评估点
python face_similarity_eval.py --fpirs 1e-6,1e-5,1e-4,1e-3,1e-2

# 提取高相似度负样本，最多保存 5000 条
python face_similarity_eval.py \
  --extract-threshold 0.7 --extract-type negative --extract-limit 5000
```

## 输出

`--output-dir` 中会生成：

- `evaluation_summary.json`：样本/正负对统计及 TPIR@FPIR；
- `similarity_histograms.npz`：正负相似度直方图和区间边界；
- `similarity_distribution.png`：正负相似度分布图；
- `tpir_fpir_curve.png`：TPIR@FPIR 曲线；
- 可选的 `pairs_*.parquet`：阈值样本对（使用 Polars 写出）。

总样本对超过约 500 万时，默认只保留直方图，不保存每一条原始相似度，以控制
内存；小数据集会自动保留原始分数并使用精确经验阈值。需要强制保存时可传入
`--collect-scores`，但大数据集不建议这样做。

核心正确性和输出测试可直接运行：

```bash
python test_face_similarity_eval.py
```
