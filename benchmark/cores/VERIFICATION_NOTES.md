# 各模型 core 冒烟/验证记录

GPU 冒烟（N=25000 切片，对照朴素 GPU 参考直方图，7 卡实测）结论见 `logs/unified_eval/smoke_gpu/` 与
`logs/unified_eval/smoke_retry/`。计数必须严格精确（pos==理论正对数、neg==理论负对数）。

| core | 状态 | 说明 |
|------|------|------|
| baseline | ✅ PASS | 计数精确、指标与参考完全一致 |
| glm | ✅ PASS | 计数精确、指标与参考完全一致 |
| grok | ✅ PASS | 计数精确、指标与参考完全一致 |
| qwen | ✅ PASS | 计数精确、指标与参考完全一致 |
| codex-sol-2 | ✅ PASS | 计数精确；tf32 量化致阈值差 ≤1 bin |
| codex-sol-m | ✅ PASS | 计数精确；阈值差 ≤1 bin |
| deepseek | ✅ PASS | 计数精确、指标与参考完全一致 |
| deepseek-pro | ✅ PASS | 计数精确；200K 与 2M 全量均实测通过（见代理报告） |
| claude-opus | ✅ PASS | 计数精确、指标与参考完全一致 |
| claude-opus-m | ✅ PASS | 计数精确（修复 worker 中 np.bincount 收到 2-D 数组的 bug 后） |
| codex-sol | ✅ PASS | 计数精确；fp16 量化致阈值差 ~6 bin（方法固有，见其 docstring） |
| gemini | ✅ PASS | 计数精确；fp16 量化致阈值差 ~6 bin（方法固有，见其 docstring） |
| glm-pro | ❌ 放弃（复现崩溃） | ① 各 GPU worker 统计对数与行带上三角面积差 ±1~33 对（微小，
  已按容差放行）；② 正样本路径 gpu6 反复出现
  `vectorized_gather_kernel index out of bounds` → CUDA 断言 → cublasSgemmStridedBatched
  崩溃（200K 正式跑仍复现），判定为原实现（glm-pro/step3_multi_gpu.py）分组 gather 索引缺陷。
  按"修不好就放弃"规则：**glm-pro 不计成绩，200K/2M 均记录为 error 退出**。 |

## 判定规则（用户确认）

- 200K 阶段 wall 超 60s 或计数校验失败 → 不参加 2M 阶段（记录原因即可）
- core 修不好 → 放弃该模型、记录原因、继续下一个
- 计数有微小误差（如 glm-pro ±几十对）→ **允许继续后续阶段**，框架校验按
  rel 1e-5 / 绝对 10 对容差放行，结果中记录 delta 与 strict_ok
