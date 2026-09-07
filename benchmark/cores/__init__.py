"""模型核心计算注册表。

每个 core 模块须实现:
    compute(feats, ids, gpus, workdir) -> (pos_hist, neg_hist, meta)
契约与工具见 benchmark.common。

core 是从各模型目录中的原始实现提炼出的“核心计算”（并行方式/分块/精度/
负载均衡等关键差异保留自原始代码），读取、校验、指标全部走统一框架。
"""
import importlib

CORE_NAMES = [
    'baseline',            # cluster_utils 基准方法（仓库外参考实现）
    'claude-opus', 'claude-opus-m',
    'codex-sol', 'codex-sol-2', 'codex-sol-m',
    'deepseek', 'deepseek-pro',
    'gemini',
    'glm', 'glm-pro',
    'grok',
    'qwen',
]


def get_core(name):
    # 注册表 id 用连字符（如 codex-sol），模块文件用下划线（codex_sol.py）：
    # 先试下划线名，失败（模块自身不存在）再退回原样，兼容两种命名。
    last_err = None
    for mod_name in dict.fromkeys([name.replace('-', '_'), name]):
        try:
            return importlib.import_module(f'benchmark.cores.{mod_name}')
        except ModuleNotFoundError as e:
            if e.name != f'benchmark.cores.{mod_name}':
                raise                      # 目标模块内部依赖缺失，如实抛出
            last_err = e
    raise last_err
