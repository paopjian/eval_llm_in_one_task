# cc-switch-cli 内部 Token 消耗记录

> 生成时间:2026-09-07 (UTC)
> 工具:`cc-switch` v5.10.4 (即 GitHub `saladday/cc-switch-cli` 项目发布的 CLI 二进制)
> **约定:文中所有 USD/美元数值仅为内部定价折算的"标记"**,只作量级参考与对比,不代表实际计费或账单金额,不作明确表述(总账见 `总账-token消耗.md`)。

## 0. 背景与环境要点(复现方法)

- 二进制位置:`/root/.local/bin/cc-switch`(仓库名 cc-switch-cli,安装后命令名为 `cc-switch`)。
- 默认数据目录:`/root/.cc-switch/`(SQLite: `cc-switch.db` + `session-scan-cache.db` + `session-pages-v1/`)。
- 本机根文件系统 `/` 为**只读**,cc-switch 初始化时无法写 `cc-switch.db.init.lock`,
  因此通过环境变量把数据目录重定向到可写副本运行:

```bash
export CC_SWITCH_CONFIG_DIR=/root/zhaokj/test_model/eval_llm_in_one_task/ccswitch-workdir/home
cc-switch provider list          # 验证可运行
cc-switch provider current       # 查看当前 provider
```

- 数据目录副本:`ccswitch-workdir/home/`(复制自 `/root/.cc-switch/`,含 WAL,未改动)。
- 实时远程配额 `cc-switch provider quota <id>` 返回 `not_available`(该 provider 未配置 Usage Query),
  故"内部 token 消耗"以本地 DB 记录为准。

## 1. 数据来源(cc-switch 内部如何记录 token)

| 表 / 文件 | 内容 | 行数 |
|---|---|---|
| `proxy_request_logs`(cc-switch.db) | 逐请求 token 消耗:输入/输出/缓存读/缓存写 tokens、按模型定价折算 USD、请求时刻、模型、provider、app | 9,625 |
| `usage_daily_rollups`(cc-switch.db) | 按天聚合缓存(codex 早期用量),字段含 date/app/provider/model/tokens/费用 | 25 |
| `session_log_sync` | CLI 会话日志文件增量同步位置(断点续扫) | 163 |
| `session_usage_dedup` | 会话用量去重键 | 22 |
| `model_pricing` | 模型定价(每百万 token 价格,192 行) | 192 |
| `session-pages-v1/` | 各 app(claude/codex/gemini)会话页缓存(manifest/page json) | — |

`proxy_request_logs.data_source`:`session_log`(claude 会话日志)、`codex_session`(codex rollout)、`pi_session`。
即 cc-switch 通过增量扫描各 CLI 的会话日志,把每次 API 调用的 usage 字段录入本地库并换算费用。

## 2. 总账(截至 2026-09-07)

### 2.1 proxy_request_logs — 近期窗口 2026-08-11 ~ 09-07

| 指标 | 数值 |
|---|---|
| 请求数 | 9,625 |
| 输入 tokens | 230,555,620 |
| 输出 tokens | 5,542,021 |
| 缓存读 tokens | 839,687,335 |
| 缓存写 tokens | 49,676,412 |
| 估算成本(USD) | **$1,198.12** |

按 app×模型(TOP):

| app | provider | model | req | input_tok | output_tok | cache_read | cache_write | cost USD |
|---|---|---|---|---|---|---|---|---|
| claude | _session | claude-opus-5 | 7,606 | 33,584,287 | 3,896,384 | 664,810,603 | 49,406,190 | 906.53 |
| codex | _codex_session | gpt-5.6-sol | 1,817 | 196,393,952 | 1,370,069 | 163,149,532 | 0 | 288.90 |
| claude | _session | qwen3.8-max | 56 | 336 | 57,188 | 3,633,946 | 188,612 | 1.72 |
| claude | _session | claude-sonnet-5 | 11 | 51,199 | 4,134 | 73,030 | 7,702 | 0.27 |
| claude | _session | claude-sonnet-4-6 | 13 | 39 | 3,859 | 309,766 | 27,781 | 0.26 |
| claude | _session | glm-5.3 / glm-5.3-flash | 97 | 511,565 | 206,351 | 7,613,056 | 0 | 0.00* |
| claude | _session | grok-4.5 | 3 | 2,882 | 1,243 | 1,984 | 0 | 0.01 |
| pi | claude-awsq | claude-opus-5 / sonnet-5 | 21 | 11,360 | 2,793 | 95,418 | 46,127 | 0.44 |

*glm 为第三方/特殊定价,费用字段为 0 或按自定义 pricing 折算(见 model_pricing)。

### 2.2 usage_daily_rollups — 早期 codex 窗口 2026-03-17 ~ 08-04

| 指标 | 数值 |
|---|---|
| 天数 | 25 |
| 输入 tokens | 24,224,106 |
| 输出 tokens | 2,090,013 |
| 缓存读 tokens | 221,213,184 |
| 估算成本(USD) | **$248.49** |

模型:gpt-5.4(2026-03-17~04-23)、gpt-5.6-sol(2026-07-10~08-04),provider=`_codex_session`,全部成功。
单日最大:2026-08-04 请求 370 次 / $51.63;2026-07-15 缓存读 30.1M。

## 3. 记录文件

| 文件 | 说明 |
|---|---|
| `requests_detailed.csv` | 9,625 条请求级明细(全部 24 字段,按时间升序) |
| `daily_by_model.csv` | 按 日期×app×provider×model 汇总 |
| `model_summary.csv` | 按 app×provider×model 汇总(即上文 TOP 表) |
| `usage_daily_rollups.csv` | 早期 codex 按天聚合原始导出 |
| `README.md` | 本说明 |

## 4. 备注

- 费用为 cc-switch 按 `model_pricing` 定价折算的估算值,非账单。
- 时区:proxy_request_logs.created_at 为 Unix epoch 秒,CSV 日汇总使用本地时区(UTC)。
- 明细中不含 API Key 等敏感字段;providers 的 `settings_config`(含凭据)未导出。
