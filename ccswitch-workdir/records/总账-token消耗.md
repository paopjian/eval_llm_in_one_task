# cc-switch-cli 内部 Token 消耗总账

> 数据截止:2026-09-07(cc-switch.db 快照至 09-07 03:57 UTC)
> 工具:`cc-switch` v5.10.4(即 GitHub saladday/cc-switch-cli);数据目录副本:`ccswitch-workdir/home/`
> 环境注意:根分区只读,用 `CC_SWITCH_CONFIG_DIR=ccswitch-workdir/home` 重定向运行,详见 `README.md`

## 0. 口径与约定(先读)

- **USD / 美元数值仅为"标记"**:文中所有 USD 数值(含表头"USD标记")都是按本地 `model_pricing` 定价折算出的参考标记,只用于**量级参考与组间对比**,不代表实际计费或账单金额;不表述为确切的美元消耗。中转站侧原始数值同理,仅照录、单位未确认。
- **记录来源**:
  - cc-switch 库:`proxy_request_logs`(逐请求,9,625 行,claude/codex/pi 会话扫描)、`usage_daily_rollups`(早期 codex 官方历史迁移的按天聚合);
  - 原生 CLI(grok/gemini 不被 cc-switch 记录):grok 会话 `updates.jsonl` 的 `turn_completed` 回执;Gemini CLI 会话文件逐轮 `tokens`。
- 说明:glm-5.3 / glm-5.3-flash 不在定价表,USD标记为 0;`total=input+output+thoughts`(gemini),cached 另计。

## 1. 各组消耗汇总(本次核对范围)

| 组 | CLI | 模型(实际) | 时段 | 请求/条目 | 输入(新) | 输出 | 缓存读 | 缓存写/思考 | USD标记* |
|---|---|---|---|---|---|---|---|---|---|
| qwen 测试 | Claude Code | qwen3.8-max | 09-06 | 56 req | 336 | 57,188 | 3,633,946 | cacheW 188,612 | 1.72 |
| Zhipu GLM(普通) | Claude Code | glm-5.3-flash | 09-05~06 | 33 req | 307,729 | 59,890 | 1,935,424 | — | 0* |
| Zhipu GLM pro版 | Claude Code | glm-5.3 | 09-06 | 64 req | 203,836 | 146,461 | 5,677,632 | — | 0* |
| 测试grok | grok CLI→bytecatcode | grok-4.6-build | 09-05 | 26 model calls | 1,616,538 | 59,882 | 1,429,248 | reasoning 31,514 | ticks 2,705,331,800 |
| 测试gemini | Gemini CLI→bytecatcode | gemini-3.5-flash | 09-05 | 39 轮 | 1,458,328 | 32,006 | cached 952,938 | thoughts 9,842 | — |
| 测试opus5 **max** | Claude Code | claude-opus-5 | 09-05 | 154 req | 72,174 | 78,461 | 14,831,901 | cacheW 1,082,946 | 16.51 |
| 测试opus5 **medium** | Claude Code | claude-opus-5 | 09-06 | 45 req | 90 | 22,035 | 2,794,731 | cacheW 98,840 | 2.57 |
| codex sol **max**①(codex-sol) | Codex | gpt-5.6-sol | 09-05 | 134 req | 10,158,748 | 122,847 | 9,186,129 | — | 13.14 |
| codex sol **max**②(codex-sol-2 重启) | Codex | gpt-5.6-sol | 09-05 | 97 req | 10,789,186 | 216,626 | 8,826,341 | — | 20.73 |
| codex sol **medium** | Codex | gpt-5.6-sol | 09-06 | 35 req | 809,180 | 7,961 | 676,864 | — | 1.24 |

\* USD标记仅为内部折算参考值:智谱两行为 Coding Plan 套餐口径且模型无定价行故记 0;不做任何计费表述。

有 USD标记的测试组标记值合计 ≈ 55.90(qwen 1.72 + opus5 max 16.51 + opus5 medium 2.57 + sol max 33.87 + sol medium 1.24;glm/grok/gemini 不计)。

### 任务执行时间(UTC;口径:首请求~末请求,另列会话生命周期)

| 组 | 开始 | 结束 | 请求跨度 | 会话生命周期 | 备注 |
|---|---|---|---|---|---|
| qwen 主任务 | 09-06 04:13:23 | 09-06 04:50:43 | 37m20s | 40m41s | 冒烟 04:07~04:10 |
| GLM 普通主任务(flash) | 09-05 18:56:09 | 09-06 02:34:05 | 7h37m56s | 7h53m26s | 31 请求隔夜分散,含大段空闲 |
| GLM pro 主任务 | 09-06 02:58:53 | 09-06 03:54:07 | 55m14s | 1h13m42s | |
| grok 主任务 | 09-05 18:21:27 | 09-05 18:54:10 | 32m42s(会话) | 同左 | 24 次 model calls,分两段回合 |
| gemini 主任务 | 09-05 15:57:18 | 09-05 16:09:42 | 12m23s(首末消息) | 同左 | 37 轮;冒烟 1m02s |
| opus5 max | 09-05 09:26:21 | 09-05 13:07:15 | 3h40m54s | 7h10m57s | 13:07 后无 API 请求至 16:36(人工查看/收尾?) |
| opus5 medium | 09-06 05:22:57 | 09-06 05:50:33 | 27m36s | 35m54s | |
| sol max① 两段合计 | 09-05 13:16:12 | 09-05 16:05:15 | 2h49m03s | — | 段间空档 ≈24min(15:34→15:58);1a 段 2h18m12s、1b 段 7m02s |
| sol max② 重启 | 09-05 16:16:35 | 09-05 17:16:04 | 59m29s | — | |
| sol medium | 09-06 06:38:46 | 09-06 08:41:10 | 2h02m24s | — | |

说明:claude/codex 会话另有"生命周期=创建~最后活跃"(扫描缓存),若明显大于请求跨度,说明存在无 API 调用的间隔(人工查看/中断)。根目录 opus5 大会话(4fb,283 请求)跨 09-05 08:47→09-06 02:56(≈18h,隔夜,含长空闲),未计入上表。

### 有效执行时间(剔除 >10min 空档;事件时间戳口径)

规则:对会话原始文件逐条事件时间戳,相邻间隔 ≤10min 的计为连续执行并累加;间隔 >10min 视为"隔夜/等待确认/中断"剔除。阈值可调(另附 5min 口径对照)。

| 组 | 会话墙钟 | 有效执行(>10min剔除) | 剔除空档 | 备注 |
|---|---|---|---|---|
| qwen 主 | 40m41s | 40m41s | 23h02m(1 段) | 空档为文件尾部 09-07 异常事件(无 API 请求,疑似重开/系统写入) |
| GLM flash 主 | 7h53m26s | 46m50s | 7h06m35s(2 段) | 5min 口径为 29m43s;6~10min 间隔较多,可能含其自跑长任务 |
| GLM 5.3 主 | 1h13m42s | 57m15s | 16m27s(1 段) | |
| grok 主 | 32m42s(会话) | 23m49s | — | 按 API 回合时长(apiDurationMs 合计:1,209,865+219,443 ms) |
| gemini 主 | 12m23s | 12m23s | 0 | |
| opus5 max | 7h10m57s | 2h00m07s | 5h10m50s(7 段) | 5min 口径 1h37m01s;长停顿多(人工确认/中断) |
| opus5 medium | 35m54s | 35m54s | 0 | |
| sol max①(1a+1b) | 2h25m58s | 2h25m58s | 0 | 段间另有 ~24min 间隔(未计入会话内) |
| sol max② 重启 | 59m52s | 59m52s | 0 | |
| sol medium | 2h02m58s | 10m25s | 1h52m32s(3 段) | ⚠️ codex 事件较疏,长空档可能含其自跑长进程,此口径或低估 |

方法说明:claude/gemini 会话文件事件密度高,该口径较可靠;codex rollout 与 grok 需配合备注解读。空档合计 = 墙钟 − 有效执行。

### 关键会话/文件夹对照

| 组 | 文件夹 | 会话 |
|---|---|---|
| qwen | `/root/zhaokj/test_model/qwen` | c81fd4dd(54 req,任务);根目录 a98c010d(2,冒烟) |
| GLM 普通 | `.../glm` | 2dac401e(31,任务)+199e19ab(2,冒烟) |
| GLM pro | `.../glm-5.3` | c1763a82(64,任务) |
| grok | `.../grok` | grok 01a072cd(主)+01a072c5(冒烟,根目录) |
| gemini | `.../gemini` | a12cece3(37轮,主)+c2b2f9b2(2轮,冒烟) |
| opus5 max | `.../claude-opus` | 010e1f51(154,代码迭代 v1→v9) |
| opus5 medium | `.../claude-opus-m` | f894caa7(45) |
| sol max① | `.../codex-sol` | 01a071b5(120)+01a07234(14);⚠️ 运行1 可能接触过参考答案 |
| sol max② | `.../codex-sol-2` | 01a07254(97);重启后的干净重跑 |
| sol medium | `.../codex-sol-m` | 01a0756e(35) |

## 2. 全库记录总览(所有已记录用量)

| 来源 | 范围 | 请求 | 输入 | 输出 | 缓存读 | 缓存写 | USD标记合计* |
|---|---|---|---|---|---|---|---|
| proxy_request_logs | 2026-08-11 ~ 09-07(claude 7,786 + codex 1,817 + pi 22) | 9,625 | 230,555,620 | 5,542,021 | 839,687,335 | 49,676,412 | 1,198.12 |
| usage_daily_rollups | 2026-03-17 ~ 08-04(codex gpt-5.4/5.6-sol,25 天) | — | 24,224,106 | 2,090,013 | 221,213,184 | 0 | 248.49 |

\* 同上,仅为定价折算标记。未列入第 1 节的其余记录(主要为 CV/日常项目 claude 会话、根目录 opus5 三个会话标记 34.90、codex 其他会话、pi 少量)明细见 `requests_detailed.csv`、`daily_by_model.csv`、`model_summary.csv`。

## 3. 中转站(bytecatcode)侧原始数值(仅照录,单位未确认)

可用接口:`/v1/dashboard/billing/usage`(OpenAI 兼容;日期参数无效,仅累计值):

| key(provider) | 累计 total_usage(原始) | 备注 |
|---|---|---|
| default | 89,800.0448 | 仅照录 |
| default-copy(测试opus5) | 867.3876 | 仅照录 |
| byte(测试gemini) | 102.9394 | 仅照录 |
| default-copy-copy(测试grok) | — | 401,key 不被账单接口接受 |

其余管理面板类接口(`/api/*`、`/user/balance`)被服务端断开,后台数字需在 bytecatcode 站内确认;以上数值**不换算、不折算、不下结论**,仅作为对照线索留存。

## 4. 记录文件清单(records/)

| 文件 | 内容 |
|---|---|
| `ledger_summary.csv` | 本总账汇总(USD 列为标记值) |
| `requests_detailed.csv` | cc-switch 9,625 条请求级明细 |
| `daily_by_model.csv` / `model_summary.csv` | 按日×模型 / 按模型汇总 |
| `usage_daily_rollups.csv` | 早期 codex 按天聚合 |
| `qwen_requests_detailed.csv` / `qwen_sessions_summary.csv` | qwen 56 条明细 / 会话 |
| `glm_requests_detailed.csv` / `glm_sessions_summary.csv` | glm 97 条明细 / 会话 |
| `grok_usage_turns.csv` | grok 4 条 turn_completed 用量回执 |
| `gemini_turns_detailed.csv` | gemini 39 轮明细 |
| `opus5_sessions_summary.csv` / `opus5_max_medium_summary.csv` | opus5 五会话 / max·medium 两组 |
| `codex_sol_rollouts.csv` | codex 4 个 rollout 原始统计 |
| `README.md` | 环境与获取方法 |

## 5. 待确认/注意事项

1. glm-5.3、glm-5.3-flash 未入 `model_pricing` → USD标记为 0;如需自动折算需补价格。
2. 官网另有 glm-5.1 思考调用,但本地 cc-switch 与 claude 会话文件均无对应行(本地只记 5.3/5.3-flash),差异原因未定位。
3. grok 的 costUsdTicks 单位未知;测试grok 的 bytecatcode key 对账单接口返回 401。
4. 测试gemini 实测模型为 gemini-3.5-flash,而 provider 配置写的是 gemini-3.8-flash(bytecatcode 侧行为待核实)。
5. sol max① 会话可能接触过参考答案,benchmark 有效结论以重启的 codex-sol-2 为准,但两段都计入消耗记录。
