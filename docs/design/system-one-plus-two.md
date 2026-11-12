# System One + System Two：在 LitJev 上加慢思考的可能性

讨论稿，2026-09-18。**已被 `self-routing-architecture.md` 取代**：方向定为冻结骨干、只训并列决策头，
由模型自己决定快答还是慢想。本稿保留作为引擎外门控方案的对照。目标是回答一个问题：LitJev 现在是一只纯 System One（一次前向读头，零输出 token）。
在不破坏 Jev 协议、不训练的前提下，能不能在同一只 Qwen 上接一层 System Two，让它在需要时慢下来想一想？

## 0. 代码现状决定了哪些事是免费的

读完 `backend.py` / `slots.py` / `decision.py` / `prompting.py`，有三条事实：

1. **前缀是 chat template，`enable_thinking=False`。** `build_decision_messages` 的 system prompt 写死了
   "Do not explain or reason aloud"。Qwen3 系列自带 `<think>` 模式，所以 System Two 在模型层面不需要
   新模型，只需要换一套 prefix 和一次 generate。
2. **共享 prefill 的缓存可以复制。** `_score` 先跑 prefix 得到 `past_key_values`，再 `reorder_cache`
   复制成 N 份，接每题后缀。任何"先写一段东西再分叉"的流程都能塞进这两步之间。
3. **置信度已经在 decision 层。** `concentration()` 对每题给出 0–1 的集中度。升级门控不需要新信号。

另外两个约束：`/v1/systemone` 的标准响应只有 `model / answers / usage`，`usage.output_tokens` 恒为 0；
`DecisionSchema` 是无序 Mapping，题目之间没有依赖表达。

## 1. 五个层级，从便宜到贵

按"多花多少前向"排序。每一级都可以独立落地。

### L1：链式 System One（多跳，不生成）

问题：Jev 的隔离让"先判 A，再据 A 判 B"表达不了。

做法：允许题目声明依赖。引擎按拓扑序分轮次跑：第一轮跑无依赖的题，把它们的答案以文本形式
追加到共享前缀（`Decided: intent = replace`），再从新缓存分叉跑依赖题。每一轮仍是"缓存分叉 + 读头"，
零生成，只是前向次数从 2 变成 1 + 轮数。

```
prefix ──fork──> [intent, urgency]          round 1
   │
   └─ append "intent=replace, urgency=1.6"
             ──fork──> [escalate]           round 2
```

代价：一轮多一次前向，几十毫秒。收益：能表达决策树，仍然是 System One 的速度。
这不算 System Two，但它是很多"需要推理"的场景真正需要的东西。

### L2：想一次，判多题（shared thinking）

问题：有些 state 需要先消化才能答对，比如带算术、带多条事实、带截图细节。

做法：prefill 之后，在共享前缀上打开 `<think>`，用 `generate` 写一段推理，遇到 `</think>` 或到预算就停，
然后从**推理之后**的缓存分叉所有题目分支，照常读头。

```
prefix ──generate <think>…</think> (B tokens) ──fork──> [q1 … qN] 读头
```

关键性质：

- 推理成本被 N 道题**分摊**，只生成一次。
- 输出仍然是读头，类型安全不变，`answers` 结构不变。
- `usage.output_tokens` 第一次非零，而且语义正好对上 Jev 的 usage 字段。
- 推理内容对所有题可见，但题目之间依然互相不可见，隔离承诺保住。

代价：27B 在 HF Transformers 上生成大概每秒二三十个 token，256 token 的预算就是十秒量级。
这不是 Jev 的 70–500 ms，是另一档产品。

### L3：置信度门控升级（adaptive escalation）

问题：绝大多数题 System One 已经够用，全开 thinking 是浪费。

做法：先跑 L0（现状），对每题看 `confidence`。低于阈值的题进入 L2：只对这些题做一次 shared thinking
再重读。高于阈值的题直接返回。

```
S1 → confidence ≥ τ ? 返回 : 加入 escalate 集合
escalate 非空 → 一次 thinking → 只重读 escalate 集合
```

这是 Kahneman 意义上的双系统：快系统先答，慢系统只在快系统不确定时介入。
τ 直接控制延迟和准确率的折中，可以画出一条 coverage–accuracy 曲线，
这就是 selective prediction 的标准评测，MMLU-Pro 上现成能跑。

需要注意：`concentration` 未校准，用它做门控只表示"分布尖不尖"，不表示"对不对"。
L3 的门控质量取决于 L5。

### L4：生成加验证（open-ended + noul）

问题：Jev 只能在声明过的选项里选，开放式答案不在范式内。

做法：System Two 生成一个自由文本答案（比如一个抽取出来的字段、一个短回复），
然后把它塞进 state，用 System One 的 `noul` 问"这个答案对吗"，用 `score` 问"质量几档"。
生成的是候选，裁决的仍是读头。

这一层不需要改协议：调用方发两次请求就能拼出来。LitJev 可以提供一个便利端点，但不是必须。

### L5：System Two 给 System One 当老师（RLCD-lite）

问题：读头的概率不是校准过的。Jev 靠 RLCD 对齐，我们没有。

做法：在无标签数据上跑 L2 得到 System Two 的答案，把它当伪标签，
用来拟合 System One 的温度（现有 `litjev-calibrate` 直接能用），
或者进一步对 System One 的读头做轻量微调，让它的分布逼近 System Two 的分布。

这是自蒸馏，不是 RLCD，但方向一致：让快系统的置信度更诚实。
它也回答了博客第 8 节承诺的"未来尝试复现 RLCD"的第一步该从哪下手。

## 2. 协议怎么放

原则：`/v1/systemone` 一个字节都不动，它是和 Jev 互换的保证。

新增一个扩展端点，请求体是 Jev 的 `SystemOneRequest` 加一个可选块：

```json
{
  "model": "litjev",
  "state": "...",
  "questions": { "...": {} },
  "thinking": { "budget": 256, "escalate_below": 0.6 }
}
```

- `budget = 0` 且没有 `escalate_below`：退化成现状。
- 只有 `budget`：L2，全题共享一次思考。
- 有 `escalate_below`：L3，先 S1 再选择性升级。
- `depends_on` 挂在题目对象上：L1，仍然属于扩展，标准端点收到会 422。

响应形状不变，`usage.output_tokens` 反映思考 token 数。debug 端点额外给 `thinking.text`、
`thinking.tokens`、`escalated` 列表，以及每题是由 S1 还是 S2 给出的 `provenance.system`。

端点名可以是 `/v1/systemtwo`，也可以是 `/v1/systemone/think`。倾向前者：语义上它就不是 System One。

## 3. 实现落点

改动集中在 `backend.py`，其余模块几乎不动。

- `prompting.py`：加一套允许推理的 system prompt，chat template 用 `enable_thinking=True`。
  后缀的 `Answer:` 边界检查不变，字母码机制不变。
- `backend.py`：在 prefill 和 `reorder_cache` 之间插入一段 `generate`，输入是 prefix 的缓存，
  停止条件是 `</think>` 或预算。Qwen 混合注意力的缓存对象本来就支持增量生成，
  所以这一步是标准用法。生成结束后，缓存里已经包含推理 token，`prefix_length` 相应变长，
  后面的分叉代码不用改。
- `decision.py`：`forward_calls` 不再恒为 2；每题 provenance 加 `system: "one" | "two"`。
- L1 的拓扑调度放在 `SchemaDecisionEngine` 上层，一个新的 `ChainedDecisionEngine`，
  它多次调用 provider，每轮改 state。

不需要动：`schema.py` 的三种题型、`slots.py` 的编码、calibration、前端协议。

## 4. 怎么验证它值得做

MMLU-Pro 是现成的测试床，因为它的直答和 CoT 差距是公开数据里最大的之一。
用现有 `litjev-mmlu` 加一个 `--thinking-budget` 参数，跑四条线：

| 线 | 配置 | 看什么 |
|---|---|---|
| S1 | 现状 | 基线准确率和延迟 |
| S2 全开 | budget 256 / 512 | 上限准确率，延迟代价 |
| S1 + 门控 | τ ∈ {0.3, 0.5, 0.7} | coverage–accuracy 曲线 |
| S1 校准后门控 | 先 L5 再 L3 | 门控质量是否随校准变好 |

如果 S1 + 门控能用 20% 的升级率拿到 S2 全开 80% 的增益，这件事就成立了。
如果拿不到，说明 `concentration` 作为门控信号太弱，L5 要先做。

同样的四条线可以在 Doom / chess 上跑，但那里没有标签，只能看回合数和轨迹，说服力弱一些。

## 5. 哪些不做

- 不做 System Two 自己生成 JSON。裁决永远是读头，否则类型安全的卖点没了。
- 不做跨请求的记忆或多轮对话。System Two 的推理只活在这一次请求的缓存里。
- 不做题目之间互相可见。L1 的依赖是显式声明的，不是隐式泄漏。
- 不承诺延迟。System Two 是另一档产品，文档里要分开写。

## 6. 建议的第一刀

先做 L2 和 L3，跳过 L1。理由：

- L2 的代码量最小，一段 generate 插进现有两次前向之间。
- L3 只是在 L2 外面套一个阈值判断和一次重读。
- 两者一起就能跑第 4 节的评测，拿到第一条 coverage–accuracy 曲线，
  这条曲线决定后面 L1 / L5 值不值得投入。

L1 的价值很实在，但它解决的是"决策树"而不是"慢思考"，可以作为独立特性另开分支。

## 7. 待确认

- Qwen3.8-27B 的 thinking 模式在 Transformers 下是否稳定输出 `</think>`，还是要靠预算硬截。
- 混合注意力缓存在 `generate` 之后再 `reorder_cache` 是否保持循环状态正确，需要一个小模型测试
  对照"全量重算"结果，和现有 cached/full-input 等价性测试同一套路。
- 思考 token 的预算在 MMLU-Pro 上多少才够，256 可能只够写两三步。
