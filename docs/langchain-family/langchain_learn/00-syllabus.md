# LangChain 源码解剖教材 · 总纲

> 对象：**只讲 LangChain** —— `langchain` v1（`libs/langchain_v1/langchain/`）与 `langchain-core`。
> 学习者：你（已手写过 ReAct / Plan-and-Execute / Reflection，会 LangGraph 基础）
> 学习方式：**我讲一块源码 → 你手敲一块代码 → 你验证 → 你做破坏实验 → 你过自检题**
> 章节锚点：每一章都对应 `docs.langchain.com` 上的一个官方页面（见 §四、§五的"官方锚点"列）

> **范围声明**：LangGraph 与 Deep Agents **不在本教材范围内**——它们是两个独立框架，各有总纲：LangGraph → `../langgraph_learn/00-syllabus.md`；Deep Agents → `../deepagents_learn/00-syllabus.md`。
> 但要说清一件事：LangChain v1 的 `create_agent` 返回的就是一个 `CompiledStateGraph`，它的持久化、中断、流式都由 LangGraph 提供。所以本教材会在 **ch05 / ch06 / ch10** 碰到 LangGraph 的 API——只讲到"会用、知道谁在干活"为止，不钻实现。想钻，去那本。

---

## 一、这本教材和官方教程的区别

| | 官方 Quickstart | 本教材 |
|---|---|---|
| 出发点 | "怎么用" | "**它内部长什么样**" |
| 代码 | 复制粘贴即可跑 | **要求你手敲**，因为敲的过程会暴露你不懂的地方 |
| 验证 | 看输出对不对 | 看**你能预测输出**，以及**改坏了会不会报错** |
| 落点 | 会调用 API | 能读 `factory.py`、能写 middleware、能 debug |

一句话：**官方教程教你当用户，本教材教你能改它。**

---

## 二、学习契约（请先同意这四条）

1. **不复制粘贴。** 每章代码请手敲。敲错、报错、修好——这是学习本身。
2. **先预测再运行。** 每段代码运行前，先写下你预期会打印什么。不符就是认知缺口，记下来。
3. **每章必做破坏实验。** 至少故意改坏一次，看懂报错。不做的章节视为未完成。
4. **每章写一条"认知修正"。** 格式固定：`我原本以为…实际上…`。

---

## 三、全书结构：两部分，对应官网的两个分区

官网 Python 文档把 LangChain 的内容分成两大块：**Core components**（能跑起来）和 **Advanced usage / Multi-agent / MCP / Production**（能上生产）。本教材照这个切分：

| 部分 | 官网对应分区 | 解决的问题 | 章节 |
|---|---|---|---|
| **Part I 核心机制** | Core components + Middleware | agent **跑得起来**，且你知道它每一步在干什么 | ch00–ch11 |
| **Part II 高阶应用场景** | Advanced usage / Multi-agent / MCP / Agent development / Production | agent **敢上生产**：上下文、记忆、审批、多agent、检索、权限、容错、流式、评测、部署 | ch12–ch22 |
| **Part III 结课项目** | Learn → Tutorials | 把两部分合成一个能写进简历的东西 | 结课 |

**Part I 只做一件事**：把 `create_agent` 那一张图彻底拆开。
**Part II 只做一件事**：把官网那 20 多个"高级用法"页面归并成 11 个**场景**，每个场景给你一条可验收的产出。

> 为什么现在加 Part II：因为 Part I 学完你只会写"一个会调工具的 agent"。真实项目里 80% 的代码量花在 Part II 那 11 件事上——上下文塞爆、记忆接不上、审批做不出、多 agent 互相甩锅、工具权限收不住、线上不知道哪一步走偏。

---

## 四、Part I 核心机制（ch00–ch11）

> 章号 = 你手敲的文件序号。`ch` 指章节文档，`.py` 指对应代码。

| 章 | 标题 | 源码主要落点 | 官网锚点 | 状态 |
|---|---|---|---|---|
| **0** | 地图与环境 | `libs/*` 包结构 | `/oss/python/langchain/install` | ☑ |
| **1** | 一次 invoke 的真相 | `agents/factory.py` 图装配段 | `/oss/python/langchain/agents` | ☑ |
| **2** | Message 解剖 | `langchain_core/messages/` | `/oss/python/langchain/messages` | ☑ |
| **3** | Model 层 | `chat_models/init_chat_model.py` | `/oss/python/langchain/models` | ☑ |
| **4** | Tool 层 | `langchain_core/tools/` | `/oss/python/langchain/tools` | ☑ |
| **5** | State 与 Reducer | `agents/middleware/types.py` 的 `AgentState` + `factory.py:_resolve_schemas` | `/oss/python/langchain/short-term-memory` | ☑ |
| **6** | 会话持久化 | `langgraph/checkpoint/` + `channels/{base,binop}.py` | `/oss/python/langchain/short-term-memory`、`/oss/python/langgraph/checkpointers` | ☑ |
| **7** | 结构化输出 | `agents/structured_output.py` | `/oss/python/langchain/structured-output` | ☑ |
| **8** | Middleware 上：钩子地图 | `agents/middleware/types.py:385`（类）+ `factory.py:1163-1894`（编图） | `/oss/python/langchain/middleware` | ☑ |
| **9** | Middleware 下：写有用的 | 同上 + `HumanInTheLoop` | `/oss/python/langchain/middleware/custom` | ☐ |
| **10** | 流式 | `create_agent` 的 stream 契约 | `/oss/python/langchain/streaming` | ☐ |
| **11** | 测试与可观测 | `test` + LangSmith | `/oss/python/langchain/test/` | ☐ |

> ch08–ch09 是 Part II 的**总闸门**：Part II 里 7 个场景的官方实现都是 middleware。先把钩子地图吃透，后面每一章都会省一半时间。
>
> ch05 / ch06 / ch10 会用到 LangGraph 的 `add_messages`、`checkpointer`、`stream_mode`。**只要会用**，别停下钻——那本在隔壁。

---

## 五、Part II 高阶应用场景（ch12–ch22）

### A 组 · 能力扩展（让 agent 处理更长的任务）

| 章 | 场景 | 官方锚点 | 源码落点 | 验收产出 |
|---|---|---|---|---|
| **12** | **上下文工程**：长任务里塞什么、丢什么、何时摘要 | `/oss/python/langchain/context-engineering`、`/oss/python/concepts/context` | `agents/middleware/summarization.py`、`ContextEditingMiddleware` | 一个跑 20 轮以上的 agent，能画出每轮 token 曲线，并验证摘要在哪一轮触发 |
| **13** | **记忆三层**：短期 / 长期 / 用户画像 | `/oss/python/langchain/short-term-memory`、`/oss/python/langchain/long-term-memory`、`/oss/python/concepts/memory` | `agents/middleware/` + `create_agent(store=...)` | 新开一个 `thread_id`，agent 仍记得上一轮会话里你说过的偏好 |
| **14** | **人在环（HITL）**：审批、拒绝、改参数重试 | `/oss/python/langchain/human-in-the-loop`、`/oss/python/concepts/human-in-the-loop` | `agents/middleware/human_in_the_loop.py`、`interrupt_on` | 一个"发邮件必须先审批"的 agent：批准走完、拒绝不留痕、改参数后重跑 |
| **15** | **多智能体四形态**：subagents / handoffs / router / custom workflow | `/oss/python/langchain/multi-agent/`（index、subagents、handoffs、router、skills、custom-workflow） | `factory.py` 的子图组合 + `Command` 跳转 | **同一需求用四种形态各写一遍**，产出一张形态选型对照表（本书最值钱的一章） |
| **16** | **检索增强（RAG）**：检索当工具，而不是当流水线 | `/oss/python/langchain/retrieval`、`/oss/python/langchain/knowledge-base` | `langchain_core/vectorstores`、`documents` | 一个 **agentic RAG**：模型自己决定"先检索还是直接答"，并能在检索为空时改口 |
| **17** | **MCP 接入**：把外部能力当工具挂进来 | `/oss/python/langchain/mcp/`（index、tools、connections、auth） | `langchain/mcp`（v1 内置，替代 `langchain-mcp-adapters`） | 本地 stdio + 远程 HTTP 各接一个 server；处理鉴权失败与工具名冲突 |

### B 组 · 工程化落地（让 agent 敢上生产）

| 章 | 场景 | 官方锚点 | 源码落点 | 验收产出 |
|---|---|---|---|---|
| **18** | **护栏与安全**：输入输出双向过滤、PII 脱敏、工具白名单 | `/oss/python/langchain/guardrails`、`/oss/python/langchain/security-policy` | `agents/middleware/`（PII、tool selector） | 同一句越权指令，被输入护栏拦下；同一份日志，PII 被替换而非删行 |
| **19** | **容错**：重试、降级、限额、超时 | `/oss/python/langchain/middleware/built-in` | `ModelFallbackMiddleware` / `ToolRetryMiddleware` / `ModelCallLimitMiddleware` | 注入 3 类故障（429、工具抛错、死循环），验证各自的兜底路径被真的走到 |
| **20** | **流式与前端**：token 流、工具卡片、时间旅行 | `/oss/python/langchain/streaming`、`/oss/python/langchain/event-streaming`、`/oss/python/langchain/frontend/` | `create_agent.stream()` + `useStream` | 一个最小前端：token 逐字出 + 工具卡片 + 中断后续流 + 回滚到上一检查点 |
| **21** | **测试与评测**：假模型单测、轨迹评测、trace 定位 | `/oss/python/langchain/test/`（unit、integration、evals） | `FakeChatModel`、`InMemorySaver`、AgentEvals | 一套不花钱的单元测试 + 一条 LLM-as-judge 的轨迹评测，并演示"用 trace 定位哪一步走偏" |
| **22** | **部署**（选学，按需） | `/oss/python/langchain/deploy`、`/oss/python/langchain/studio`、`/oss/python/langchain/ui` | LangSmith Cloud / 自托管 | 本地用 Studio 跑起来、Agent Chat UI 接上、再谈托管 |

### Part II 推荐路径（不要从头到尾线性学）

| 你的目标 | 走这条 |
|---|---|
| **最小可用生产版**（先能用） | ch08 → ch09 → **ch12 → ch13 → ch14** → ch19 → ch21 |
| **多智能体/复杂编排** | ch08 → ch09 → ch05 → **ch15** → ch18 → ch12 |
| **接外部系统/知识库** | ch04 → **ch16 → ch17** → ch18 |
| **要做前端/产品** | ch10 → **ch20** → ch21 |

---

## 六、Part III 结课项目

用 **LangChain 的不同写法**做同一件事，写出差异报告：

| 版本 | 用什么写 | 你要证明的事 |
|---|---|---|
| V1 | `create_agent` 裸跑（model + tools） | 最少代码能到什么程度、什么时候就不够用了 |
| V2 | `create_agent` + middleware 全套 | 每一件 middleware 换来了什么、代价是什么 |
| V3 | 多形态重写（subagents / handoffs / router 各一版） | 同一个需求，四种拓扑各自的崩点 |

推荐题目（从官网 Tutorials 里挑，实现细节可查官方，但**你要写出自己的版本**）：

- `/oss/python/langchain/sql-agent`（SQL 分析，最贴合后端工程）
- `/oss/python/langchain/knowledge-base`（语义检索引擎）
- `/oss/python/langchain/multi-agent/subagents-personal-assistant`（多 agent 分工）

报告里必须回答：**三版各多少行、各在什么条件下会崩、你会把哪一版放进生产。**

---

## 七、每章统一模板（你按这个节奏走）

```
⓪ 官方锚点              官网哪一页 / 哪个教程，先扫一眼再读源码
① 本章你会得到什么        一句话，可验收
② 源码解剖               真实文件 + 真实代码片段 + 它在干什么
③ 你敲这段               ~30-60 行，带"为什么这么写"的逐行说明
④ 你应该看到             期望输出（先预测，再对照）
⑤ 三个破坏性实验         故意改坏，看报错
⑥ 自检题 3 道            答不上来就别进下一章（答案内联可见）
⑦ 承上启下              下一章为什么自然接上
```

Part II 的章节多一段：

```
⑧ 场景判据              什么时候**不该**用它（这一条比"怎么用"重要）
```

---

## 八、进度与打卡

| 部分 | 章 | 状态 | 日期 | 认知修正条数 |
|---|---|---|---|---|
| I | 0 地图与环境 | ☑ | | |
| I | 1 一次 invoke 的真相 | ☑ | | |
| I | 2 Message 解剖 | ☑ | | |
| I | 3 Model 层 | ☑ | | |
| I | 4 Tool 层 | ☑ | | |
| I | 5 State 与 Reducer | ☑ | 2026-10-22 | 3 |
| I | 6 会话持久化 | ☑ | 2026-10-23 | 1 |
| I | 7 结构化输出 | ☑ | 2026-10-24 | 2 |
| I | 8 Middleware 钩子地图 | ☑ | 2026-10-24 | 7 |
| I | 9 Middleware 实战 | ☐ | | |
| I | 10 流式 | ☐ | | |
| I | 11 测试与可观测 | ☐ | | |
| II | 12 上下文工程 | ☐ | | |
| II | 13 记忆三层 | ☐ | | |
| II | 14 人在环 | ☐ | | |
| II | 15 多智能体四形态 | ☐ | | |
| II | 16 检索增强 | ☐ | | |
| II | 17 MCP 接入 | ☐ | | |
| II | 18 护栏与安全 | ☐ | | |
| II | 19 容错 | ☐ | | |
| II | 20 流式与前端 | ☐ | | |
| II | 21 测试与评测 | ☐ | | |
| II | 22 部署 | ☐ | | |
| III | 结课项目 | ☐ | | |

> 每完成一章，把 `☐` 改成 `☑`。**跳章不算进度**；但 Part II 允许按 §五 的路径跳组。

---

## 九、本教材的诚实边界

- 源码引用基于 `master` 分支的 `libs/langchain_v1`（v1 线）。你本地装的若是 v0.x，文件名与内部结构会不同（见第 0 章的版本判定法）。
- **ch00–ch04 的"你应该看到"是按源码逻辑推导的**（写作时本机没有可用的执行通道）。**ch05 起改为实测**：第 5 章全部输出都是在本机 `langchain 1.4.6 / core 1.6.12 / langgraph 1.2.16` + `qwen3:latest@ollama` 上跑出来的；第 3 章的 callbacks 补充小节也已补实测（`langchain_learn/03_callbacks_demo.py`、`03_callback_params_trace.py`）。凡标"未实测"的地方，都明确写了是读源码推的。
- **ch00–ch07 末尾各加了一节「附录 · 官方写法对照」**（锚点是 overview / agents / messages / models / tools / short-term-memory / checkpointers / structured-output 八个官方页），对应代码 `langchain_learn/0X_official_*.py`。**这 8 节的输出全部是本机实测**，而且默认零 token：跑脚本只看编译期结构，加 `--real` 才发请求。附录里那些"我原本以为…实际上…"都是实测打脸记录，不是推演。
- 模型名会漂移。凡是出现具体模型名的地方，一律以你 `.env` 的 `LLM_MODEL_ID` 为准。
- **官方锚点会漂移。** `docs.langchain.com` 的页路径调整过多次（例如 `langchain-mcp-adapters` 已并入内置 `langchain.mcp`，middleware 从单页拆成 `/middleware/` 目录）。若某条路径 404，回到 `https://docs.langchain.com/llms.txt` 找新位置——那是官网自己给的机器可读索引，比站内搜索可靠。
- **Part II 有几章本机可能跑不全**：MCP 的远程 server、部署、前端 streaming 都会碰到网络或需要额外服务。遇到环境阻塞不算你的问题，把该章降级为"读懂 + 画出调用链"。
- **ch06 的"你应该看到"回到推导态**：写这一章时 `pwsh` 被 Windows 沙箱 ACL 拦住（`grantNamedSecurityInfoW failed (Win32 5)`），本机跑不了 Python，所以 `07-ch06-checkpoint-and-thread.md` 里的输出数字是**按源码推导的期望值**，不是实测。但源码事实是硬的——全部取自你 `.venv` 里已装的 `langgraph 1.2.16`（含行号），不是 GitHub master。你跑 `06_checkpointer.py` 时数字若不同，以你的为准。
- **LangGraph 的深水区不在这里。** ch05/ch06/ch10 只到"会用"为止；图运行时、checkpointer 实现、子图作用域、时间旅行原理都在隔壁那本。

---

## 十、现在开始

- Part I 从 [第 0 章](01-ch00-environment-and-map.md) 开始，预计 40 分钟。
- Part II 不要现在看。等你完成 ch08–ch09（Middleware 钩子地图），再回来选一条 §五 的路径。
- 如果你已经在写真实项目（`rfwp_agent` 之类），可以**倒着用**：先按 §五 的路径直接跳到你当下卡住的那个场景，读那一章的"⑧ 场景判据"，再回 Part I 补底层。倒着用比顺着用省时间，代价是偶尔要停下来补课。
