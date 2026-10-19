# LangChain / LangGraph / DeepAgents 全景分析

> 写作时间：2026-02（依据官方文档与源码抓取结果）
> 证据等级标注：`[文档]` = 官方文档原文可查；`[源码]` = 官方仓库源码可查；`[推断]` = 本文基于上述事实的分析结论。

---

## 一、结论先行

三者的关系**不是三个并列的竞品**，而是**同一技术栈的三个层次**：

| 层次 | 项目 | 一句话定位 |
|---|---|---|
| 编排/运行时层（低） | **LangGraph** | 有状态、可持久化、可中断恢复的**图执行运行时** |
| 应用框架层（中） | **LangChain** | 用 `create_agent`、Model/Tool/Middleware 等抽象**快速搭 Agent** |
| 开箱即用层（高） | **DeepAgents** | 在 LangChain 的 `create_agent` 上，用中间件**预装**规划、虚拟文件系统、子代理、自动摘要 |

依赖方向严格单向：

```
DeepAgents  →  LangChain  →  LangGraph
（预装能力）   （Agent 抽象）  （图运行时 / 状态 / checkpoint）
```

官方 Deep Agents 仓库 README 的原话就是这个意思：Deep Agents 是 **"a high-level harness built on top of LangChain and LangGraph"** `[源码]`。
源码层面可以验证：`deepagents/graph.py` 里 `create_deep_agent(...)` 的最后一步就是
`return create_agent(model, ...).with_config({...})` —— 它自己不实现 Agent 循环，而是**给 LangChain 的 `create_agent` 装上一条中间件栈** `[源码]`。

**学习顺序因此只能是：LangChain → LangGraph → DeepAgents**（逆序学会变成"只会调 API 不懂为什么"，或者"读不懂 DA 源码"）。

---

## 二、三者分别是什么

### 2.1 LangChain —— Agent 应用框架

官方把它定位为 "framework for building agents"，核心是把「模型 / 工具 / 消息 / 中间件」这几个概念标准化 `[文档]`。

**核心概念**

| 概念 | 说明 |
|---|---|
| `create_agent` | 一行搭出一个 ReAct 风格 Agent（模型 ↔ 工具循环） |
| Model | 用 `"provider:model"` 统一字符串寻址，如 `"openai:gpt-5.5"`、`"claude-sonnet-4-6"`、`"ollama:devstral-2"` |
| Tool | `@tool` 装饰器，**函数名/描述/参数签名会自动进入模型 prompt** |
| Message | `HumanMessage` / `SystemMessage` / `AIMessage`（含 `tool_calls`）/ `ToolMessage` |
| Middleware | 在 Agent 循环的固定切入点做拦截：改 prompt、改工具、加记忆、加审批 |
| checkpointer | 传 `InMemorySaver()` + `config={"configurable":{"thread_id":...}}` 即有短期记忆 |

**最小可用代码** `[文档]`

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]})
print(result["messages"][-1].content_blocks)
```

**它替你做了什么**：把「LLM 判断要不要调工具 → 调工具 → 把结果喂回去 → 再判断」这个循环封装掉了。**代价**：你只拿到抽象，看不到状态机细节。

---

### 2.2 LangGraph —— 图运行时（真正的引擎）

官方定位是 low-level orchestration framework，提供 **durable execution（持久执行）** 与**状态管理**。`create_agent` 本身就是构建在 LangGraph 之上的 `[文档]`。

**核心概念**

| 概念 | 说明 |
|---|---|
| `StateGraph` | 以「状态」为中心的有向图 |
| Node | 普通 Python 函数，输入 state、输出 state 的局部更新 |
| Edge / Conditional Edge | 固定跳转 / 按函数返回值路由（`should_continue` 这种） |
| `compile()` | 编译成可执行图，可挂 checkpointer / store |
| Reducer | `Annotated[list, operator.add]` 决定字段是「覆盖」还是「追加」 |
| Functional API | `@entrypoint` + `@task`，用普通控制流写图，等价能力 |
| Checkpointer / thread_id | 断点持久化，支撑多轮记忆、时间旅行、人工审批后恢复 |
| `interrupt` | 在节点内暂停等人，恢复后从中断点继续 |

**Graph API 骨架** `[文档]`

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state) -> Literal["tool_node", END]:
    return "tool_node" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()
```

**为什么值得单独学**：一旦要做「多阶段、可回滚、中途要人工确认、跑一半崩了要能续跑」的系统，LangChain 的 `create_agent` 就不够用了，必须下潜到这一层。LangGraph 的价值不在"Agent"，而在**状态机的工程可靠性** `[推断]`。

---

### 2.3 DeepAgents —— 预装能力的 Harness

官方定位是 harness / "batteries-included"，最著名的先例是 **Claude Code、Deep Research、Manus** 这类"深度"智能体的通用骨架 `[文档]`。

**它预装了什么**（官方文档 Harness 章节 + 源码中间件栈双重印证）

| 能力 | 实现 | 暴露给模型的工具 |
|---|---|---|
| 任务规划 | Planning / TODO | `write_todos` |
| 虚拟文件系统 | Filesystem Middleware | `ls` `read_file` `write_file` `edit_file` `glob` `grep` |
| 子代理 | SubAgent Middleware | `task`（默认自动挂一个 `general-purpose` 子代理） |
| 上下文压缩 | Summarization Middleware | 自动，无工具 |
| 记忆 | Memory Middleware | 启动时把 `AGENTS.md` 注入 system prompt |
| 技能 | Skills Middleware | 按需披露 skill |
| 人工审批 | HumanInTheLoop Middleware | `interrupt_on={"edit_file": True}` |
| Shell 执行 | 后端提供 `SandboxBackendProtocol` 才有 | `execute` |

**关键源码事实（`libs/deepagents/deepagents/graph.py`）** `[源码]`

签名：

```python
create_deep_agent(
    model, tools, *,
    system_prompt, middleware, subagents, skills, memory,
    permissions, backend, interrupt_on, response_format,
    state_schema, context_schema, checkpointer, store, debug, name, cache,
)
```

中间件装配顺序（**这就是 DeepAgents 的架构本体**）：

```
FilesystemMiddleware
  → SubAgentMiddleware
  → SummarizationMiddleware
  → PatchToolCallsMiddleware
  → AsyncSubAgentMiddleware（有异步子代理时）
  → 【用户自定义 middleware 插在这里】
  → SkillsMiddleware（传了 skills 才挂）
  → Anthropic / Bedrock / Fireworks PromptCaching
  → MemoryMiddleware（传了 memory 才挂）
  → HumanInTheLoopMiddleware（有 interrupt_on 才挂）
  → UnsupportedContentMiddleware
  → ToolExclusionMiddleware
```

另外两个容易被忽略但很重要的设计：

1. **`DeepAgentState` 给 `messages` 用了 `DeltaChannel`**，注释原话是把 checkpoint 的增长从 **O(N²) 降到 O(N)** —— 长任务不爆存储的关键 `[源码]`。
2. **`recursion_limit` 被设成 9999**（普通 LangChain Agent 默认远小于此），即"允许长时间自主循环"，这是"deep"这个词的工程含义之一 `[源码]`。

**子代理有三种形态** `[源码]`：`SubAgent`（声明式，走 `task` 工具）、`CompiledSubAgent`（传现成 runnable）、`AsyncSubAgent`（远程/后台，走 LangSmith deployment）。

---

## 三、横向对比

| 维度 | LangChain | LangGraph | DeepAgents |
|---|---|---|---|
| 抽象层级 | 高（声明式 API） | 低（图/状态/边） | 最高（预装 harness） |
| 搭一个 Agent 的代码量 | ~10 行 | ~60 行 | ~5 行 |
| 能否控制循环细节 | 只能通过 middleware | 完全可控 | 通过 middleware / profile |
| 持久化 & 恢复 | 挂 checkpointer 即有 | 原生、可精细控制 | 继承自下层 |
| 长期任务能力 | 需自己加摘要 | 需自己画图 | 内置摘要 + 虚拟 FS + 子代理 |
| 学习曲线 | 平缓 | 陡（要懂状态机/通道/checkpoint） | 看起来最平缓，**读懂最难** |
| 典型坑 | 误以为它是"万能 Agent" | reducer 写错导致消息被覆盖；忘记 `thread_id` | 以为它"更聪明"，其实是工具更多；不看 middleware 顺序就改不动行为 |
| 适合谁 | 做业务 Agent、快速验证 | 做平台/长任务/需审批系统 | 做"深度研究/编码代理"类产品 |

---

## 四、一个真实的对照实验（官方文档给的，很有说服力）

官方 Quickstart 让两个 Agent 回答同一道题：统计 Gutenberg 上《了不起的盖茨比》全文中包含 `Gatsby` 的行数、`Daisy` 首次出现的行号 `[文档]`。

- `create_agent`（只有 `fetch_text_from_url` 一个工具）：
  返回 `null`，并诚实解释"我没有代码执行环境，无法可靠地逐行计数，所以不编造数字"。
- `create_deep_agent`（同一模型、同一个自定义工具）：
  **258 行 / 第 181 行**，并说明过程：文件太大被自动落到 `/large_tool_results/x246ax2x`，然后用 `grep` 工具检索、拿到带行号的匹配结果、再人工核对大小写。

**这个实验的真正含义** `[推断]`：模型的"能力"边界，很大程度上是**工具体系 + 上下文管理策略**的边界，而不是模型权重本身。DeepAgents 的增量价值是：
1. 大输出自动落盘（不让它撑爆 context）；
2. 给模型**确定性工具**（grep 而不是"心算"）；
3. 有 `write_todos` 让它先拆解任务。

这三点正好对应你后续要学的三件事：**上下文工程、工具设计、任务分解**。

---

## 五、该选哪个（决策表）

```
需求是"业务里塞个会调工具的对话"?
  └─ 是 → LangChain create_agent（够用，别过度设计）
  └─ 否 ↓

需求是"多阶段流程 / 需要人工审批 / 要能断点续跑 / 要可视化状态"?
  └─ 是 → LangGraph（自己画图）
  └─ 否 ↓

需求是"长时间自主研究/编码，要读写文件、要拆任务、要子代理"?
  └─ 是 → DeepAgents（先跑通，再按 middleware 改造）
  └─ 否 ↓

两边都要 → 用 DeepAgents 起手，卡住的地方下潜到 LangGraph 那一层改
```

一个工程上常见的演进路径：`create_agent` 原型 → 发现需要审批/恢复 → 换 LangGraph 手写图 → 发现重复造轮子（摘要、文件、子代理）→ 回到 DeepAgents 并定制 middleware。**所以这三者都要懂，只是学习的先后问题。**

---

## 六、证据与未验证项（诚实边界）

**已验证（本次实际抓取）** `[文档]`/`[源码]`
- 三方官方定位、Docs Map 结构、Quickstart 可运行代码（LangChain / LangGraph 各一份）
- `deepagents/graph.py` 的签名、中间件栈顺序、`DeltaChannel`、`recursion_limit=9999`
- `create_agent` 的 checkpointer/thread_id 用法、`content_blocks` 输出约定
- LangGraph Graph API 与 Functional API 的等价骨架

**未验证 / 需你自行确认**
- 各模型名（`gpt-5.5`、`claude-sonnet-4-6`、`gemini-3.1-pro-preview` 等）是否在你的账号可用、价格如何 —— 文档会随模型发布快速漂移，**以 `init_chat_model` 报错信息为准**
- 具体版本号（langchain / langgraph / deepagents 的当前 release tag）—— 用 `pip index versions <pkg>` 或 PyPI 页确认
- LangChain Academy 的 Deep Agents 课程：课程页已存在，但官方标注为"持续监控/尚未正式发布"状态，**开课时间未知**
- 各框架的 token 成本实测数据（本文未做基准测量）
