# LangGraph 教材 · 总纲（待展开）

> 状态：**只有大纲，正文未写。**
> 对象：`langgraph` 1.x（源码 `langchain-ai/langgraph`，与 `langchain` 是**两个仓库**）
> 姊妹教材：LangChain → `../langchain_learn/00-syllabus.md`；Deep Agents → `../deepagents_learn/00-syllabus.md`

---

## 一、范围声明

本教材只管 **图运行时**：图怎么被执行、状态怎么合并、检查点怎么存、中断怎么恢复、流式从哪来。

- **LangChain**（`create_agent`、middleware、模型/工具抽象）不在这里 → `../langchain_learn/`
- **Deep Agents**（文件系统、子代理、skills 那套 harness）不在这里 → `../deepagents_learn/`

判据一句话：**问「图是怎么跑的」翻这本；问「agent 该长什么样」翻那两本。**

---

## 二、8 章地图

| 章 | 标题 | 官方锚点（`docs.langchain.com/oss/python/`） | 源码落点 | 验收产出 |
|---|---|---|---|---|
| **L0** | 图运行时是什么 | `langgraph/overview`、`thinking-in-langgraph`、`pregel` | `langgraph/pregel/` | 能说清一次"超步（superstep）"里发生的事：谁读状态、谁写、为什么能并行 |
| **L1** | Graph API 与 Functional API | `langgraph/graph-api`、`functional-api`、`choosing-apis` | `langgraph/graph/` | 同一逻辑两种写法各写一遍，说出什么时候**必须**用 Graph |
| **L2** | State / Reducer / Channel | `langgraph/use-graph-api`、`langgraph/add-memory` | `langgraph/channels/`、`graph/message.py` | 自己写一个 reducer，并解释两个并行节点为什么不会互相覆盖 |
| **L3** | 持久化：checkpointer 与 store | `langgraph/persistence`、`checkpointers`、`stores` | `langgraph/checkpoint/` | 换成 `SqliteSaver` 落盘，**杀掉进程再启动**接着跑 |
| **L4** | 时间旅行 | `langgraph/use-time-travel` | `get_state_history` / `update_state` | 回滚到第 3 步，并 fork 出一条新分支 |
| **L5** | 中断与恢复 | `langgraph/interrupts` | `langgraph/types.py` 的 `interrupt()`、`Command(resume=)` | 一个缺参数就要问人的节点，能从断点接着跑完 |
| **L6** | 子图与作用域 | `langgraph/use-subgraphs` | 子图编译 + checkpointer 传递 | 说清"每次调用一个检查点"和"整条线程共享"的差别在哪一行 |
| **L7** | 流式全模式 | `langgraph/streaming`、`event-streaming` | `stream_mode` 各取值 | 六种 `stream_mode` 各跑一次，画出各自吐什么 |

每章仍按 LangChain 那本的七段模板走（①得到什么 / ②源码解剖 / ③你敲这段 / ④你应该看到 / ⑤三个破坏实验 / ⑥自检题 + 内联答案 / ⑦承上启下）。

---

## 三、推荐路径（按你的实际目标选一条）

| 目标 | 读哪几章 |
|---|---|
| 只想读懂 `create_agent` 为什么这么跑 | L0 → L2 → L7 |
| 要做**多轮有记忆**的对话/客服 | L2 → L3 → L4 |
| 要做**人工审批**的长流程 | L3 → L5 → L6 |
| 要做**前端实时输出** | L2 → L7 + `langchain/frontend/` |
| 要做**可回滚/可审计**的生产 agent | L3 → L4 → L6 + `langgraph/fault-tolerance` |

---

## 四、与 LangChain 那本的接口

| LangChain 那本 | 在这里变成 |
|---|---|
| ch05 `AgentState` / `add_messages` | L2：Channel 与 Reducer 的真实实现 |
| ch06 `create_agent(checkpointer=...)` | L3：checkpointer 全貌；L6：子图作用域 |
| ch10 `stream()` 与 `stream_mode` | L7：六种模式的分工 |
| ch09 写 middleware | L5：`interrupt()` 就是 `HumanInTheLoopMiddleware` 的底座 |
| ch15 多智能体四形态 | L6：subagents 就是"子图 + 独立状态" |

---

## 五、诚实边界

- 正文未写。锚点来自官网机器可读索引 `https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt`。
- `langgraph` 与 `langchain` **不在同一个仓库**：`langchain-ai/langgraph` vs `langchain-ai/langchain`（`libs/langchain_v1`）。读源码时别在 langchain 仓库里找 `pregel/`。
- 本机 `pwsh` 目前被 Windows 文件权限拦住（`grantWrite` Win32 5），未实机验证；"你应该看到"均为按源码推导的期望值。
