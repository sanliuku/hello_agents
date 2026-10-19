# LangChain / LangGraph / DeepAgents 学习方案

> 本方案基于你**当前仓库的真实状态**定制，不是通用课表。

---

## 一、起点诊断：你已经有什么，缺什么

### 已有（这是你的优势，别重复学）

| 已具备 | 在哪 | 对学习的作用 |
|---|---|---|
| 手写 LLM 流式客户端 | `datawhale_learn/llms/ollama_client.py`（`think()`） | 你懂"一次模型调用"长什么样，不会被框架抽象骗 |
| 手写 Plan-and-Execute | `datawhale_learn/agents/planner_agent.py` | 对应 DeepAgents 的 `write_todos`，你已知其原理 |
| 手写 ReAct 循环 | `datawhale_learn/agents/react_agent.py`（`max_steps=5`，`Thought→Action→Observation`） | **关键**：`create_agent` 就是这个循环的产品化版本 |
| 手写 Reflection | `datawhale_learn/agents/reflection_agent.py` | 对应后续的 self-critique / 子代理评审模式 |
| 手写工具注册 + 记忆 | `tools/tool_base.py`、`memorys/memory.py` | 对应 LangChain 的 `@tool` 与 checkpointer |
| 多智能体协作经验 | `autogen_learn/`、`agentscope_learn/`、`camel_learn/` | 已理解"角色分工/消息传递"，学 SubAgent 会很快 |
| **LangGraph 已跑通一个工作流** | `langgraph_learn/ask_answer.py`（搜索问答，状态图 + 检查点，`langgraph 1.2.12`） | 已经跨过 LangGraph 最难的第一道坎 |
| OpenAI 兼容环境已配好 | `.env`：`LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL` / `LLM_TIMEOUT` | **所有练习可直接复用，不用重配** |

### 还缺的（本方案要补的四块）

1. **LangChain 的 `create_agent` 抽象**：你没用过 `create_agent`，只装过 `langchain-openai`。
2. **Middleware 机制**：这是 LangChain 的灵魂，也是读懂 DeepAgents 的唯一钥匙。
3. **LangGraph 的可靠性三件套**：`checkpointer` 深用（多轮/时间旅行）、`interrupt`（人工审批）、`durable execution`（崩溃续跑）。你现在的 `ask_answer.py` 只是浅用。
4. **DeepAgents 的 harness 思维**：什么是"预装能力"，`graph.py` 里那 10 层中间件各管什么。

### 因此：跳过什么（明确不做的事）

- ❌ 不再学"什么是 LLM / token / prompt engineering 基础"
- ❌ 不再学"ReAct 原理"（你手写过）
- ❌ 不学 RAG / 向量库（与本主题无关，别被文档导航带跑）
- ❌ 不学 LangSmith Deployment / 云部署（等能写复杂图了再说）

---

## 二、学习目标（可验收，不说"了解/熟悉"）

学完这个方案，你应该能做到：

1. 不看文档，手写出 `create_agent(model, tools, system_prompt, checkpointer)` 并解释每个参数**在 LangGraph 层对应什么**。
2. 写一个自定义 `AgentMiddleware`，在模型调用前动态改 system prompt、在工具调用前加权限判断。
3. 用 LangGraph Graph API 手画出一个带 `tool_node` 的 Agent，并挂上 checkpointer 实现多轮记忆 + `interrupt` 实现人工审批。
4. 读 `deepagents/graph.py` 能说出那条中间件栈的顺序与每一层的作用，并能解释 `DeltaChannel` 和 `recursion_limit=9999` 为什么存在。
5. 用 `create_deep_agent` 改造一个真实任务（读写文件 + 子代理），并与 `create_agent` 版本做**可量化的对照实验**。

---

## 三、三阶段方案（推荐 3 周 / 每天 1.5 小时；赶时间可压成 9 天）

> 每天结构固定：**读 20 分钟 → 写 50 分钟 → 记 20 分钟**（笔记进 `docs/langchain-family/notes/`）
> 铁律：**每个阶段必须有能跑的代码 + 一个"破坏性实验"**（故意改坏，看报错，理解边界）。

---

### 阶段一（第 1 周）：LangChain —— 把"抽象"落到实处

**认知主线**：`create_agent` = 你那个 `ReactAgent` + 中间件插槽 + LangGraph 运行时。

| Day | 任务 | 读什么 | 写什么 |
|---|---|---|---|
| 1 | 跑通第一个 Agent | Quickstart | `langchain_learn/01_hello_agent.py` |
| 2 | 消息与工具 | Messages / Tools | `02_message_anatomy.py`：打印一次完整 tool-call 往返的 4 条消息 |
| 3 | 模型与结构化输出 | Models / Structured output | `03_structured.py`：让模型返回 pydantic 对象 |
| 4 | **记忆** | short-term-memory | `04_memory.py`：同 thread_id 第二轮能记住；换 thread_id 就忘 |
| 5 | **Middleware（重点）** | Middleware / Philosophy | `05_middleware.py`：写一个 middleware，给 system prompt 追加"必须用中文回答" |
| 6 | 自定义工具实战 | Tools / Runtime | 把 `datawhale_learn/tools/duck_search.py` 包成 `@tool`（注意 `[推断]`：它的 `proxy` 硬编码 `127.0.0.1:7897`，需改） |
| 7 | 复盘 | 通读 docs Map | 笔记：`01-langchain.md`，画出"一次 invoke 的时序图" |

**验收（周自检）**
- [ ] 能不看代码，徒手写出 `create_agent` 五参数版本
- [ ] 能说清 `AIMessage.tool_calls` → `ToolMessage` → 再进模型的完整链条
- [ ] 能写一个改变模型行为的 middleware，并解释它在循环的哪个位置生效
- [ ] `04_memory.py` 能演示"同 thread 记得、换 thread 忘"

**常见坑**
- 模型名必须能被你 `.env` 的端点识别；报 `model not found` 就先用 `LLM_MODEL_ID` 原值
- 第三方 OpenAI 兼容端点**可能不支持 function calling** → 表现为 `tool_calls` 永远为空，此时换 `gpt-4o-mini` / `qwen-plus` 等支持工具调用的模型验证
- 忘记给 `config={"configurable":{"thread_id":...}}` → checkpointer 不报错但完全不起作用（静默失败，最坑）

---

### 阶段二（第 2 周）：LangGraph —— 从"会用"到"能控"

**认知主线**：LangGraph 的价值不是 Agent，而是**状态机的可靠性**（持久化、恢复、审批）。

| Day | 任务 | 读什么 | 写什么 |
|---|---|---|---|
| 8 | 两版对照 | Quickstart（Graph API + Functional API 一起读） | `langgraph_learn/02_api_compare.py`：同一任务两种写法 |
| 9 | **State 与 Reducer** | Graph API overview | 故意把 `Annotated[list, add_messages]` 改成 `list`，观察消息被覆盖 |
| 10 | **Checkpoint 深用** | Persistence | 打印 `graph.get_state(config)`；用 `get_state_history` 做**时间旅行** |
| 11 | **Interrupt / HITL（重点）** | Interrupts | `03_human_approval.py`：执行危险工具前暂停，人工输入 approve 后继续 |
| 12 | **Durable execution** | Durable execution | 中途 `Ctrl+C`，再从同 thread_id 续跑，验证状态没丢 |
| 13 | 子图与组合 | Subgraphs | 把你的搜索流程做成子图，被主图调用 |
| 14 | 复盘 | 通读 docs Map | 笔记：`02-langgraph.md`，画"Pregel 超步"示意 |

**验收**
- [ ] 能解释 `compile(checkpointer=...)` 之后，状态到底存在哪、以什么粒度存
- [ ] 能写出"暂停 → 人工修改 state → 恢复"的完整用例
- [ ] 能解释为什么 reducer 写错会导致"消息被吃掉了"
- [ ] 能说出 Pregel / Channel 两个词在 LangGraph 里指什么（**这是 debug 的分水岭**）

**常见坑**
- `interrupt` 后必须用 `Command(resume=...)` 恢复，不是再 `invoke` 一次
- 节点内部用 `print` 调试会打乱流式输出，改用 LangSmith 或 `stream_mode="debug"`

---

### 阶段三（第 3 周）：DeepAgents —— 读懂"预装能力的组合艺术"

**认知主线**：DeepAgents 没有任何黑魔法，它就是**一条精心排序的 middleware 栈**。

| Day | 任务 | 读什么 | 写什么 |
|---|---|---|---|
| 15 | 跑通 deep agent | Overview / Quickstart | `deepagents_learn/01_hello.py`（**必须显式传 model**，见下方坑） |
| 16 | **读源码** | `libs/deepagents/deepagents/graph.py` | 精读 `create_deep_agent`：签名 → 装配 → `create_agent(...).with_config(...)` |
| 17 | 虚拟文件系统 | Harness / Backends | 复现官方实验：给它一个大文件，看它如何用 `grep` 而不是"心算" |
| 18 | 任务规划 | Harness（Task planning） | 观察 `write_todos` 的调用时机；对比你手写的 `Planner.plan()` |
| 19 | 子代理 | Subagents | 定义一个专属子代理（如"数学审查员"），验证主代理何时派活 |
| 20 | 记忆与权限 | Memory / Permissions | 加 `AGENTS.md` 进 system prompt；加一条 `deny` 规则验证被拦 |
| 21 | **三版对照实验（毕业项目）** | comparing-deep-agents-to-other-approaches | 见下方里程碑 |

**验收**
- [ ] 能默写中间件栈的**前 6 层顺序**并说出每层干什么
- [ ] 能解释 `DeltaChannel` 解决的是"checkpoint 体积 O(N²)"问题
- [ ] 能解释 `recursion_limit=9999` 与"deep"的关系
- [ ] 能改出一个"去掉 `write_todos`"的变体，并说明它对任务完成率的影响

**常见坑（我已从源码中确认的两条）**
1. `model=None` **已弃用**，源码里默认构造 `ChatAnthropic(model_name="claude-sonnet-4-6")` 并打 deprecation warning → **你是 OpenAI 兼容环境，必须显式传 model 对象**，否则会去找 `ANTHROPIC_API_KEY` 而报错。
2. `execute`（跑 shell）工具**只有后端实现 `SandboxBackendProtocol` 时才可用**，默认 `StateBackend` 下它会直接返回错误信息 —— 这不是 bug。

---

## 四、里程碑项目：三版对照（这才是"学懂"的证据）

复用你**已有的** `langgraph_learn/ask_answer.py`（搜索问答工作流），做成三份实现，统一同一批 5 个测试问题，产出一份对照报告。

| 版本 | 实现方式 | 观察什么 |
|---|---|---|
| V1 | `create_agent(tools=[search])` | 需要多少行？失败时怎么办？ |
| V2 | 你现有的 LangGraph 手写图（+checkpointer +interrupt） | 可控性换来了什么？代码量代价？ |
| V3 | `create_deep_agent(tools=[search])` | 它自己加的 `write_todos`/`grep` 有没有让结果更好？ |

**报告必须包含**（写进 `docs/langchain-family/03-three-way-comparison.md`）：

1. 代码行数对比
2. 每个问题的回答质量（是否编造、是否可验证）
3. token 消耗与轮次（用 LangSmith 或手动统计）
4. "同一问题，三者分别在哪一步走偏"
5. 一句话结论：**什么场景下我愿意为 DA 多付的那些 token 买单**

> 这个项目同时是你简历上能讲的东西：不是"学过三个框架"，而是"我测过它们在 X 任务上的权衡"。

---

## 五、时间与资源预算

| 项 | 预算 | 说明 |
|---|---|---|
| 总时长 | 3 周 × 每天 1.5h ≈ 31h | 压缩版：9 天 × 3h |
| token 成本 | 约 ¥20–60 `[推断]` | 主要花在阶段三的长任务；**建议用便宜模型做阶段一/二，阶段三再上好模型** |
| 新装依赖 | `langchain`、`deepagents`（会带入 `langchain-anthropic`） | 复用现有 `.venv` 与 `.env`，不重建环境 |
| 必装的可观测性 | LangSmith（`LANGSMITH_TRACING=true`） | 阶段二第 10 天之后**强烈建议开**，否则调图全靠猜 |

---

## 六、风险与对策

| 风险 | 表现 | 对策 |
|---|---|---|
| 版本漂移 | 文档 API 与你装的版本不符 | 每阶段开头先 `pip index versions` 对齐；报错优先搜 `reference.langchain.com` |
| 端点不支持 tool calling | `tool_calls` 恒为空 | 换模型验证；这是**环境问题不是代码问题**，别死磕 |
| 代理依赖 | 搜索工具需 `127.0.0.1:7897` | 现有代码已依赖该代理；不可用时改用不联网的本地文件工具做练习 |
| 上下文爆炸 | 阶段三长任务报 context 超限 | 正是要学的：观察 DA 的 summarization middleware 怎么救场 |
| 只读不写 | 看完文档感觉"懂了" | 每个 Day 的"破坏性实验"是强制项，不做等于没学 |

---

## 七、每日自检清单（贴在显示器上）

- [ ] 今天的代码能跑通，且**我能预测它的输出**
- [ ] 我至少故意改坏了一次，并看懂了报错
- [ ] 我能用一句话向外行解释今天学的概念
- [ ] 笔记里记下了"我原本以为…实际上…"的一条认知修正
- [ ] 没有留下"以后再看"的悬念（有就当场查 5 分钟，查不到记进 TODO 但标注原因）
