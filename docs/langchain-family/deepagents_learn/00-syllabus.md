# Deep Agents 教材 · 总纲（待展开）

> 状态：**只有大纲，正文未写。**
> 对象：`deepagents` 0.x（harness，建在 LangChain `create_agent` 之上，后者又建在 LangGraph 之上）
> 姊妹教材：LangChain → `../langchain_learn/00-syllabus.md`；LangGraph → `../langgraph_learn/00-syllabus.md`

---

## 一、范围声明

本教材只管 **harness 层**：一个 agent 自己要带什么（提示词、文件系统、子代理、skills）、权限与沙箱、以及怎么上生产。

Deep Agents 自己**几乎没有新运行时**——它给的是一套默认的 middleware 栈 + 默认工具 + 默认提示词。所以：

- 不懂 `create_agent` 与 middleware 的钩子 → 先回 `../langchain_learn/` ch04 / ch09
- 不懂 checkpointer、interrupt、stream → 先回 `../langgraph_learn/` L3 / L5 / L7

**判据一句话：Deep Agents 是"配置 + 约定"，不是"新引擎"。** 想改它，改的是 middleware 栈。

---

## 二、必读一页

`docs.langchain.com/oss/python/deepagents/comparison` —— 它对比 Claude Agent SDK。

**选型比用法重要**：这里决定"到底该不该用 Deep Agents"，而不是"怎么调它的参数"。

---

## 三、6 个主题

> 不按章编号。它是参数说明书，**按需查，不要逐页精读**。

| 主题 | 官方锚点（`docs.langchain.com/oss/python/deepagents/`） | 什么场景才需要 |
|---|---|---|
| **D0** 核心与定制 | `overview`、`quickstart`、`customization`、`models`、`comparison` | 先搞清 `create_deep_agent` 到底替你塞了什么：system prompt、todo 工具、文件系统工具、子代理能力 |
| **D1** 子代理 | `subagents`、`dynamic-subagents`、`async-subagents` | 任务能拆成互不干扰的子任务，且希望主代理上下文保持干净 |
| **D2** 文件系统与沙箱 | `backends`、`permissions`、`sandboxes`、`interpreters`、`tools`、`multimodal` | 需要读写真实文件，或要把危险操作关进隔离环境 |
| **D3** 上下文管理 | `skills`、`memory`、`retrieval`、`context-engineering`、`profiles` | 长任务要控制"模型此刻看到什么"——这是 Deep Agents 存在的主要理由 |
| **D4** 审议与人工 | `human-in-the-loop`、`rubric` | 敏感工具要审批；或任务有客观标准，想让 agent 自己迭代到达标 |
| **D5** 生产化 | `going-to-production`、`fault-tolerance`、`event-streaming`、`streaming`、`middleware`、`acp`、`ag-ui` | 长期在线、要接编辑器/前端、要控成本 |

---

## 四、每个主题固定要回答的三问

1. **它替你写了什么？**（默认提示词、默认工具、默认 middleware —— 去源码里找出那段字符串/那个列表）
2. **不想要怎么拆？**（删哪个 middleware、传什么参数、换成自己的 `backend`）
3. **坏在哪？**（文件系统写爆上下文、子代理递归、沙箱逃逸、审批被绕过 —— 各写一个破坏实验）

---

## 五、与 LangChain 那本的接口

| LangChain 那本 | 在这里变成 |
|---|---|
| ch04 Tool 层 | D2 的工具全是 `@tool` + 内建 backend 约定 |
| ch05 `AgentState` | D1 子代理各有自己的 state；父只看结果 |
| ch09 Middleware 实战 | D0/D5 的全部内容就是"一套预置 middleware" |
| ch13 记忆三层 | D3 的 `memory` + `skills` |
| ch14 HITL | D4 的 `interrupt_on` 参数 |
| ch15 多智能体四形态 | D1 的 `subagents`（其中一形态的产品化版本） |

---

## 六、诚实边界

- 正文未写。锚点来自官网机器可读索引 `https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt`。
- **版本漂移最快的一个**：`deepagents` 还在 0.x，`create_deep_agent` 的参数名在 0.2 → 0.7 之间变过。动工前先 `pip index versions deepagents -i https://pypi.org/simple` 确认版本，再对着**你那个版本**的源码读，别信博客。
- 官方安全模型是"trust the LLM"，所以**边界必须落在 tool 与 sandbox 层**——D2 才是真正该下功夫的地方。
- 本机 `pwsh` 被 Windows 文件权限拦住（`grantWrite` Win32 5），未实机验证。
