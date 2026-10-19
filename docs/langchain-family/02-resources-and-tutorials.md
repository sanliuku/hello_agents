# 学习资源 / 教程 / 源码索引

> 优先级：⭐ = 必读，○ = 选读，△ = 参考备用
> 抓取时间：2026-02。链接失效时优先去 `docs.langchain.com` 根目录或 GitHub 仓库首页找新入口。

---

## 0. 三个"省时间"的技巧（先看这个）

### 技巧 1：给 AI 装官方文档 MCP（强烈推荐）
把官方文档直接接进你的编码 Agent（Claude Code / Cursor / Codex / Deep Agents Code 等）：

- `docs-langchain` → `https://docs.langchain.com/mcp`
- `reference-langchain` → `https://reference.langchain.com/mcp`

好处：不用自己扒网页，Agent 能查到最新 API。官方配置说明在 `https://docs.langchain.com/use-these-docs.md` `[文档]`

### 技巧 2：一切文档页都有 `.md` 原始版
`docs.langchain.com` 用 Mintlify 托管，任意页面 URL 后加 `.md` 就能拿到**干净 Markdown**（没有导航噪音），非常适合喂给模型或存进仓库：

```
https://docs.langchain.com/oss/python/langchain/quickstart.md
https://docs.langchain.com/oss/python/langgraph/quickstart.md
```

### 技巧 3：整站索引在 `llms.txt`
`https://docs.langchain.com/llms.txt` —— 官方给出的"所有页面清单"，用来发现你没听过的页面（如 `philosophy`、`middleware`、`harness`）比猜 URL 靠谱 `[文档]`

---

## 1. LangChain

### ⭐ 官方文档（Python）
| 页面 | 路径 | 为什么读 |
|---|---|---|
| Quickstart | `/oss/python/langchain/quickstart` | 第一个可跑 Agent，含"真实世界 Agent"进阶版 |
| Philosophy | `/oss/python/langchain/philosophy` | 讲清设计取舍，**理解 Middleware 的前提** |
| **LangChain vs LangGraph vs Deep Agents** | `/oss/python/langchain/` 首页对比页 | 官方版"三者分工"，建议在动手前读 |
| Agents | `/oss/python/langchain/agents` | `create_agent` 全部参数与行为 |
| Models | `/oss/python/langchain/models` | `init_chat_model`、`provider:model` 字符串规则 |
| Messages | `/oss/python/langchain/messages` | 四种消息 + `content_blocks` + `tool_calls` |
| Tools | `/oss/python/langchain/tools` | `@tool`、描述如何影响调用、`ToolRuntime` 注入 |
| Middleware | `/oss/python/langchain/middleware` | **核心**：在 Agent 循环哪里插钩子 |
| Structured output | `/oss/python/langchain/structured-output` | 让模型返回 pydantic 对象 |
| Memory | `/oss/python/langchain/short-term-memory` | checkpointer + thread_id |
| Streaming | `/oss/python/langchain/streaming` | token / 步骤级流式 |
| Runtime (Context) | `/oss/python/langchain/runtime` | 运行时注入依赖，别用全局变量 |
| Testing | `/oss/python/langchain/test` | Agent 怎么测（很多人跳过，面试会问） |

### ⭐ 源码
- 主仓库：`https://github.com/langchain-ai/langchain`（monorepo，`libs/` 下分包）
- 关键分包：`libs/langchain`（高层 API，`create_agent` 在这里）、`libs/core`（Runnable/Message/Tool 基类）、`libs/partners/*`（各模型厂商适配）
- **读源码入口建议**：先看 `libs/langchain/langchain/agents/factory.py`（`create_agent` 的装配逻辑），你会发现它就是"拼一个 LangGraph"

---

## 2. LangGraph

### ⭐ 官方文档
| 页面 | 路径 | 为什么读 |
|---|---|---|
| Quickstart | `/oss/python/langgraph/quickstart` | 同一道题给 **Graph API + Functional API 两版**，对比着看理解最快 |
| Graph API overview | `/oss/python/langgraph/graph-api` | State / Node / Edge / Reducer 的正式定义 |
| Functional API | `/oss/python/langgraph/functional-api` | `@entrypoint` / `@task`，写起来像普通函数 |
| **Persistence** | `/oss/python/langgraph/persistence` | Checkpointer / Store / thread_id / 命名空间 |
| **Durable execution** | `/oss/python/langgraph/durable-execution` | 崩了怎么续跑，这是它最大卖点 |
| Interrupts / HITL | `/oss/python/langgraph/interrupts` | 人工审批、修改、恢复 |
| Memory | `/oss/python/langgraph/add-memory` | 短期 vs 长期（store）的分界 |
| Subgraphs | `/oss/python/langgraph/subgraphs` | 组合复杂流程的手段 |
| Streaming | `/oss/python/langgraph/streaming` | 各种 stream_mode |
| Deploy | `/oss/python/langgraph/deploy` | LangSmith Deployment，一键托管 |

### ⭐ 源码
- 仓库：`https://github.com/langchain-ai/langgraph`（`libs/langgraph` 核心；`libs/checkpoint*` 持久化；`libs/prebuilt` 起步模板）
- **必读文件**：`libs/langgraph/langgraph/graph/state.py` → `StateGraph.compile()`，看它怎么把 nodes/edges 变成 Pregel 执行
- **必读概念**：`Pregel` / `Channel` / `add_messages` reducer —— 这三点不懂，debug 时全是黑盒

### 教程 & 课程
- ⭐ **LangChain Academy — LangGraph Essentials**：官方免费课，视频 + 可跑 notebook，**这是三个框架里官方课程最成熟的一个** `[文档]`
- 官方 Examples 仓库：`https://github.com/langchain-ai/langgraph` 的 `examples/`（含 multi-agent、RAG、reflection、human-in-the-loop 模式）
- 参考实现模式库：LangGraph "thinking in graphs" 系列文章（`docs.langchain.com` 里搜 graph 设计模式）

---

## 3. DeepAgents

### ⭐ 官方文档
| 页面 | 路径 | 为什么读 |
|---|---|---|
| Overview | `/oss/python/deepagents/overview` | 定位、什么时候用、内置能力清单 |
| Quickstart | `/oss/python/deepagents/quickstart` | 5 行起一个 deep agent |
| **Harness（核心）** | `/oss/python/deepagents/harness` | 讲 `write_todos` 规划 + 虚拟文件系统（`grep`/`read_file`）如何工作 |
| Customization | `/oss/python/deepagents/customization` | 改 prompt / middleware / tools / **prompt assembly 顺序** |
| Models | `/oss/python/deepagents/models` | 注意：`model=None` 已弃用，必须显式传 |
| Memory | `/oss/python/deepagents/memory` | `AGENTS.md` 长期记忆怎么注入 |
| Permissions | `/oss/python/deepagents/permissions` | `FilesystemPermission` 规则，deny / interrupt |
| Subagents | `/oss/python/deepagents/subagents` | 同步 / 编译 / **异步（远程）**三种子代理 |
| Backends | `/oss/python/deepagents/backends` | StateBackend（默认）/ FilesystemBackend / **SandboxBackend（解锁 `execute`）** |
| **Deep Agents vs LangChain vs LangGraph vs pydantic-ai** | `/oss/python/deepagents/comparing-deep-agents-to-other-approaches` | 官方四向对比，别自己猜 |

### ⭐ 源码
- 仓库：`https://github.com/langchain-ai/deepagents`（Python）；JS 版 `https://github.com/langchain-ai/deepagentsjs` `[文档]`
- **唯一必读文件**：`libs/deepagents/deepagents/graph.py`（约 900 行，一个函数讲完整个架构）
  - 看 4 处：`create_deep_agent` 签名 → 中间件栈装配 → `general_purpose` 子代理默认注入 → `create_agent(...).with_config({"recursion_limit": 9999})`
- 值得顺带看的：`middleware/summarization.py`（上下文压缩策略）、`middleware/filesystem.py`（虚拟 FS 的路径路由）、`middleware/subagents.py`（`task` 工具怎么调度）

### 课程
- LangChain Academy 有 **Deep Agents** 课程页，但截至抓取时官方标注"尚未正式发布/持续监控"，**别把它当必修课等**，直接读文档 + 读 `graph.py` 更快 `[文档]`

---

## 4. 跨框架资源

| 资源 | 地址 | 用途 |
|---|---|---|
| LangChain Forum | 文档站点内 `/forum` 入口 | 官方社区的疑难问答，比 GitHub issue 得到回复快 |
| LangSmith（可观测性） | `https://smith.langchain.com` | `LANGSMITH_TRACING=true` 后自动记录每一步；**调试 Agent 必用** |
| 官方 docs 仓库 | `https://github.com/langchain-ai/docs` | 文档本身开源，有 `src/oss/**/*.mdx`，比网页好 diff |
| LangChain Skills | `https://github.com/langchain-ai/langchain-skills` | `npx skills add langchain-ai/langchain-skills --skill '*'`，给编码 Agent 装框架知识 |
| Datawhale 中文教程 | Datawhale 组织下的 agent 相关开源书 | 中文入门友好；你仓库里已有 `datawhale_learn/`，可交叉印证 |

---

## 5. 本次抓取中**已验证可访问**的直达链接

```
官方文档
├─ https://docs.langchain.com/llms.txt                        (整站索引)
├─ https://docs.langchain.com/oss/python/langchain/quickstart.md
├─ https://docs.langchain.com/oss/python/langgraph/quickstart.md
├─ https://docs.langchain.com/use-these-docs.md               (MCP 配置)
├─ https://docs.langchain.com/mcp                             (docs MCP)
└─ https://reference.langchain.com/mcp                        (API reference MCP)

源码（raw 直链，可直接 curl）
├─ https://github.com/langchain-ai/langchain
├─ https://github.com/langchain-ai/langgraph
├─ https://github.com/langchain-ai/deepagents
└─ https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/deepagents/deepagents/graph.py
```

---

## 6. 如何持续跟踪版本变化

框架迭代快，**读书不如读 changelog**：

```bash
# 看本地实际装的版本
python -c "import langchain, langgraph, deepagents; print(langchain.__version__, langgraph.__version__, deepagents.__version__)"

# 看远端有哪些版本（确认文档对应的是不是你装的）
pip index versions langchain
pip index versions langgraph
pip index versions deepagents
```

> 经验法则 `[推断]`：文档和你本地版**差一个 minor 版本以上**时，先怀疑版本，再怀疑自己。报错里出现的类名/参数名，直接去 `reference.langchain.com` 搜，比翻文档快。
