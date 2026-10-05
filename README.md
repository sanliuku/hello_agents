# hello_agents

AI Agent 学习与实践项目：一边跟随《Hello Agents》教程从零实现 Agent 的核心组件，一边上手主流多智能体框架。目标是通过动手实现理解 LLM 调用、Planner、ReAct、Reflection、记忆、工具使用与多智能体协作。

## 目录结构

```
hello_agents/
├── datawhale_learn/     # 从零实现 Agent 核心组件（不依赖框架）
├── autogen_learn/       # Microsoft AutoGen：软件开发团队多智能体协作
├── agentscope_learn/    # AgentScope v2：三国狼人杀多智能体游戏
├── camel_learn/         # CAMEL：角色扮演式双智能体协作写作
├── langgraph_learn/     # LangGraph：搜索问答工作流（状态图 + 检查点）
├── docs/                # 学习笔记（空目录，占位）
├── .env                 # LLM 配置，已在 .gitignore 中
├── .vscode/             # 解释器与调试配置
└── README.md
```

## 环境配置

### 1. 虚拟环境与依赖

实测环境：Python 3.11.5（`.venv`）、agentscope 2.0.8、camel-ai 0.2.90、langgraph 1.2.12、autogen-\* 0.7.5、ddgs 9.16.0。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openai python-dotenv ddgs autogen-agentchat "autogen-ext[openai]" agentscope camel-ai colorama langgraph langchain-openai
```

- `autogen-agentchat` 只依赖 `autogen-core`，**不会**自动带入 `autogen-ext`，而 `autogen_learn/llms/llm_client.py` 用到 `autogen_ext.models.openai`，所以必须显式安装 `autogen-ext`。
- 搜索工具用的是 `ddgs` 包（不是 `duckduckgo_search`）。
- 项目暂未提供 `requirements.txt`。

### 2. `.env`

在项目根目录创建 `.env`（已被 `.gitignore` 忽略，不要提交真实密钥）：

```env
LLM_API_KEY=你的密钥
LLM_MODEL_ID=模型名称
LLM_BASE_URL=服务地址
# 可选，请求超时秒数
LLM_TIMEOUT=60
```

各模块读取的变量与默认值：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `LLM_API_KEY` | OpenAI 兼容服务密钥 | 无；`datawhale_learn` 缺失会抛 `ValueError` |
| `LLM_MODEL_ID` | 模型名 | 无；autogen 默认 `gpt-4o`，agentscope 默认 `qwen2.5`，langgraph 默认 `gpt-4o-mini` |
| `LLM_BASE_URL` | 服务地址（OpenAI 兼容接口） | autogen / langgraph 默认 `https://api.openai.com/v1`，agentscope 默认 `http://localhost:11434/v1` |
| `LLM_TIMEOUT` | 请求超时秒数 | `60`（`ollama_client.py`、`book_write.py`） |

各模块都用 `load_dotenv()` 从脚本所在目录向上查找 `.env`，在项目内运行即可读到根配置。

### 3. 其他前置条件

- **LLM 服务**：所有脚本都需要可用的 OpenAI 兼容服务与 API Key（云端或本地 Ollama 均可）。
- **联网搜索**：`test_search` / `test_react` / `langgraph_learn` 需要访问 DuckDuckGo；搜索默认走本地代理 `http://127.0.0.1:7897`（`datawhale_learn/tools/duck_search.py` 的 `proxy` 参数、`langgraph_learn/ask_answer.py` 中硬编码），没有该代理需自行修改代码。
- **agentscope_learn**：默认连本地 Ollama，启动前会硬检查 `localhost:11434`，即使 `LLM_BASE_URL` 指向云端也会提示 `ollama pull qwen2.5` 后退出。

## 模块说明

### datawhale_learn — 从零实现 Agent 基础组件

不依赖 Agent 框架，直接用 `openai` / `ddgs` 等基础库搭建：

- `llms/ollama_client.py` — `HelloAgentsLLM`：兼容 OpenAI 接口的流式 LLM 客户端，`model / apiKey / baseUrl / timeout` 缺省时从 `.env` 读取，对外提供 `think(messages, temperature=0)`
- `agents/planner_agent.py` — Plan-and-Execute：`Planner.plan()` 拆解计划、`Executor.execute()` 逐步执行、`PlanAndSolveAgent.run()` 编排
- `agents/react_agent.py` — `ReActAgent(llm_client, tool_executor, max_steps=5)`：Thought → Action → Observation 循环，以 `Finish[答案]` 结束
- `agents/reflection_agent.py` — `ReflectionAgent(llm_client, max_iterations=3)`：执行 → 反思 → 改进迭代，反馈中出现「无需改进」时提前结束
- `memorys/memory.py` — `Memory`：短期记忆，`add_record(type, content)` / `get_trajectory()` / `get_last_execution()`
- `tools/tool_base.py` — `ToolExecutor`：`registerTool()` / `getTool()` / `getAvailableTools()` 工具注册与执行
- `tools/duck_search.py` — `search(query, max_results=5, region="wt-wt", timeout=10, proxy=...)`：DuckDuckGo 搜索

练习脚本内部使用 `datawhale_learn.*` 绝对导入，**必须在项目根目录以模块方式运行**：

```bash
python -m datawhale_learn.test_llm          # 手写 LLM 客户端演示
python -m datawhale_learn.test_search       # 工具注册 + DuckDuckGo 搜索
python -m datawhale_learn.test_plan         # 计划执行（Plan-and-Execute）
python -m datawhale_learn.test_react        # ReAct
python -m datawhale_learn.test_refleactoin  # Reflection（文件名拼写如此）
```

其中 `test_plan.py` / `test_react.py` / `test_refleactoin.py` 没有 `__main__` 保护，导入即执行。

### autogen_learn — AutoGen 软件开发团队

- `llms/llm_client.py` — `create_openai_model_client()`：`OpenAIChatCompletionClient`，读取 `.env`；`model_info` 固定为 `family="ollama"`、`context_length=32768` 等（为了兼容非官方模型）
- `roles/` — 四个角色：`create_product_manager()` / `create_engineer()` / `create_code_reviewer()`（`AssistantAgent`）与 `create_user_proxy()`（`UserProxyAgent`）
- `teams/code_team.py` — `software_development_team(llm_client)`：`RoundRobinGroupChat` 轮询协作，`TextMentionTermination("TERMINATE")`，`max_turns=20`
- `test_software_development_team.py` — 端到端演示：协作开发一个「美国大米价格」Streamlit 应用，用 `Console` 流式输出对话

运行注意：`test_software_development_team.py` 用 `autogen_learn.*` 绝对导入，而 `teams/code_team.py` 用 `roles.*` 裸导入，需要让两份路径同时可见：

```powershell
$env:PYTHONPATH = "autogen_learn"
python -m autogen_learn.test_software_development_team
```

更干净的做法是把 `autogen_learn/teams/code_team.py` 中的 `from roles.xxx import ...` 改为 `from autogen_learn.roles.xxx import ...`，之后在根目录直接 `python -m autogen_learn.test_software_development_team` 即可。

### agentscope_learn — 三国狼人杀（AgentScope v2）

用刘备、关羽、张飞、诸葛亮、赵云、曹操、司马懿、周瑜、孙权 9 名三国角色玩狼人杀，默认 9 人局。

- `roles.py` — `GameRoles`：狼人 / 预言家 / 女巫 / 猎人 / 村民 / 守护者的技能、阵营与胜负条件，6 / 8 / 9 人标准配置，以及三国人物的性格设定
- `prompt.py` — `ChinesePrompts`：中文角色提示词（`get_role_prompt(role, character)`）
- `outputs.py` — 结构化输出模型（Pydantic）：`DiscussionModelCN`、`WerewolfKillModelCN`、`WitchActionModelCN`、`GameAnalysisModelCN`，以及 `get_vote_model_cn()` / `get_seer_model_cn()` / `get_hunter_model_cn()` 动态生成候选枚举
- `utils.py` — `GameModerator`（法官播报）、`majority_vote_cn()`、`check_winning_cn()`、`format_player_list()` 等；`MAX_GAME_ROUND = 10`、`MAX_DISCUSSION_ROUND = 3`
- `test_game.py` — `ThreeKingdomsWerewolfGame`：夜晚（狼人讨论+击杀、预言家查验、女巫解药/毒药）与白天（自由讨论 + 投票、猎人开枪）主循环；`__init__` 默认 `player_count=6`，`main()` 中传 9

AgentScope v2 与 v1 的差异（`test_game.py` 文件头注释有记录）：`sys_prompt` → `system_prompt`；改用 `await agent.reply(...)`，结构化字段 `structured_model` → `structured_schema`，结果由 `msg.metadata` 变为 `msg.structured_output`；`MsgHub` / `fanout_pipeline` / `sequential_pipeline` 已移除，改为 `reply + observe` 手动广播、`asyncio.gather` 并行投票。

运行（脚本内部是裸导入，**不能**用 `-m`，在项目根目录按脚本路径运行；模型走本地 Ollama）：

```bash
python agentscope_learn/test_game.py
```

### camel_learn — CAMEL 角色扮演协作

- `book_write.py` — CAMEL `RolePlaying` 双角色协作：`assistant_role_name="心理学家"` × `user_role_name="作家"`，协作撰写 1000 字《拖延症心理学》短篇电子书；`ModelFactory.create(model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL, ...)` 接入任意 OpenAI 兼容服务，`temperature=0.7`；最多 30 轮，出现 `CAMEL_TASK_DONE` 结束

```bash
python camel_learn/book_write.py
```

### langgraph_learn — LangGraph 搜索问答工作流

- `ask_answer.py` — `SearchState`（`messages` / `user_query` / `search_query` / `search_results` / `final_answer` / `step`）+ 三个节点 `understand_query_node` → `search_node` → `generate_answer_node`，用 `StateGraph` 线性编排并配 `InMemorySaver` 检查点；交互式循环（`quit` / `q` / `退出` / `exit` 结束，每轮一个 `thread_id`）；搜索失败时降级为「基于模型已有知识回答」。模型用 `ChatOpenAI`，搜索用 `ddgs`（文件头注释里的 Tavily 已不再使用）

```bash
python langgraph_learn/ask_answer.py
```

## 命令速查

| 模块 | 运行方式 |
| --- | --- |
| datawhale_learn | 项目根目录下 `python -m datawhale_learn.test_react`（另有 4 个 `test_*`） |
| autogen_learn | `$env:PYTHONPATH = "autogen_learn"` 后 `python -m autogen_learn.test_software_development_team` |
| agentscope_learn | `python agentscope_learn/test_game.py` |
| camel_learn | `python camel_learn/book_write.py` |
| langgraph_learn | `python langgraph_learn/ask_answer.py` |

## 已知问题 / 待办

- `datawhale_learn/test_refleactoin.py` 文件名拼写错误（应为 `test_reflection.py`）
- `datawhale_learn/test_llm.py` 重复实现了一份 `HelloAgentsLLM`（与 `llms/ollama_client.py` 相同），仅作演示
- `autogen_learn/llms/llm_client.py:30` 的分支判断是死代码（`model_info` 恒为非空，后续分支不会执行）
- 根目录 `__pycache__/` 中留有旧的扁平布局 pyc，可删除
- `.gitignore` 只忽略 `.env` 与 `.workbuddy/`，建议补上 `.venv/` 与 `__pycache__/`
- `docs/` 目录仍为空