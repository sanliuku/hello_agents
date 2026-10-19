# 第一步计划（Day 1 · 约 2 小时）

> 阶段一第 1 天。目标不是"学完 LangChain"，而是**亲手跑通一次完整的工具调用闭环，并意识到它就是你手写的 ReActAgent**。

---

## 0. 本步要达成什么（一句话）

用你**现有的** `.env` 和 `.venv`，跑起 `create_agent`，看着模型自己决定调用 `get_weather`，并把中间那 4 条消息打印出来。

**完成标志**：屏幕上出现 `AIMessage(tool_calls=[...])` 和 `ToolMessage(...)`，且最终答案是"sunny in San Francisco"。

---

## 1. 环境检查（10 分钟）

在项目根目录（`F:\wbb_project\github_lib\hello_agents`）打开终端，激活已有环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

确认现状（你的 README 记录是 Python 3.11.5、langgraph 1.2.12）：

```powershell
python -V
python -c "import langgraph, langchain_openai; print('langgraph', langgraph.__version__)"
python -c "import langchain" ; echo "exit=$LASTEXITCODE"
```

- 如果 `import langchain` 报 `ModuleNotFoundError` → 这就是今天的第一个任务，安装：

```powershell
pip install -U langchain
```

- 确认 `.env` 里四个变量存在（**不要**把真实 key 贴到聊天里）：

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print({k: ('已设置' if os.getenv(k) else '缺失') for k in ['LLM_API_KEY','LLM_MODEL_ID','LLM_BASE_URL']})"
```

> 校验点：三项都是"已设置"。若 `LLM_MODEL_ID` 缺失，用你平时能跑的模型名填上（例如 `gpt-4o-mini`）。

---

## 2. 建目录 + 写第一个 Agent（30 分钟）

新建 `langchain_learn/`（和 `langgraph_learn/` 平级），创建 `01_hello_agent.py`：

```python
"""Day 1: 第一个 LangChain Agent —— 观察 ReAct 闭环的四条消息。

运行（项目根目录）:
    python -m langchain_learn.01_hello_agent
或:
    python langchain_learn/01_hello_agent.py
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

# 关键：你的 .env 是 OpenAI 兼容端点，所以显式构造模型，而不是写 "openai:gpt-5.5"
model = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
    timeout=int(os.getenv("LLM_TIMEOUT", "60")),
)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。当用户询问天气时必须调用此工具。"""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. 用中文回答。",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

# ---- 观察区：把完整消息序列打印出来 ----
print("=" * 60)
for i, msg in enumerate(result["messages"]):
    print(f"\n[{i}] {type(msg).__name__}")
    if getattr(msg, "tool_calls", None):
        print("    tool_calls:", msg.tool_calls)
    print("    content:", (msg.content or "")[:200])

print("\n最终回答：", result["messages"][-1].content_blocks)
```

**这段代码里每一行的来历**（别当模板抄，要能对上号）：

| 代码 | 对应你今天要理解的机制 |
|---|---|
| `@tool` + docstring | 装饰器把**函数名、描述、参数类型**转成模型的 tools schema。描述写不好，模型就不调它 |
| `create_agent(model, tools, system_prompt)` | 组装"模型 ↔ 工具"循环，内部其实是一个 LangGraph 图 |
| `invoke({"messages":[...]})` | 入口约定：永远是消息列表 |
| `result["messages"]` | 返回的不是"一句话"，而是**整条对话轨迹**（这就是你能观察的原因） |
| `content_blocks` | 官方推荐的新式内容读取方式（替代裸 `content` 的字符串假设） |

---

## 3. 运行（10 分钟）

```powershell
python langchain_learn/01_hello_agent.py
```

**期望输出结构**（4 条消息，顺序固定）：

```
[0] HumanMessage        content: What's the weather in San Francisco?
[1] AIMessage           tool_calls: [{'name':'get_weather','args':{'city':'San Francisco'},'id':...}]
[2] ToolMessage         content: It's always sunny in San Francisco!
[3] AIMessage          content: 旧金山现在天气晴朗…
```

看到 `[1]` 有 `tool_calls`、`[2]` 是 `ToolMessage` —— **这一步就成功了**。你把 `datawhale_learn/agents/react_agent.py` 里那个 `Thought→Action→Observation` 手工循环，用工业级方式复现了一遍。

---

## 4. 三个破坏性实验（30 分钟，最重要的一步）

> 不做这一步，你今天只是"跑通了个 demo"；做完，你才理解"工具描述即 prompt"。

### 实验 A：删掉 docstring
把 `get_weather` 的 `"""查询指定城市的天气…"""` 删掉，重跑。
**观察**：模型还会调它吗？（很可能不调，直接回答，或调错）
**结论**：工具的 docstring **不是注释，是 prompt 的一部分**。

### 实验 B：把描述改得含糊
改成 `"""一个函数。"""`，重跑。
**观察**：调用率下降 / 参数传错（比如 `city` 传成 "weather"）。
**结论**：描述要写"什么时候该用它"，而不只是"它是什么"。

### 实验 C：给两个功能重叠的工具
再加一个 `get_weather_by_coords(lat, lon)`，两者都注册，问"旧金山天气"。
**观察**：它选了哪个？为什么？把 `tools` 顺序换一下再试。
**结论**：模型在工具选择上有隐式偏好，这是"工具过多导致选错"的根源 —— 直接对应 DeepAgents 为什么要用**子代理隔离工具集**。

把三次实验结果记进笔记（下一步）。

---

## 5. 写笔记（20 分钟）

新建 `docs/langchain-family/notes/01-langchain-hello.md`，模板：

```markdown
# Day 1 笔记 · 第一个 LangChain Agent

## 我跑通的
- 命令：
- 输出（贴关键 10 行）：

## 认知修正（必填，至少 1 条）
- 我原本以为：
- 实际上是：

## 三个破坏性实验的结论
- A 删 docstring →
- B 含糊描述 →
- C 双工具冲突 →

## 与 datawhale_learn 的对照
- 我的 ReactAgent 用 ___ 行实现循环，create_agent 用 ___ 行
- 我缺的能力：middleware 插槽 / checkpoint / 流式 / 结构化输出 中，我真正缺的是 ___

## 待查（带原因的 TODO）
- [ ] 问题 —— 为什么现在不查：
```

---

## 6. 验收清单（必须全打勾才进 Day 2）

- [ ] 我**能预测**程序输出的 4 条消息类型与顺序（合上代码也能说出来）
- [ ] 我看到了 `tool_calls`，并知道它是 `AIMessage` 上的字段
- [ ] 三个破坏性实验都做了，且能说出"为什么"
- [ ] 我能用一句话解释 `@tool` 的 docstring 去了哪里（→ 进了模型 prompt）
- [ ] 笔记写完了，含至少 1 条"认知修正"
- [ ] 没有遗留的"以后再看"悬念（有则记进 TODO 并写明原因）

---

## 7. 排错表

| 现象 | 原因 | 处理 |
|---|---|---|
| `model not found` / `401` | `.env` 的 key 或 base_url 与模型名不匹配 | 打印三个变量核对；确认 base_url 以 `/v1` 结尾 |
| 日志里 `tool_calls` 一直是空 | 该模型/端点不支持 function calling | 换 `gpt-4o-mini`、`qwen-plus` 等支持工具调用的模型验证；**这是环境问题，别改代码** |
| `ValueError: No LLM found for...` | 用了 `init_chat_model("openai:gpt-5.5")` 但账号无该模型 | 改用 `ChatOpenAI(model=os.getenv("LLM_MODEL_ID"), ...)` |
| 中文乱码 | Windows 控制台编码 | `chcp 65001` 或 `$env:PYTHONIOENCODING="utf-8"` |
| `ModuleNotFoundError: dotenv` | 缺 `python-dotenv` | `pip install python-dotenv` |
| 运行很慢 | 端点在境外/走代理 | 检查是否需要 `127.0.0.1:7897` 代理（你项目里已有这个约定） |

---

## 8. 做完之后，Day 2 会自然接到哪

Day 2 是 **Messages / Tools 深入**：把今天看到的 4 条消息拆到字段级（`tool_call_id` 为什么必须存在、`content_blocks` 与 `content` 的区别），并给工具加**错误返回**与**多轮调用**。

如果你今天想多走一步，可选加餐（30 分钟）：给 `create_agent` 传 `checkpointer=InMemorySaver()`，用同一个 `thread_id` 连续 invoke 两次，问"我刚才问了什么城市？" —— 这会把 Day 4 的记忆课提前点亮，且能亲眼验证"上下文真的是被存起来的"。
