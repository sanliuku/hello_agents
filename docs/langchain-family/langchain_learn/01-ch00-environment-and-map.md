# 第 0 章 · 地图与环境

> 本章目标：**搞清 `langchain` 这个 pip 包到底是什么，并让环境能跑。** 预计 40 分钟。

---

## ① 本章你会得到什么

1. 能回答"我 `pip install langchain` 装的到底是什么，它和 `langchain-core`、`langgraph` 什么关系"
2. 一个能跑的环境（**注意：你的 `.venv` 目前不存在，要重建**）
3. 一个环境自检脚本 `00_env_check.py`，以后每章开跑前先跑它

---

## ② 源码解剖：仓库里到底有几个包

GitHub 上 `langchain-ai/langchain` 的 `libs/` 目录（本次实际抓取）：

```
libs/
├── core/             → pip 包名 langchain-core     ← 基类层：Message / Tool / Runnable / ChatModel 抽象
├── langchain/        → pip 包名 langchain (v0.x)   ← 旧线，已冻结
├── langchain_v1/     → pip 包名 langchain (v1.x)   ← 新线！create_agent 在这里
├── langgraph/（在另一个仓库 langchain-ai/langgraph）→ pip 包名 langgraph  ← 图运行时
├── partners/         → pip 包名 langchain-openai / langchain-anthropic / ...
└── standard-tests/   → 给第三方模型厂商做一致性测试用（你不装）
```

**关键认知（很多人搞错）**：`langchain` 是**薄薄的一层门面**，真正的重活在两个依赖里：
- `langchain-core`：定义 `BaseMessage`、`BaseTool`、`BaseChatModel` 这些**基类**
- `langgraph`：提供 `StateGraph`、`ToolNode`、checkpoint 这些**运行时**

`libs/langchain_v1/langchain/agents/factory.py` 的 import 段就是证据：

```python
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
```

看到没——**它自己没有实现任何"Agent 循环"**，是拿 LangGraph 的 `StateGraph` 拼出来的。这解释了上一份文档的结论：层次是 `DeepAgents → LangChain → LangGraph`。

**版本判定法**（先确定你装的是哪条线）：

```python
import langchain, langchain_core, langgraph
print(langchain.__version__)        # 1.x → 读 libs/langchain_v1
from langchain.agents import create_agent   # v1 才有
```

---

## ③ 你敲这段：重建环境

### 3.1 为什么必须重建（重要）

我检查了你的工作区：

- `.venv/` **不存在**（`glob` 搜遍全仓库无 `Scripts/python.exe`、无 `pyvenv.cfg`）
- `.vscode/settings.json` 里解释器指向 `E:/project/hello_agents/.venv/Scripts/python.exe` —— **旧盘符，已失效**

所以 README 里那套"实测环境"是项目搬到 F: 盘之前的历史记录。现在得重来一遍。

### 3.2 建 venv（在项目根目录执行）

```powershell
cd F:\wbb_project\github_lib\hello_agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
```

> 期望：`Python 3.11.x` 或更高。**如果 `Activate.ps1` 报"禁止运行脚本"**，先执行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 3.3 装依赖（注意 pip 源）

```powershell
pip install -U pip -i https://pypi.org/simple
pip install -U langchain langchain-openai python-dotenv -i https://pypi.org/simple
```

**为什么要写 `-i`**：你这台机器 pip 默认镜像可能不可达（历史上踩过），显式指定官方源最稳。慢是慢点，但不会卡在 `ReadTimeout`。

> 只装 `langchain` 就够了？不够。`langchain` 不带任何厂商适配器，你必须额外装 `langchain-openai`（哪怕你连的是国内兼容端点，只要接口是 OpenAI 协议就用它）。

### 3.4 验证依赖装对了

```powershell
pip list | Select-String -Pattern "langchain|langgraph|langsmith"
```

**你应该看到**（版本号会不同，形态应该像这样）：

```
langchain            1.x.x
langchain-core       1.x.x
langchain-openai     1.x.x
langgraph            1.x.x
langgraph-checkpoint 3.x.x
langgraph-prebuilt   1.x.x
langgraph-sdk        ...
langsmith            0.x.x
```

看到 **`langgraph` 自动被装上了**——这就是 ② 说的"langchain 依赖 langgraph"。这一点很重要，后面第 1 章会用到。

### 3.5 修掉失效的 VS Code 解释器路径

把 `.vscode/settings.json` 的第一行改成工作区相对路径（这样换盘符不会再坏）：

```json
"python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
```

### 3.6 敲自检脚本

新建 `langchain_learn/00_env_check.py`（先手敲，别复制）：

```python
"""第 0 章：环境自检。跑通它，才算有资格进第 1 章。"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def line(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ---------- 1. Python 与包版本 ----------
line("1. 版本")
print("python      :", sys.version.split()[0])
for mod_name in ("langchain", "langchain_core", "langgraph"):
    try:
        mod = __import__(mod_name)
        print(f"{mod_name:12}:", getattr(mod, "__version__", "unknown"))
    except ImportError:
        print(f"{mod_name:12}: !!! 未安装")

# ---------- 2. v1 的入口是否可用 ----------
line("2. create_agent 是否可导入")
try:
    from langchain.agents import create_agent

    print("OK:", create_agent.__module__)
except Exception as exc:  # noqa: BLE001
    print("FAIL:", type(exc).__name__, exc)
    raise SystemExit("请先完成 3.2 的安装，再回来跑本脚本") from exc

# ---------- 3. .env 变量 ----------
line("3. .env 变量")
keys = ["LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL", "LLM_TIMEOUT"]
for k in keys:
    v = os.getenv(k)
    if k == "LLM_API_KEY":
        print(f"{k:12}:", "已设置（长度 %d）" % len(v) if v else "缺失")
    else:
        print(f"{k:12}:", v or "缺失（将用默认值）")

# ---------- 4. 真的调一次模型 ----------
line("4. 连通性测试（调一次真实模型）")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
    timeout=int(os.getenv("LLM_TIMEOUT", "60")),
)

try:
    reply = llm.invoke("只回复两个字：就绪")
    print("模型回复 :", reply.content)
    print("消息类型 :", type(reply).__name__)
except Exception as exc:  # noqa: BLE001
    print("FAIL:", type(exc).__name__, exc)
    raise SystemExit("模型调不通，先查 .env（key / base_url / model 三者要匹配）") from exc

# ---------- 5. 工具调用能力（决定后面能不能学） ----------
line("5. 工具调用（function calling）能力")


# 定义工具并绑定，看模型会不会把它当成工具而不是纯文本
from langchain_core.tools import tool


@tool
def ping(text: str) -> str:
    """回显传入的文本。测试用。"""
    return f"pong:{text}"


bound = llm.bind_tools([ping])
resp = bound.invoke("请调用 ping 工具，参数 text='hello'")
calls = getattr(resp, "tool_calls", None) or []
print("tool_calls :", calls if calls else "!!! 空 —— 该模型/端点可能不支持 function calling")

line("自检结果")
if calls:
    print("✅ 全部通过，可以进入第 1 章")
else:
    print("⚠️ 能对话但不会调工具。后续 8 章都会受影响，")
    print("   请换一个支持工具调用的模型（如 gpt-4o-mini / qwen-plus）再跑本脚本。")
```

### 3.7 运行

```powershell
python langchain_learn/00_env_check.py
```

---

## ④ 你应该看到

```
1. 版本
python      : 3.11.x
langchain   : 1.x.x
langchain_core: 1.x.x
langgraph   : 1.x.x

2. create_agent 是否可导入
OK: langchain.agents.factory

3. .env 变量
LLM_API_KEY : 已设置（长度 xx）
LLM_MODEL_ID: your-model
LLM_BASE_URL: https://...

4. 连通性测试
模型回复 : 就绪
消息类型 : AIMessage

5. 工具调用能力
tool_calls : [{'name': 'ping', 'args': {'text': 'hello'}, 'id': 'call_...', 'type': 'tool_call'}]

自检结果
✅ 全部通过，可以进入第 1 章
```

**注意第 2 项的输出 `langchain.agents.factory`** —— 这正是你以后要读的那个文件。教材后面所有源码引用都落在它身上。

---

## ⑤ 三个破坏性实验

### 实验 A：把 `base_url` 删掉
注释掉 `base_url=...` 那一行，重跑第 4 步。
→ 会打到 `api.openai.com`，用国内 key 必然 401。**结论**：`base_url` 决定"请求发给谁"，`model` 只是请求体里的一个字段，两者不校验一致性——这正是"模型名写错却报 401 而不是 model not found"的原因。

### 实验 B：把 `LLM_MODEL_ID` 改成不存在的名字
临时把 model 传成 `"definitely-not-a-model"`。
→ 观察报错文本：是 `404 model not found` 还是别的？**这个报错文本你要记住**，以后它会反复出现（通常是"模型名对但端点不对"或反之）。

### 实验 C：把 `api_key` 传成空串
→ 观察是否在**构造时**报错还是**调用时**报错。**结论**：LangChain 的模型对象是**惰性**的，构造不联网，第一次 `invoke` 才真正发请求。这一点影响你写单元测试的方式（第 11 章会用到）。

---

## ⑥ 自检题

1. `pip install langchain` 之后，`langgraph` 为什么也出现在 `pip list` 里？
2. 你的 `.env` 里 `LLM_MODEL_ID` 写的是 `gpt-4o-mini`，但 `LLM_BASE_URL` 指向国内某厂商端点，会发生什么？为什么？
3. 自检脚本第 5 步为什么要单独测 `bind_tools`？它和第 4 步的"能对话"差别在哪？

### 参考答案

> 先自己做，再对答案。每条后面的 **延伸** 是超出题面的部分，用来连到后面的章。

**1. `pip install langchain` 之后，`langgraph` 为什么也出现在 `pip list` 里？**

因为 `langchain` v1 的 `create_agent` 就是拿 LangGraph 的 `StateGraph` / `ToolNode` 搭出来的，`langgraph` 是它的**运行时依赖**，pip 会连带安装。这是"LangChain 搭在 LangGraph 之上"的**物理证据**，不是宣传语。

自己验证：`pip show langgraph` 看 `Required-by`；或直接打开 `python -c "import langchain.agents.factory as f; print(f.__file__)"` 那个文件，读头部的 import 段——第 1 章会把里面的 `graph.add_node` / `add_edge` 逐行讲。

> **延伸**：所以"我只想用 LangChain、不想学 LangGraph"是伪命题。图在你脚下，你迟早要看它。

**2. `.env` 里 `LLM_MODEL_ID=gpt-4o-mini`，但 `LLM_BASE_URL` 指向国内某厂商端点，会发生什么？**

三种可能，取决于那个端点：

| 结果 | 原因 |
|---|---|
| 能跑，但模型不是你以為的那个 | 该端点把 `gpt-4o-mini` 当**别名**映射到自家模型（能力和价格都变了） |
| `400 model not found` | 端点不认这个名字 |
| 能聊天、但工具永不触发 | 端点接受请求但**忽略 `tools` 字段**，永远返回纯文本 |

`model` 只是**请求体里的一个字符串参数**，认不认它由**端点**决定。所以 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` 三者必须属于同一家服务。

> **延伸**：不要靠猜，靠自检脚本第 5 步实测。第三行那种失败是最阴的——它不报错。

**3. 自检脚本第 5 步为什么要单独测 `bind_tools`？它和第 4 步"能对话"差别在哪？**

第 4 步测的是**文本通路**（请求进去、文本出来）；第 5 步测的是**函数调用通路**（请求里多了 `tools` schema、响应里多了 `tool_calls` 结构）。两条通路在协议上是不同的能力，很多国内模型和代理端点**只支持前者**。

不支持时的表现：`create_agent` 里模型永远不调工具，你不报错、不抛异常，只是"它怎么不查天气"。新手会 100% 误判成自己代码写错。

> **延伸**：这也是本教材每章都先 `dump` `tool_calls` 的原因——它是判断"链路是否健全"最便宜的信号，比读日志快。

---

## ⑦ 承上启下

环境里已经有了 `create_agent`。下一章我们干一件很"暴力"的事：**让它把内部那张图打印出来给你看**——你会亲眼看到"Agent 循环"是一条 `model → tools → model` 的回路，而不是什么魔法。

➡️ [第 1 章 · 一次 invoke 的真相](02-ch01-what-invoke-does.md)

---

## 附录 · 官方写法对照（overview 页）

> 正文是"顺着源码往下钻"。这一节反着来：**先把官方页面的最小写法抄一遍，再回头和我们钻的东西对表。**
> 官方锚点：<https://docs.langchain.com/oss/python/langchain/overview>
> 代码：`langchain_learn/00_official_agent_equation.py`（A–D 段零 token，`--real` 才联网）

### A0.1 官方那句话，就是全书的地基

> **Agent = Model + Harness.** LangChain provides `create_agent`: a minimal, highly configurable harness. The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior.

拆开看只有两个词：

| 词 | 指什么 | 教材里谁在讲 |
|---|---|---|
| **Model** | 那个会说话的东西，负责"下一步做什么" | 第 3 章 |
| **Harness** | 套在它外面的循环：prompt、tools、middleware | 第 1 / 4 / 8 章 |

所以后面所有章节，本质上都在回答一个问题：**这个 harness 的每一颗螺丝钉拧在哪。**

### A0.2 官方最小例（原样，6 行）

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

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```

三处值得停一下：

1. `model` 收**字符串**，不是对象。字符串晚绑定，所以配 `.env` 才不会把密钥写进代码。
2. `tools` 收**裸函数**，连 `@tool` 都不用加。官方在 agents 页说得很直白："pass any Python callable, LangChain tool, or tool dict"。
3. 返回值是 `dict`，答案在 `messages[-1].content_blocks`，不是 `content`。

### A0.3 `"provider:model"` 到底怎么拆

实测输出（`python langchain_learn/00_official_agent_equation.py`）：

```text
  openai:gpt-5.5                     -> provider=openai         model=gpt-5.5
  google_genai:gemini-3.6-flash      -> provider=google_genai   model=gemini-3.6-flash
  anthropic:claude-sonnet-4-5        -> provider=anthropic      model=claude-sonnet-4-5
  ollama:qwen3:latest                -> provider=ollama         model=qwen3:latest
```

**最后一行是个真坑**：ollama 的模型名自己带冒号（`qwen3:latest`，冒号后面是 tag）。所以必须 `partition(":")` 或 `split(":", 1)`；用 `split(":")` 会把它切成三段，然后你就得到一个叫 `qwen3` 的不存在的模型。

顺带解释本机 `.env` 的写法：我们不走 `ollama` provider（那要装 `langchain-ollama`，本机实测**未装**），而是走 ollama 暴露的 **OpenAI 兼容口** `http://localhost:11434/v1`，所以 provider 填 `openai`，模型名填 `qwen3:latest`。

### A0.4 `create_agent` 的 15 个参数，只有 4 个是 harness

实测打印（本机 `langchain 1.4.6`）：

| 堆 | 参数 | 决定什么 |
|---|---|---|
| **harness 层**（4 个） | `model` / `tools` / `system_prompt` / `middleware` | agent 怎么想、能做什么 |
| **运行时层**（11 个） | `response_format` / `state_schema` / `context_schema` / `checkpointer` / `store` / `interrupt_before` / `interrupt_after` / `debug` / `name` / `cache` / `transformers` | 这次跑起来怎么存、怎么停、怎么换模型 |

这张表能省你很多时间：**遇到"怎么让 agent 记住上一轮"这种问题，答案永远在运行时层，不在 harness 层。**

### A0.5 三个框架的判据（别背定义，背"什么时候选它"）

| 框架 | 什么时候选它 | 本机状态 |
|---|---|---|
| **Deep Agents** | 要开箱即用：上下文压缩、虚拟文件系统、子 agent 都已装好 | 已装 |
| **LangChain** | 要自己配 harness，逐项拧 | 本教材主战场 |
| **LangGraph** | 要确定性流程和 agent 混编、要自己控状态机 | 已装（隔壁那本） |
| **LangSmith** | 不管用哪个框架，都要 trace / 调试 / 评测 | 上线前必备 |

### A0.6 真跑一次的实测输出

```text
[E] 真跑一次：把官方例子的 model 换成本机模型
  消息条数：4（含 Human / AI / Tool）
  最后一条类型：AIMessage
  content_blocks（官方 overview 最后那行 print）：
    {'type': 'text', 'text': '杭州今天天气晴朗，阳光明媚，非常适合外出活动！'}
```

`4 条消息` 这个数字要记住：`Human → AI(带 tool_calls) → Tool → AI`。**官方那 6 行代码背后已经跑了一整圈循环**，只是你没看见。

### 认知修正

> 我原本以为 `create_agent` 那十几个参数都是"配置 agent 能力的"；实际上只有 `model / tools / system_prompt / middleware` 四个属于 harness，其余十一个是**运行时**——它们不改 agent 的性格，只改这一次执行怎么落地。
