# 第 2 章 · Message 解剖

> 本章目标：**能逐字段解释一次工具调用往返的 4 条消息**，并且能手工续写这段历史——不靠 agent。预计 70 分钟。

---

## ① 本章你会得到什么

1. 能说出 `BaseMessage` 的 6 个字段各自归谁管（框架 / 模型供应商 / 你自己）
2. 能解释 `tool_call_id` 是**唯一**把"请求"和"结果"绑在一起的钥匙，缺一个 API 直接拒
3. 能读懂 `invalid_tool_calls`，知道"模型吐了坏 JSON"时框架替你做了什么
4. 能**不借助 `create_agent`**，自己拼 `[System, Human, AI(tool_calls), Tool]` 四条消息喂给模型，拿到续答

第 3 条和第 4 条是分水岭：会了这两条，你就不再是"调 API 的人"，而是"能造历史的人"。

---

## ② 源码解剖

本章只碰三个文件：

| 文件 | 里面是什么 |
|---|---|
| `libs/core/langchain_core/messages/base.py` | `BaseMessage` 基类、`content_blocks`、`.text`、`merge_content` |
| `libs/core/langchain_core/messages/ai.py` | `AIMessage`、`AIMessageChunk`、`UsageMetadata` |
| `libs/core/langchain_core/messages/tool.py` | `ToolMessage`、`ToolCall`、`default_tool_parser` |

### 2.1 `BaseMessage`：只有 6 个字段

```python
class BaseMessage(Serializable):
    content: str | list[str | dict[Any, Any]]
    additional_kwargs: dict[Any, Any] = Field(default_factory=dict)
    response_metadata: dict[Any, Any] = Field(default_factory=dict)
    type: str
    name: str | None = None
    id: str | None = Field(default=None, coerce_numbers_to_str=True)

    model_config = ConfigDict(extra="allow")
```

逐个说清"这个字段归谁管"——这是本章最省事的记忆方式：

| 字段 | 谁写的 | 用途 |
|---|---|---|
| `content` | 供应商原始内容（可能被框架规范） | 消息正文；字符串或内容块列表 |
| `additional_kwargs` | **供应商原样透传** | OpenAI 的 `tool_calls`、DeepSeek 的 `reasoning_content` 都先落在这 |
| `response_metadata` | 框架/供应商 | token 数、模型名、`model_provider`、`output_version` |
| `type` | 框架（子类写死） | `"human"` / `"ai"` / `"tool"` / `"system"`，**反序列化的判别键** |
| `name` | 你（可选） | 多角色对话里区分同名消息 |
| `id` | 供应商 | 消息 ID；`coerce_numbers_to_str=True` 说明有些端点会返回数字 |

两个细节值得停一下：

- **`extra="allow"`**：pydantic 允许未声明字段。所以你在消息上看到任何"文档里没有"的属性，都是靠这条放行的。
- **`type` 必须是字符串且各子类唯一**，源码注释写明了目的："to allow for easy identification of the message type when deserializing"。你以后写自定义消息类型，这行注释就是规范。

### 2.2 `content` 有三种形态，`.content_blocks` 把它们归一

`content` 的实际取值：

1. `str`（最常见）
2. `list[str | dict]`（多模态、Anthropic 风格）
3. 各种供应商私有格式（OpenAI 的 `image_url`、Bedrock 的 `image` …）

v1 引入了 `content_blocks` 属性做**归一化**，源码逻辑值得抄下来：

```python
    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        blocks: list[types.ContentBlock] = []
        content = ([self.content] if isinstance(self.content, str) and self.content
                   else self.content)
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})          # 裸字符串 → text 块
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type not in types.KNOWN_BLOCK_TYPES:
                    blocks.append({"type": "non_standard", "value": item})   # 不认识 → 打包
                else:
                    if "source_type" in item:                          # v0 老格式
                        blocks.append({"type": "non_standard", "value": item})
                        continue
                    blocks.append(cast("types.ContentBlock", item))    # 已经是 v1 块

        # 后续若干轮尝试把 non_standard 解开
        for parsing_step in [
            _convert_v0_multimodal_input_to_v1,
            _convert_to_v1_from_chat_completions_input,
            _convert_to_v1_from_anthropic_input,
            _convert_to_v1_from_genai_input,
            _convert_to_v1_from_converse_input,
        ]:
            blocks = parsing_step(blocks)
        return blocks
```

设计模式很清楚：**先归拢 → 再逐家翻译 → 解不开就留在 `non_standard` 里**。注释原话是 "This is the last stop - if we can't parse it here, it is left as non-standard"。

> **工程启示**：这就是"渐进式兼容"的写法——新格式加一个 translator，老代码不动。你以后写自己的消息/事件层可以直接套这个模式。

配套的 `.text` 属性只捞 `type == "text"` 的块：

```python
    @property
    def text(self) -> TextAccessor:
        if isinstance(self.content, str):
            text_value = self.content
        else:
            blocks = [b for b in self.content
                      if isinstance(b, str)
                      or (b.get("type") == "text" and isinstance(b.get("text"), str))]
            text_value = "".join(b if isinstance(b, str) else b["text"] for b in blocks)
        return TextAccessor(text_value)
```

`TextAccessor` 是个 `str` 子类，同时支持 `msg.text`（新）和 `msg.text()`（旧，会发 deprecation 警告，2.0.0 删除）。**你现在写代码一律用属性形式。**

### 2.3 `AIMessage`：三个新增字段

```python
class AIMessage(BaseMessage):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    usage_metadata: UsageMetadata | None = None
    type: Literal["ai"] = "ai"
```

`ToolCall` 是个 TypedDict，就三个必需键：

```python
class ToolCall(TypedDict):
    name: str          # 工具名
    args: dict[str, Any]   # 参数（已经是 dict，不是 JSON 字符串）
    id: str | None     # 标记这次调用
    type: NotRequired[Literal["tool_call"]]
```

`id` 字段的注释正面回答了本章的核心问题：

> "An identifier is needed to associate a tool call request with a tool call result in **events when multiple concurrent tool calls are made**."

**为什么需要 id？因为模型一次可能调 3 个工具。** 3 个请求、3 个结果，靠 `id` 配对；靠顺序配会在并发下错乱。

### 2.4 "坏 JSON" 去哪了：`invalid_tool_calls`

模型输出的工具参数是**字符串**（`'{"city": "杭州"}'`），要 `json.loads` 才能用。它会失败。源码不抛异常，而是分流：

```python
def default_tool_parser(raw_tool_calls):
    tool_calls, invalid_tool_calls = [], []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue                                    # 没有 function 字段：直接丢
        function_name = raw_tool_call["function"]["name"]
        try:
            function_args = json.loads(raw_tool_call["function"]["arguments"])
            tool_calls.append(tool_call(name=function_name or "", args=function_args or {},
                                        id=raw_tool_call.get("id")))
        except json.JSONDecodeError:
            invalid_tool_calls.append(invalid_tool_call(
                name=function_name, args=raw_tool_call["function"]["arguments"],
                id=raw_tool_call.get("id"), error=None))
    return tool_calls, invalid_tool_calls
```

三个要点：

1. **`json.JSONDecodeError` 是唯一的失败分支**，其他异常不管（比如 `KeyError` 会往上冒）
2. 坏的进 `invalid_tool_calls`，`args` 保留**原始字符串**，`error=None`（此时还不知道错在哪，后续执行工具时才发现）
3. 没有 `function` 键的原始调用被**静默丢弃** —— 这就是"模型明明说要调工具，但 `tool_calls` 是空的"的一种成因

还有一个容易被忽略的机制：**供应商原始格式自动升级**。有些端点（或老版本）只在 `additional_kwargs["tool_calls"]` 里塞 OpenAI 原始结构，`tool_calls` 字段是空的。`AIMessage._backwards_compat_tool_calls` 会在构造时补上：

```python
    @model_validator(mode="before")
    @classmethod
    def _backwards_compat_tool_calls(cls, values):
        check_additional_kwargs = not any(values.get(k) for k in
            ("tool_calls", "invalid_tool_calls", "tool_call_chunks"))
        if check_additional_kwargs and (raw := values.get("additional_kwargs", {}).get("tool_calls")):
            try:
                if issubclass(cls, AIMessageChunk):
                    values["tool_call_chunks"] = default_tool_chunk_parser(raw)
                else:
                    values["tool_calls"], values["invalid_tool_calls"] = default_tool_parser(raw)
            except Exception:
                logger.debug("Failed to parse tool calls", exc_info=True)
        ...
```

**记忆版结论**：`additional_kwargs` 是"原始账本"，`tool_calls` 是"标准化视图"，后者由前者推导，且推导失败会被 `debug` 级日志吞掉（所以你看不到报错，只会看到工具没被调）。

### 2.5 `ToolMessage`：三个必知字段

```python
class ToolMessage(BaseMessage, ToolOutputMixin):
    tool_call_id: str                                   # 必填！回填哪一次调用
    type: Literal["tool"] = "tool"
    artifact: Any = None                                # 不发给模型的完整产物
    status: Literal["success", "error"] = "success"      # 工具执行成功/失败
```

逐个看，都有工程含义：

**`tool_call_id`** —— 与 `AIMessage.tool_calls[i].id` **必须一一对应**。少一条、错一个，API 立刻 400（实验 A/B 会让你亲手撞墙）。

**`artifact`** —— 注释原文：

> "Artifact of the Tool execution which is **not meant to be sent to the model**. Should only be specified if it is different from the message content, e.g. if only a subset of the full tool output is being passed as message content but the full output is needed in other parts of the code."

这是**上下文工程的第一个开关**：给模型看摘要（`content`），给程序留全量（`artifact`）。第 10 章之后的 DeepAgents、以及你 rfwp_agent 里"大对象只给统计摘要 + artifact_ref"的设计，根子就在这里。

#### 2.5.1 `artifact` 到底解决什么问题（以及三个坑）

你问的"压缩后放 `content`、原文放 `artifact`"——**正是它的设计意图**，源码注释那句 "if only a subset of the full tool output is being passed as message content" 说的就是这个。但要补三点，否则会误用。

先看清数据流：`content` 和 `artifact` 走的是**两条完全不同的路**。

```
              ┌──────────────────────────────────────────────┐
工具返回全量 → │ ToolMessage                                  │
              │   content  ──→ state["messages"] ──→ 模型上下文（计费）│
              │   artifact ──→ 留在消息对象上 ──→ 你的代码 / 中间件   │
              └──────────────────────────────────────────────┘
                                    │
                                    └──→ 整个 state 进 checkpointer（序列化落盘）
```

**坑 1：框架不会自动帮你压缩。**

`ToolNode` 拿到一个字符串返回值，只会把它塞进 `content`；`artifact` 保持 `None`。想分流，必须你自己动手，有两条路：

```python
# 路线 A：直接返回 ToolMessage（最可控，推荐）
from langchain_core.messages import ToolMessage

@tool
def run_tests() -> ToolMessage:            # 注意：返回类型是 ToolMessage
    """跑测试并返回摘要 + 全量日志。"""
    full_log = "..."                       # 20 万字符
    return ToolMessage(
        content="共 2000 行；FAILED 3 处；首条：tests/test_api.py:42",
        artifact={"stdout": full_log, "failed": ["tests/test_api.py:42"]},
        tool_call_id="",                   # 会被 ToolNode 用真实 id 覆盖
    )
```

```python
# 路线 B：content_and_artifact 响应格式（更省事，取决于版本）
@tool(response_format="content_and_artifact")
def run_tests() -> tuple[str, dict]:
    """跑测试。"""
    return "摘要…", {"stdout": full_log}   # (content, artifact)
```

> 路线 B 依赖 `langchain_core` 的 `response_format="content_and_artifact"` 支持，并要求 `ToolNode` 认得这个元组。你实跑一次确认；若报错，说明你的版本没接上，用路线 A。**教材里凡是标"取决于版本"的地方，都是让你实测而不是猜——这也是本教材的诚实边界。**

**坑 2：`artifact` 不占模型上下文，但**占** checkpoint。**

`artifact` 是 `ToolMessage` 的字段，而 `ToolMessage` 在 `state["messages"]` 里。所以：

| | 进模型请求？ | 进 checkpoint？ |
|---|---|---|
| `content` | ✅（每个 token 都计费） | ✅ |
| `artifact` | ❌ | ✅ |

开 checkpointer（第 6 章）后，state 会被序列化落盘。塞 20 万字符进 `artifact`，模型确实看不见，但**每次 `invoke` 结束都要把这份字节写进 SQLite**，多轮对话会越滚越大。所以超大产物该走第三条路——

**坑 3：超大对象别放 `artifact`，放**引用**。**

三层分流，判据是"模型下一步需要读到哪些字符才能决策"：

| 层 | 放什么 | 谁读 |
|---|---|---|
| `content` | 决策所需的**摘要/结论**（几行到几百字） | 模型 |
| `artifact` | 结构化**中等**产物（几 KB～几百 KB，JSON 可序列化） | 你的中间件、下游节点 |
| 外部存储 | **超大**产物（日志、文件、图片），`artifact` 里只放路径/URI/对象 key | 你按需再去读 |

外部那一层的写法：

```python
path = save_to_disk(full_log)              # 落到 storage/artifacts/xxx.log
return ToolMessage(
    content=f"日志 2000 行，FAILED 3 处。完整日志：{path}",
    artifact={"log_path": path, "failed": ["tests/test_api.py:42"]},   # 只留引用
    tool_call_id="",
)
```

> **另一条硬要求**：`artifact` 必须是**可序列化**的（`str` / `int` / `dict` / `list` 这套 JSON 家族最稳）。塞自定义对象、打开的文件句柄、DB 连接进 `artifact`，checkpoint 序列化时会出问题——而且往往在第一轮跑得通、第二轮 `resume` 时才炸。

**一句话记忆**：`content` 是**给模型的**，`artifact` 是**给代码的**，外部存储是**给时间的**。三者按"谁要读、读多少"分层，不要拿 `artifact` 当垃圾桶。

**`status`** —— 让"工具报错"变成一种**正常数据**而不是异常。`"error"` 的 ToolMessage 照样回到模型，模型会看到错误并自己决定要不要换个参数重试。

还有一个 `model_validator` 在偷偷兜底：

```python
    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values):
        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)
        if not isinstance(content, (str, list)):
            try:
                values["content"] = str(content)      # 返回 dict/list 之外的东西 → 强转字符串
            except ValueError as e:
                raise ValueError(...)
        elif isinstance(content, list):
            values["content"] = []
            for i, x in enumerate(content):
                if not isinstance(x, (str, dict)):
                    values["content"].append(str(x))  # 列表里的怪元素 → 强转
                else:
                    values["content"].append(x)

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)   # 数字 id 也收
        return values
```

**这就是为什么你的工具可以 `return 42`**——框架会自动包成 `ToolMessage(content="42")`。`ToolOutputMixin` 的注释也说了：如果工具返回的不是它的实例，"the output will automatically be coerced to a string and wrapped in a `ToolMessage`"。

### 2.6 一次往返的 4 条消息（本章的核心图）

```
① SystemMessage      ← 你写的
② HumanMessage       ← 用户
③ AIMessage          ← 模型第一次响应，content 可能是空串，但 tool_calls 有值
   tool_calls = [{"name":"get_weather", "args":{"city":"杭州"}, "id":"call_001"}]
④ ToolMessage        ← 你/框架执行工具后造出来的
   content = "杭州：晴，26℃"
   tool_call_id = "call_001"     ← 与 ③ 的 id 严格相等
⑤（回到模型后）AIMessage  ← 模型读完 ④，给出面向用户的自然语言答案
```

**④ 的 `tool_call_id` 是整条链的铰链。** 换掉它 → 断；删掉 ④ → 断。

### 2.7 顺带看一眼流式：`AIMessageChunk`

流式时 `AIMessageChunk` 多一个字段：

```python
class AIMessageChunk(AIMessage, BaseMessageChunk):
    tool_call_chunks: list[ToolCallChunk] = Field(default_factory=list)
    chunk_position: Literal["last"] | None = None
```

`ToolCallChunk.args` 是**字符串片段**，不是 dict。合并规则写在源码 docstring 里，直接抄：

```python
    left_chunks  = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None,  args="1}",   index=0)]

    (AIMessageChunk(content="", tool_call_chunks=left_chunks)
     + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
```

**同 `index` 的片段按字符串拼接**（`merge_lists` 负责），拼完才 `parse_partial_json` 成 `tool_calls`。

这解释了你在调试流式时必然遇到的现象：**中途打印 `chunk.tool_calls`，args 会是半截/空的**。那不是 bug，是还没收完。

另外 `usage_metadata` 里有一组你会反复用的字段：

```python
class UsageMetadata(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: NotRequired[InputTokenDetails]    # cache_read / cache_creation / audio
    output_token_details: NotRequired[OutputTokenDetails]  # reasoning / audio
```

`cache_read` / `cache_creation` 就是**提示词缓存命中情况**。想算清一次 Agent 到底花多少钱，靠的是它，不是 `total_tokens`。

---

## ③ 你敲这段

### 3.1 新脚本 `langchain_learn/02_message_anatomy.py`

前 5 段**完全不花 token**（纯本地构造/解析），最后两段才调模型。这个顺序是刻意的：先在没有不确定性的时候把结构钉死。

```python
"""第 2 章：消息解剖。A–E 段纯本地，不调模型；F–G 段才真发请求。"""

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langchain_core.messages.tool import default_tool_parser
from langchain_core.tools import tool

from _shared import dump_messages, get_model

# ============ A. 一条 AIMessage 的真实字段 ============
print("=" * 64)
print("A. AIMessage 全字段")
print("=" * 64)
ai = AIMessage(
    content="",                                           # 注意：可以是空串
    tool_calls=[{"name": "get_weather", "args": {"city": "杭州"}, "id": "call_001"}],
)
print("type       :", ai.type)
print("tool_calls :", ai.tool_calls)
print("model_dump():")
print(json.dumps(ai.model_dump(), ensure_ascii=False, indent=2))
print("\nmessage_to_dict() 的骨架:", list(message_to_dict(ai).keys()))

# ============ B. 坏 JSON → invalid_tool_calls（纯本地） ============
print("\n" + "=" * 64)
print("B. default_tool_parser：坏 JSON 去哪了")
print("=" * 64)
raw = [
    {"id": "call_ok", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
    {"id": "call_bad", "function": {"name": "add", "arguments": '{a: 1, b: 2}'}},  # 非法 JSON
    {"id": "call_nofunction"},                                                     # 没有 function
]
ok_calls, bad_calls = default_tool_parser(raw)
print("tool_calls        :", ok_calls)
print("invalid_tool_calls:", bad_calls)
print("→ 第 1 条成功；第 2 条进 invalid（args 保留原始字符串，error=None）；第 3 条被静默丢弃")

# ============ C. additional_kwargs 里的原始格式会被自动升级 ============
print("\n" + "=" * 64)
print("C. _backwards_compat_tool_calls：只有 additional_kwargs 会怎样")
print("=" * 64)
legacy = AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [{
            "id": "call_x", "type": "function",
            "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'},
        }]
    },
)
print("tool_calls（本该是空的，被自动填上）:", legacy.tool_calls)

# ============ D. content 两种形态 vs content_blocks ============
print("\n" + "=" * 64)
print("D. content 的形态与 .text / .content_blocks")
print("=" * 64)
m1 = HumanMessage(content="纯文本")
m2 = HumanMessage(content=[
    {"type": "text", "text": "请描述这张图："},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
])
for label, m in (("m1", m1), ("m2", m2)):
    print(f"\n{label}.content       =", m.content)
    print(f"{label}.text           =", repr(m.text))
    print(f"{label}.content_blocks =", json.dumps(m.content_blocks, ensure_ascii=False))
print("→ 裸字符串被归一成 {'type':'text'} 块；provider 私有块先进 non_standard，"
      "再由 translator 尝试解开（解不开就留着）")

# ============ E. ToolMessage 的三个特殊字段 ============
print("\n" + "=" * 64)
print("E. ToolMessage：coerce / status / artifact")
print("=" * 64)
tm_num = ToolMessage(content=42, tool_call_id=123)          # 数字 content + 数字 id
print("\ncontent(数字 42) →", repr(tm_num.content), "| tool_call_id(数字 123) →", repr(tm_num.tool_call_id))
tm_big = ToolMessage(
    content="stdout 前 200 字……（给模型看的摘要）",
    artifact={"stdout": "x" * 5000, "files": ["a.csv"]},     # 全量，不发给模型
    tool_call_id="call_001",
    status="error",
)
print("status  :", tm_big.status)
print("artifact 键:", list(tm_big.artifact.keys()), "| content 长度:", len(tm_big.content))
print("→ 模型只看 content；artifact 留给你的程序")

# ============ 以下开始真调模型 ============
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前天气。用户询问天气时必须调用此工具。"""
    return f"{city}：晴，26℃，微风"


model = get_model().bind_tools([get_weather])

# ============ F. 手工续写历史：不靠 create_agent ============
print("\n" + "=" * 64)
print("F. 自己拼 4 条消息，让模型续答")
print("=" * 64)

messages = [
    SystemMessage("你是天气助手，用中文简洁回答。"),
    HumanMessage("杭州今天天气怎么样？"),
]

first = model.invoke(messages)
print("第 1 次响应 tool_calls:", first.tool_calls)

call = first.tool_calls[0]
tool_result = get_weather.invoke(call["args"])      # 我们手动执行工具
print("手动执行结果:", tool_result)

messages.append(first)
messages.append(
    ToolMessage(content=tool_result, tool_call_id=call["id"])   # 铰链：id 必须一致
)

second = model.invoke(messages)
dump_messages(messages + [second])

# ============ G. 流式：args 是半截 JSON ============
print("\n" + "=" * 64)
print("G. 流式下的 tool_call_chunks")
print("=" * 64)
acc = None
for chunk in model.stream([HumanMessage("杭州天气怎么样？")]):
    for tc in chunk.tool_call_chunks:
        print("  chunk:", tc)
    acc = chunk if acc is None else acc + chunk
print("\n合并后的 tool_calls:", acc.tool_calls)
```

### 3.2 运行

```powershell
python langchain_learn/02_message_anatomy.py
```

---

## ④ 你应该看到

**A 段** `model_dump()` 里应能看到（节选，字段顺序可能不同）：

```json
{
  "content": "",
  "additional_kwargs": {},
  "response_metadata": {},
  "type": "ai",
  "name": null,
  "id": null,
  "tool_calls": [{"name": "get_weather", "args": {"city": "杭州"}, "id": "call_001", "type": "tool_call"}],
  "invalid_tool_calls": [],
  "usage_metadata": null
}
```

注意 `type: "tool_call"` 是**你在构造时没写**的——`_backwards_compat_tool_calls` 里的 `create_tool_call(...)` 补上的。

**B 段**应是：

```
tool_calls        : [{'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': 'call_ok', 'type': 'tool_call'}]
invalid_tool_calls: [{'name': 'add', 'args': '{a: 1, b: 2}', 'id': 'call_bad', 'error': None, 'type': 'invalid_tool_call'}]
```

**C 段**应是 `[{'name': 'add', 'args': {'a': 3, 'b': 4}, 'id': 'call_x', 'type': 'tool_call'}]`——原本空的，被补上了。

**D 段** `m1.content_blocks` 是 `[{"type": "text", "text": "纯文本"}]`；`m2` 的第二块**可能**被翻译成 `{"type": "image", ...}`，也可能仍是 `non_standard`——取决于你装的版本里 translator 覆盖到哪一步。**这个"可能"是本教材允许的不确定性，不是你敲错了。**

**F 段**消息轨迹应是 4 条：`System / Human / AI(tool_calls) / Tool / AI(自然语言)`——其实是 5 条（我把它算成"一次往返 4 条 + 最终答复 1 条"，这也是上一章说"轨迹 4 条"的由来：`create_agent` 不把 System 放进 `state["messages"]`，它是单独传的）。

> ⚠️ 这里修正上一章的一句话：第 1 章说"轨迹 4 条"，指的是 `result["messages"]`（Human / AI / Tool / AI）。System 不在里面——`create_agent` 用 `system_prompt` 参数单独持有它。你在 F 段手动拼历史时把它显式放进了列表，所以看到 5 条。**两种都合法，区别在于"谁持有 system"。**

**G 段**应看到若干 `chunk: {'name': 'get_weather', 'args': '{"city": ', 'index': 0, ...}` 这样的半截片段，最后 `合并后的 tool_calls` 是完整的 dict。

---

## ⑤ 三个破坏性实验

只改 F 段那 4 行。

### 实验 A：删掉 `ToolMessage`，只留 `AIMessage(tool_calls)`

```python
messages.append(first)
# messages.append(ToolMessage(...))   ← 注释掉
second = model.invoke(messages)
```

预期报错形如（措辞依端点而定）：

```
Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls'
must be followed by tool messages responding to each 'tool_call_id'.
The following tool_call_ids did not have response messages: call_001"}}
```

**结论**：这不是 LangChain 的约束，是**模型 API 的硬约束**。LangChain 只是转达。所以"消息历史不合法"这个错误，永远在发请求那一刻才暴露。

### 实验 B：把 `tool_call_id` 改成一个不存在的值

```python
ToolMessage(content=tool_result, tool_call_id="call_does_not_exist")
```

同样是 400，措辞会变成"没有对应的 tool_call 请求"。

**结论**：`tool_call_id` 是**双向严格匹配**。这就是为什么 `create_agent` 每次进 `model` 节点第一件事是 `_patch_invalid_tool_calls(state["messages"])`（第 1 章 §2.7）——它要保证历史里没有这种会立刻 400 的坏账。

### 实验 C：让工具返回 dict，看 `coerce_args` 干活

```python
@tool
def get_weather(city: str) -> dict:      # 注意：返回 dict
    """查询指定城市当前天气。"""
    return {"city": city, "temp": 26, "desc": "晴"}
```

重跑 F 段，打印 `messages[3].content` 的类型。

**结论**：你会看到 `content` 仍是 `str`（dict 被 `str()` 了）。若想让模型看到结构化内容，得**自己**用 `ToolMessage(content=..., artifact=...)` 显式构造，或让工具返回 `ToolMessage`/`ToolOutputMixin`——**框架默认行为是"字符串化"，不是"结构化"**。这一条能解释你以后 90% 的"工具返回了对象但模型好像没看懂"。

---

## ⑥ 自检题

1. `additional_kwargs` 和 `tool_calls` 都能存工具调用，为什么要两份？
2. 模型一次返回 3 个 `tool_calls`，你要造几条 `ToolMessage`？它们靠什么和请求配对？
3. `ToolMessage.artifact` 的存在，说明"上下文"和"数据"是两回事。举一个你项目里该用它、而不是塞进 `content` 的场景。
4. 你在流式里打印 `chunk.tool_calls[0]["args"]`，有时是 `{}`、有时是 `{"city": "杭州"}`。为什么？

### 参考答案

> 先自己做，再对答案。每条后面的 **延伸** 是超出题面的部分，用来连到后面的章。

**1. `additional_kwargs` 和 `tool_calls` 都能存工具调用，为什么要两份？**

`additional_kwargs` 是**供应商原始报文**（每家字段名、嵌套都不同，OpenAI 是 `function.arguments` 字符串，别家可能是别的形状）；`tool_calls` 是**跨供应商的标准化视图**。前者保证信息不丢、能还原原始请求；后者保证框架能统一处理。

两者的关系是**单向推导**：`_backwards_compat_tool_calls` 在构造时把原始 → 标准。反过来不成立——你改了 `tool_calls` 不会回写 `additional_kwargs`。

> **延伸**：这条给你一个调试动作——当"模型明明说要调工具，但 `tool_calls` 是空的"，先看 `additional_kwargs` 里有没有 raw。**有 raw 而 `tool_calls` 空 = 解析失败**，而失败只写 `logger.debug("Failed to parse tool calls")`。把日志级别调到 DEBUG 才看得见。这是本教材里第一个"静默失败"陷阱。

**2. 模型一次返回 3 个 `tool_calls`，要造几条 `ToolMessage`？靠什么配对？**

**3 条**，`tool_call_id` 分别等于那 3 个调用的 `id`。**顺序无所谓**——配对只认 id，不认位置。

> **延伸**：这个"顺序无关"能当诊断工具用。你把 3 条 `ToolMessage` 故意乱序放，模型照样答对 → 说明配对机制正常，问题在工具本身；如果乱序就报 400 → 说明你的端点对顺序有额外要求（个别兼容实现确实如此）。

**3. `artifact` 说明"上下文"和"数据"是两回事。举个该用它而不是塞 `content` 的例子。**

判据一句话：**模型下一步需要读到哪些字符才能决策？** 只有这些进 `content`，其余全归 `artifact`。

具体例子：一个跑测试的工具，输出 20 万字符日志。

```python
ToolMessage(
    content="共 2000 行；FAILED 3 处；首条：tests/test_api.py:42 assert 500 == 200",
    artifact={"stdout": full_log, "failed": ["tests/test_api.py:42", ...]},
    tool_call_id=call["id"],
)
```

模型看到的是三行摘要，足够决定"去修 test_api.py:42"；完整日志留在 `artifact`，给你的下游节点、落盘、或下一轮精确检索用。

> **延伸**：LangGraph 的 `ToolNode` 默认只把 `content` 塞回 `messages`，`artifact` 留在 `ToolMessage` 对象上。所以你的中间件、你的自定义节点**还能拿到它**——它不是被丢弃，只是不发给模型。这一点在第 9 章写 middleware 时会直接用到。

**4. 流式里 `chunk.tool_calls[0]["args"]` 有时是 `{}`、有时完整，为什么？**

因为流式的 `args` 是**逐 token 拼出来的字符串**，不是一次性到齐的 dict。`parse_partial_json` 对半截 JSON 只能尽力而为：收到 `'{"city": '` 时凑不出键值对，就给出 `{}`；收完 `"杭州"}` 才解析成完整 dict。

源码里 `AIMessageChunk.init_tool_calls` 就是这个"尽力而为"的实现——每次拿到新 chunk 都重新尝试解析一遍。

> **延伸**：所以在流式 UI 里**不要在 chunk 阶段渲染工具参数**，等 `chunk_position == "last"` 或等 `+` 合并完再读。这也是 `chunk_position` 这个字段存在的理由：它标记"这一段是聚合后的收尾块"。

---

## ⑦ 承上启下

你现在能造历史了。但 `model.bind_tools([get_weather])` 这一行还是黑盒：**工具的函数签名是怎么变成 JSON Schema 发给模型的？`"openai:gpt-4o"` 这样的字符串又是怎么变成实例的？**

下一章拆 Model 层，你会看到 `init_chat_model` 如何解析 `"provider:model"`，以及 `bind_tools` 究竟往 `additional_kwargs` 里塞了什么。

➡️ 第 3 章 · Model 层

---

## 附录 · 官方写法对照（messages 页）

> 官方锚点：<https://docs.langchain.com/oss/python/langchain/messages>
> 代码：`langchain_learn/02_official_messages.py`（A–E 段零 token，`--real` 才联网）

正文是"切开一条消息看字段"，这一节走官方那条路：**同一件事，官方用哪几句话讲清楚**。官方页的六节刚好可以一比一映射成六个实验。

| 官方小节 | 它要你记住什么 | 对应代码段 |
|---|---|---|
| Basic usage | 三种输入写法，最后都归一成消息对象 | A |
| Message types | System / Human / AI / Tool 四类，各带什么字段 | B |
| Message content | `content_blocks` 是标准化视图，不是替代品 | C |
| Tool message → artifact | 有数据"给程序看、不给模型看" | D |
| Serialization | `dumpd` / `load` | E |
| Use with chat models | `usage_metadata`、chunk 相加 | F |

### A2.1 三种输入写法，最后都归一

实测：

```text
  ① 纯字符串     convert_to_messages -> ['HumanMessage']
  ② 消息对象     convert_to_messages -> ['SystemMessage', 'HumanMessage']
  ③ 字典格式     convert_to_messages -> ['SystemMessage', 'HumanMessage']
```

`role` 的映射表就是全部秘密：`system→SystemMessage`、`user→HumanMessage`、`assistant→AIMessage`、`tool→ToolMessage`。

**但这里有个真坑**（正文没提，官方页也没提）：

```text
  陷阱：convert_to_messages('hi') -> 2 条消息！
  它按字符遍历了。而 model.invoke('hi') 是合法的——包装发生在模型层，不是这里。
  正确写法：convert_to_messages(['hi']) -> ['HumanMessage']
```

`convert_to_messages` 收的是**序列**，裸字符串会被当字符序列遍历。所以"`model.invoke("xx")` 能跑"不代表"`convert_to_messages("xx")` 能跑"——**字符串的包装发生在模型层，不在这层。**

### A2.2 `content_blocks` 到底抹平了什么

这是官方页最值钱的一段。它给了两个原生格式完全不同的例子：

```python
# Anthropic 风格：thinking + signature
AIMessage(content=[{"type": "thinking", "thinking": "...", "signature": "WaUjzkyp"}, ...],
          response_metadata={"model_provider": "anthropic"})

# OpenAI 风格：reasoning + summary 数组
AIMessage(content=[{"type": "reasoning", "id": "rs_abc123",
                    "summary": [{"type": "summary_text", "text": "..."}, ...]}, ...],
          response_metadata={"model_provider": "openai"})
```

实测的 `content_blocks`：

```text
  Anthropic: [{'type': 'reasoning', 'reasoning': '先把问题拆成两步', 'extras': {'signature': 'WaUjzkyp'}},
              {'type': 'text', 'text': '答案是 42'}]

  OpenAI   : [{'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': '先算加法'},
              {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': '再取整'},
              {'type': 'text', 'text': '答案是 42', 'id': 'msg_abc123'}]
```

三条要点：

1. 一个叫 `thinking`、一个叫 `reasoning`，抹平后**都是 `type='reasoning'`**。
2. `content` 原封不动，`content_blocks` 只是**另一个视图**，不覆盖原数据。
3. **块数会变**：OpenAI 那个的 `summary` 两条被展开成两个 reasoning 块（2 块 → 3 块）。所以别假设 `len(content) == len(content_blocks)`。

### A2.3 `artifact`：给程序看，不给模型看

```text
  content  （进模型上下文）: It was the best of times, it was the worst of times.
  artifact （不进模型）    : {'document_id': 'doc_123', 'page': 0, 'score': 0.93}
```

正文讲 `ToolMessage` 时只提了 `tool_call_id` 是硬约束。官方这里补了另一半：**消息里有一块专门留给"不污染上下文"的数据**。检索分数、原始 JSON、页码、调试信息全塞这里——第 16 章 agentic RAG 会反复用到它。

### A2.4 序列化：能用，但会给你两条警告

实测 `dumpd` 出来是 `{'lc': ..., 'type': 'constructor', 'id': [...], 'kwargs': {...}}`，`load` 能还原成 `HumanMessage`。但你会看到两条警告：

```text
LangChainBetaWarning: The function `load` is in beta.
LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change...
```

这不是噪音，是信号：**`load` 默认允许反序列化任意对象，未来会收紧。** 官方那条安全警告要照着做——只 `load` 你自己写出去的数据；对不可信来源，显式传 `allowed_objects="messages"`。

### A2.5 真跑一次：属性和 chunk 都是实打实的

```text
  F1 invoke 返回：
    type           : AIMessage
    text           : 消息是信息从发送者传递到接收者的过程或内容。
    content_blocks : [{'type': 'text', 'text': '消息是信息从发送者传递到接收者的过程或内容。'}]
    usage_metadata : {'input_tokens': 16, 'output_tokens': 386, 'total_tokens': 402, ...}
    response_metadata 键: ['token_usage', 'model_provider', 'model_name',
                           'system_fingerprint', 'id', 'finish_reason', 'logprobs']

  F2 stream 拼接：
    chunk 个数      : 231
    拼回来的类型    : AIMessageChunk
    拼回来的 text   : 1, 2, 3, 4, 5.
```

两个数字值得盯：`output_tokens=386` 而答案只有一句话——**多出来的是 qwen3 的思考 token**，`content_blocks` 里看不到它们。`231 个 chunk` 说明**流式块的大小完全由供应商决定**，别在 chunk 阶段渲染工具参数（正文⑦那条延伸的实测依据）。
