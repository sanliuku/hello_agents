# 第 5 章 · State 与 Reducer：消息为什么越聊越少

> 本章目标：**能解释"同一份 agent 代码，加了个自定义 state 就失忆"，并且能自己写一个 reducer**。预计 80 分钟。
>
> 环境锚点：`langchain 1.4.6` / `langchain-core 1.6.12` / `langgraph 1.2.16`。本章所有输出都是在本机跑出来的，不是推的。

---

## ① 本章你会得到什么

1. 能说出 `AgentState` 只有 **3** 个字段，以及每个字段上挂的那个"元数据"是干什么的
2. 能解释 `state_schema=` **不是扩展、是替换**——这是本章最重要的一句话
3. 能画出"一个 state 变三份 schema"（State / Input / Output），并说出哪个字段会进哪一份
4. 能手写 reducer，并用一句话说清它的契约：`(当前值, 本次写入) -> 新值`
5. 能把"agent 失忆"这个玄学现象，定位到 `messages` 上那个 `Annotated[..., add_messages]` 是不是还在

第 2 条和第 5 条是分水岭。会了这两条，你看任何一份 `state_schema` 代码都能一眼判断它会不会炸。

---

## ② 源码解剖

本章碰三个文件，都在你的 `.venv` 里，路径照抄就能打开：

| 文件 | 里面是什么 |
|---|---|
| `.venv/Lib/site-packages/langchain/agents/middleware/types.py` | `AgentState`、`OmitFromSchema` 三兄弟、`EphemeralValue` 的用法 |
| `.venv/Lib/site-packages/langchain/agents/factory.py` | `state_schemas` 怎么拼、`_resolve_schemas` 怎么合并成三份 |
| `.venv/Lib/site-packages/langgraph/graph/message.py` | `add_messages` 的完整实现（就是那个 reducer） |

### 2.1 `AgentState` 只有三个字段

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

三行代码，三种不同的元数据。逐个拆：

| 字段 | 类型包装 | 挂的元数据 | 后果 |
|---|---|---|---|
| `messages` | `Required[...]` | `add_messages` | 这是一个 **reducer**：每次写入是"合并"，不是"替换" |
| `jump_to` | `NotRequired[...]` | `EphemeralValue` + `PrivateStateAttr` | 只在**这一步**有效，下一步自动清空；且不进 input/output schema |
| `structured_response` | `NotRequired[...]` | `OmitFromInput` | 你能在输出里拿到，但**不能**从输入塞进去 |

`Annotated[X, f]` 这层包装是关键：**类型还是 `list[AnyMessage]`，但旁边多了一个函数 `f`**。LangGraph 建图时看到这个注解，就不再给 `messages` 开"LastValue 通道"，而是开一个"每次写入都要过 `f` 的通道"。

实测（`05_state_reducer.py` A 段，`get_type_hints(AgentState, include_extras=True)`）：

```text
  messages               类型=typing.Required[typing.Annotated[list[typing.Annotated[...]]]]
                         ↳ 元数据：_add_messages
  jump_to                类型=typing.NotRequired[typing.Annotated[typing.Optional[Literal['to...
                         ↳ 元数据：EphemeralValue
                         ↳ 元数据：OmitFromSchema(input=True, output=True)
  structured_response    类型=typing.NotRequired[typing.Annotated[~ResponseT, OmitFromSchema(...]]]
                         ↳ 元数据：OmitFromSchema(input=True, output=False)
```

**注意第一行打印的是 `_add_messages`，不是 `add_messages`。** 因为 `add_messages` 头上盖了 `@_add_messages_wrapper`，真正的 channel 函数是被包过的那一层。你要 `is` 比较或者打日志时，别拿 `add_messages` 直接比。

> **`EphemeralValue` 的原文注释**（`langgraph/channels/ephemeral_value.py:15`）：*"Stores the value received in the step immediately preceding, clears after."* 这就是 `jump_to` 敢叫 `jump_to` 的原因——它是个"一次性信使"，不是记忆。

### 2.2 `state_schema=` 是替换，不是追加

这是本章最贵的一条。看 `factory.py:1231` 那三行：

```python
base_state = state_schema if state_schema is not None else AgentState
# Build an ordered list: middleware schemas first (in registration order),
# base_state last so it wins any field conflict.  This lets the caller's
# explicit state_schema override middleware annotations — e.g. passing
# a DeltaChannel-annotated schema wins over BinaryOperatorAggregate from
# AgentState without requiring a post-compilation patch.
state_schemas: list[type] = [*(m.state_schema for m in middleware), base_state]
```

三件事：

1. **不传 `state_schema`，基线才是 `AgentState`；传了，基线就是你的类。** 官方 `AgentState` 不再自动参与。
2. **你的类排在最后**，而合并规则是"后来者覆盖同名字段"——所以冲突时你赢。这句话反过来读就是坑：**你一不小心就把 `messages` 的 reducer 覆盖没了**。
3. 注释里那句 `passing a DeltaChannel-annotated schema wins over ... AgentState` 是这套设计的**本意**：故意让你能覆盖掉 `AgentState` 的注解。它是特性，不是 bug。

所以正确姿势只有一种：

```python
from langchain.agents.middleware import AgentState

class MyState(AgentState[Any]):     # ← 继承
    user_id: str                    # ← 只写新字段，别碰 messages
    retry_count: int
```

`state_schema=MyState` 时，`messages` 的注解从父类**继承**下来，reducer 还在；`jump_to` / `structured_response` 也还在。

### 2.3 一个 state 变三份 schema

`_resolve_schemas`（`factory.py:470`）把上面那个有序列表合成三个动态 `TypedDict`：

```python
def _resolve_schemas(schemas: list[type]) -> tuple[type, type, type]:
    schema_hints: dict[type, dict[str, Any]] = {}
    for schema in schemas:
        # Reinsert duplicates so dict iteration reflects their final position.
        schema_hints.pop(schema, None)
        schema_hints[schema] = _get_schema_type_hints(schema)
    return (
        _resolve_schema(schema_hints, "StateSchema", None),
        _resolve_schema(schema_hints, "InputSchema", "input"),
        _resolve_schema(schema_hints, "OutputSchema", "output"),
    )
```

两个细节值得记：

- `pop` 再插回去 —— **重复出现的 schema 会挪到末尾**，也就是"按最后一次出现的位置算优先级"。同一个类被两个 middleware 都声明了，不会算两次。
- 三个 schema 的区别只在于 `omit_flag`：`None` 全留，`"input"` 剔掉标了 `OmitFromSchema(input=True)` 的字段，`"output"` 剔掉 `output=True` 的。

而 `_extract_metadata`（`factory.py:526`）要处理三层包装，这是它存在的全部理由：

```python
if get_origin(type_) in {Required, NotRequired}:
    inner_type = get_args(type_)[0]
    if get_origin(inner_type) is Annotated:
        return list(get_args(inner_type)[1:])
elif get_origin(type_) is Annotated:
    return list(get_args(type_)[1:])
return []
```

**顺序不能反**：`Required[Annotated[X, meta]]` 的元数据在内层，你直接对最外层 `get_args` 只能拿到 `Annotated[...]` 这个整体。这也解释了为什么你自己解析 `AgentState` 时，用 `tp.__metadata__` 会拿到 `None`——它在第二层。

**三兄弟的定义**（`types.py:339-346`，直接抄）：

```python
OmitFromInput  = OmitFromSchema(input=True,  output=False)   # 不进输入，可输出
OmitFromOutput = OmitFromSchema(input=False, output=True)    # 可输入，不进输出
PrivateStateAttr = OmitFromSchema(input=True, output=True)   # 两头都不进，纯内部
```

实测（C 段，用下面这个 `TicketState`）：

```python
class TicketState(AgentState[Any]):
    thread_topic: str
    scratch: NotRequired[Annotated[str, PrivateStateAttr]]
    debug_blob: NotRequired[Annotated[dict, OmitFromOutput]]
    hidden_hint: NotRequired[Annotated[str, OmitFromInput]]
```

```text
  StateSchema  : ['messages', 'jump_to', 'structured_response', 'thread_topic', 'scratch', 'debug_blob', 'hidden_hint']
  InputSchema  : ['messages', 'thread_topic', 'debug_blob']
  OutputSchema : ['messages', 'structured_response', 'thread_topic', 'hidden_hint']
```

对着一张表看，规律一目了然：

| 字段 | State | Input | Output | 谁用得着 |
|---|---|---|---|---|
| `messages` | ✓ | ✓ | ✓ | 谁都要 |
| `jump_to` | ✓ | ✗ | ✗ | 只有图内部（`PrivateStateAttr`） |
| `structured_response` | ✓ | ✗ | ✓ | 只有出口（`OmitFromInput`） |
| `scratch` | ✓ | ✗ | ✗ | 纯内部草稿 |
| `debug_blob` | ✓ | ✓ | ✗ | 能传进来，但不外泄 |
| `hidden_hint` | ✓ | ✗ | ✓ | 能吐出去，但塞不进来 |

**这张表的实际用处**：`agent.invoke({...})` 时，你能传的字段 = `InputSchema` 的字段。传个 `structured_response` 进去，它不会报错，只会被丢掉——因为输入通道里根本没这个 key。想"预置一个结构化结果"，得走别的路（第 7 章）。

### 2.4 `add_messages`：五步，一行都不能少

这是那个被 `messages` 挂着的函数。全文不长，逻辑分五段：

```python
def add_messages(left, right, *, format=None):
    left  = [message_chunk_to_message(m) for m in convert_to_messages(left)]
    right = [message_chunk_to_message(m) for m in convert_to_messages(right)]

    # ① 缺 id 的补 uuid —— 注意是就地赋值
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())

    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    # ② 有个"清空"哨兵：它之后的东西留下，之前的全不要
    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # ③ 按 id 合并
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m          # ← 同 id 不是追加，是替换
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            merged_by_id[m.id] = len(merged)
            merged.append(m)

    # ④ 真正把标记删除的摘掉
    merged = [m for m in merged if m.id not in ids_to_remove]
    ...
    return merged
```

实测五种情况（B 段）：

```text
  B1 两个不同 id → 追加
    ['u1', 'a1']  len=2
  B2 同一个 id → 原地替换，不是追加
    len=1  content='你好（改过）'
  B3 RemoveMessage(id=...) → 删掉那一条
    剩下 ['2']
  B4 删不存在的 id → 直接抛 ValueError（不是静默忽略）
    ValueError: Attempting to delete a message with an ID that doesn't exist ('404')
  B5 REMOVE_ALL_MESSAGES → 之后只保留它后面的那几条
    [('new', '清空后的第一条')]
    REMOVE_ALL_MESSAGES 的字面值是 '__remove_all__'
  B6 没给 id 的会被就地补 uuid —— 注意 left 也被改了（副作用！）
    调用前 left[0].id = None
    调用后 left[0].id = 11ce28c3-9c92-4607-9aba-d39bc2562af8
```

四条能直接用的结论：

1. **"同 id 覆盖"就是"编辑历史"的唯一入口。** 想改一条已经发出去的消息，不能 append 新的，得**同 id 重发**。
2. **`RemoveMessage(id=...)` 是删除单条的唯一方式**，而且它**必须**放在写入的那个 list 里返回，不能"在 state 里删一下"。
3. **删一个不存在的 id 会抛 `ValueError`**，不是静默。所以删消息前先确认它真在 state 里——`summarization` 那类 middleware 都会先扫一遍 id。
4. **`REMOVE_ALL_MESSAGES` 是唯一能"重置上下文"的官方按钮**，`'__remove_all__'` 后面跟的消息才是新历史。这是写"清空重来"和"摘要替换"这两件事的正确姿势。

> **B6 那个副作用值得单独说**：`add_messages` 会给**传进来的**（包括 `left`）消息对象就地写 `id`。所以"我先构造好消息列表备用，之后复用"是危险动作——第一次合并之后就带 id 了，第二次合并会按 id 判重。这是懒初始化，不是纯函数。

### 2.5 reducer 的契约与"没有 reducer 会怎样"

LangGraph 的字段默认开 **LastValue** 通道：后写覆盖先写，只留最后一个。这就是"没有 reducer 的字段，历史全丢"的全部原因。

实测（D 段，同一张两节点的图，只差一个 `Annotated`）：

```text
  有 reducer → [('u1', '问题'), ('a1', '第一条'), ('a2', '第二条')]
  没 reducer → [('a2', '第二条')]
```

连输入的那条 `HumanMessage` 都没了，因为第一个节点返回 `{"messages": [...]}` 直接把整个字段覆盖掉了。

reducer 的签名契约只有一句：**`(当前 channel 值, 本步要写入的值) -> 新 channel 值`**。它必须是**纯函数**——别原地改 `left`，也别返回 `None`（`None` 不会报错，只会让字段静默变成 `None`，错误延后到"下一个读它的人"，极其难查）。

自己写一个（F 段，累加 + 去重并集）：

```python
def merge_unique(left: list[str], right: list[str]) -> list[str]:
    out = list(left or [])
    for x in right or []:
        if x not in out:
            out.append(x)
    return out

class MetricsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tokens:   Annotated[int, operator.add]
    visited:  Annotated[list[str], merge_unique]
```

```text
  tokens  = 175   （100+50+25，operator.add）
  visited = ['a', 'b', 'c']  （a 只出现一次，自定义并集）
```

`operator.add` 能直接当 reducer，是因为它刚好符合 `(a, b) -> a + b`。**LangGraph 不关心你传的是什么，只关心签名。**这一个认知能让你省掉很多"是不是要用官方 reducer"的犹豫。

---

## ③ 你敲这段

文件：`langchain_learn/05_state_reducer.py`，约 300 行，**A–F 段零 token，只有 `--real` 才联网**。

骨架（先照着敲这七段的标题和打印语句，细节自己填，填完再对照）：

```python
# A 读注解：AgentState 三个字段 + 各自的元数据
from typing import get_type_hints
hints = get_type_hints(AgentState, include_extras=True)   # include_extras 不能省

# B 纯函数：add_messages 的五种情况
add_messages([HumanMessage("你好", id="u1")], [AIMessage("在", id="a1")])

# C 三份 schema：_resolve_schemas([TicketState])，打印三个 __annotations__
from langchain.agents.factory import _resolve_schemas

# D 图级：同一张图，WithReducer vs NoReducer 各跑一次
class WithReducer(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
class NoReducer(TypedDict):
    messages: list[AnyMessage]

# E agent 级：一个只记录、不联网的假模型，看它第二次收到几条消息
class RecordingFakeModel(BaseChatModel):
    def bind_tools(self, tools, **kwargs): return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

# F 自定义 reducer：operator.add + merge_unique
```

### 逐段说清"为什么这么写"

**A 段为什么要 `include_extras=True`？** 因为不加，`get_type_hints` 会把 `Annotated[X, f]` 直接化简成 `X`，reducer 就看不见了。**这是整章最容易踩的一个技术细节**：所有"读注解"的代码都必须带这个参数，包括你自己写 middleware 时。

**C 段为什么要用 `_resolve_schemas` 这个私有函数？** 因为它是"三份 schema"的唯一真相来源。你也可以绕过它、自己拼 TypedDict，但那样一旦官方改了合并顺序你就会跟主流程不一致。**读私有函数是为了理解，生产代码里请用编译后的结果**——`agent.get_input_jsonschema()` / `get_output_jsonschema()` 就是官方出口。

**E 段为什么要自己写假模型？** 因为真实模型"收到几条消息"这件事你观察不到，而"失忆"的表现（答非所问）又可能被模型的语言能力掩盖。`RecordingFakeModel` 把"模型看到了什么"变成可打印的数字，**这是本章唯一能拿到铁证的手段**，也是第 21 章"假模型单测"的预演。

假模型只需要实现两件事：`_llm_type`（一个字符串）和 `_generate`（返回 `ChatResult`）。`bind_tools` 必须自己实现，因为基类里它默认 `raise NotImplementedError`——而 `create_agent` 一定会调它。

**F 段为什么 reducer 里要 `list(left or [])`？** 因为 channel 的初始值可能是 `None`（字段声明为 `NotRequired` 且没人写过）。生产上的 reducer 一律先兜住 `None`，否则第一次写入就 `TypeError`。

---

## ④ 你应该看到

先预测，再对照。E 段是本节的靶心（四种写法，同一个问题"北京天气怎么样？"）：

```text
  E1 默认 AgentState
    每次模型调用收到的消息条数 = [1, 3]
    第 1 次：['HumanMessage']
            最后一条内容 = '北京天气怎么样？'
    第 2 次：['HumanMessage', 'AIMessage', 'ToolMessage']
            最后一条内容 = '北京：晴，26℃'

  E2 继承 AgentState 只加字段（推荐）
    每次模型调用收到的消息条数 = [1, 3]
    ...（与 E1 完全相同）

  E3 继承但重声明了 messages（丢 reducer）
    每次模型调用收到的消息条数 = [1, 1]
    第 2 次：['ToolMessage']
            最后一条内容 = '北京：晴，26℃'

  E4 裸 TypedDict 完全不继承（丢 reducer + 丢字段）
    每次模型调用收到的消息条数 = [1, 1]
    第 2 次：['ToolMessage']
```

`[1, 3]` vs `[1, 1]` 就是本章的全部答案：

- **有 reducer**：第二轮模型看到"用户问 + 我调了工具 + 工具结果"三条，能正常作答。
- **没 reducer**：第二轮模型只看到一条 `ToolMessage`——**一个没有主语的工具结果**。它不知道是谁问的、为什么要查，只能瞎猜着编。

而最阴的地方在于：**它不报错。** 本机用 `qwen3:latest`（ollama 的 OpenAI 兼容端点）实测，E3 那个配置照样返回了一句像模像样的"北京今天天气晴朗，温度为 26 摄氏度……"。所以你看到的不是 400，不是异常，而是"agent 有时候答得有点飘"。**这类 bug 靠日志查不出来，只能靠"打印模型实际收到了几条消息"。**

顺带一个实测对照（G 段，真模型 / 默认 state），一次正常往返的 state 长这样：

```text
  [0] HumanMessage '北京天气怎么样？一句话。'
  [1] AIMessage    '' tool_calls=['get_weather']
  [2] ToolMessage  '北京：晴，26℃' tool_call_id=call_u1ijm81j
  [3] AIMessage    '北京天气晴，气温26℃。'
```

**`[1]` 那条 `AIMessage` 的 `content` 是空的，只有 `tool_calls`。** 这不是 bug——工具调用轮里模型本来就没说话。第 2 章讲过，这里终于看到它在 state 里的样子：**一次 `invoke` = 4 条消息，其中 2 条是 AI，1 条是工具。** 你的 `messages` 就是这么一条条涨起来的。

---

## ⑤ 三个破坏性实验

三个都是"删一行/改一行"，都别跳。

### 实验 A：把 `messages` 的 `Annotated` 拆掉（最像真人的错法）

```python
# 改前
class MyState(AgentState[Any]):
    topic: str

# 改后：想"顺手规范一下类型"，把 messages 重声明一遍
class MyState(AgentState[Any]):
    messages: list[AnyMessage]      # ← 灾难在这一行
    topic: str
```

跑 E 段那套假模型，看第二次调用收到的条数。

**你会看到**：从 `3` 变成 `1`，模型第二次只收到 `['ToolMessage']`。

**结论**：`Annotated` 不是装饰，**它就是 reducer 的挂载点**。拆掉它，Type 检查一点问题没有（`list[AnyMessage]` 完全合法），运行也不报错，但语义从"追加"静默变成"覆盖"。这个坑没有任何静态检查能帮你拦——除非你自己在 CI 里断言 `get_type_hints(MyState)["messages"]` 的元数据里有 `add_messages`。

> 反过来记一条：**想加字段就只写新字段**。`class MyState(AgentState[Any]): user_id: str` 这种写法永远是安全的（E2 已证），因为它不产生任何覆盖。

### 实验 B：直接拿裸 `TypedDict` 当 `state_schema`

```python
class Broken(TypedDict):
    messages: list[AnyMessage]

agent = create_agent(model, tools=[get_weather], state_schema=Broken)
```

先跑 `agent.get_input_jsonschema()` 和 `get_output_jsonschema()`，再把 E4 跑一遍。

**你会看到**（实测）：

```text
  默认   input props: ['messages']
  默认   output props: ['messages', 'structured_response']
  Broken input props: ['messages']
  Broken output props: ['messages']          ← structured_response 没了
```

**结论**：丢的不只是 reducer，**`structured_response` 从输出 schema 里整条消失**，`jump_to` 也没有了。因为 `state_schema` 是替换：你的类里没有的字段，图上就不存在。第 7 章的结构化输出会在这种 state 上直接失效，第 8–9 章的 middleware 跳转也会一起失灵，而**报错信息不会指向 `state_schema`**。

排查口诀：**"用了 middleware 或 structured_output，就先确认 state 是不是继承了 `AgentState`。"**

### 实验 C：让 reducer 返回 `None`

```python
def bad(left, right):
    return None      # "我不需要这个字段" —— 不，你不能这么写

class S(TypedDict):
    log: Annotated[list, bad]
```

**你会看到**（实测）：

```text
  返回 None 的结果: {'log': None}
```

**不报错。** state 里那个字段静默变成 `None`，然后下一个读它的节点在 `for x in state["log"]` 那里 `TypeError`——**报错位置离病因隔了好几个节点**。

**结论**：reducer 的返回值就是新 channel 值，`None` 是一个合法的值，框架没理由拦你。所以 reducer 里所有分支都必须 `return` 一个真值。这也是为什么 `add_messages` 里 `left or []`、`list(x or [])` 这类兜底随处可见——它们防的就是"上游写了个 `None` 进来"。

> 附加观察（本机实测，不是所有版本都这样）：`add_messages` 会**就地**给传入的消息补 `id`（B6）。所以想复用同一个消息列表做多次合并，第二次开始行为就不一样了。要复用，先 `copy`。

---

## ⑥ 自检题

答不上来就别进下一章。答案就在题目下面。

**1. 为什么 `AgentState.messages` 的类型是 `list[AnyMessage]`，但两个节点分别返回一条消息后，state 里会变成两条？**

<details><summary>答案</summary>

因为注解是 `Annotated[list[AnyMessage], add_messages]`。LangGraph 建图时读这层元数据，把这个字段的通道从 **LastValue** 换成 **BinaryOperatorAggregate(add_messages)**：每次写入都走 `add_messages(旧值, 新值)`，而它是"按 id 合并/追加"。没有这层 `Annotated`，就是 LastValue，后写覆盖先写。

</details>

**2. `state_schema=MyState` 时，`AgentState` 还参与吗？**

<details><summary>答案</summary>

**不直接参与。** `base_state = state_schema if state_schema is not None else AgentState`——你传了，基线就是 `MyState`。`AgentState` 的字段能被你拿到，**只因为你继承了它**（注解继承）。所以"用裸 TypedDict"和"继承 AgentState"是两种完全不同的东西，前者会丢掉 `jump_to` / `structured_response` / `messages` 的 reducer。

</details>

**3. 哪三个字段会出现在 OutputSchema 里、但不在 InputSchema 里？为什么？**

<details><summary>答案</summary>

`structured_response`（挂 `OmitFromInput`）、以及你自己用 `OmitFromInput` / `PrivateStateAttr` 标注的字段。`jump_to` 是两头都不进（`PrivateStateAttr`），不属于"只出"。规则来源是 `_resolve_schema(..., "output")` 剔掉 `OmitFromSchema(output=True)` 的字段，`"input"` 则剔 `input=True` 的。

</details>

**4. 想把一条已经写进 state 的消息**改掉内容**，怎么做？**

<details><summary>答案</summary>

用**同一个 `id`** 再 append 一条新内容的消息。`add_messages` 遇到已有 id 是"原地替换"（B2 实测），不是追加。想删就用 `RemoveMessage(id=...)`；想全清就用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 放在写入列表的第一位。

</details>

**5. reducer 的签名是什么？它能是 async 吗？**

<details><summary>答案</summary>

`(left, right) -> new_value`，其中 `left` 是当前 channel 值、`right` 是本步写入值，返回值成为新 channel 值。**不能是 async**——它在图的状态更新阶段被同步调用；本机实测把 reducer 写成 `async def`，跑完 state 里那个字段直接变成一个 `<coroutine object>`（`type(...) == coroutine`），不报错。要异步做的事放到节点里做，节点只负责 `return` 结果。

</details>

---

## ⑦ 承上启下

你现在知道了：`messages` 是一个会被 reducer 不断追加的列表，`invoke` 返回的 state 就是这一轮结束时的快照。

但有个问题一直悬着：**这个快照在你 `invoke` 返回之后去哪了？** 你调用两次 `agent.invoke({"messages": [...]})`，第二次它还记得第一次吗？

答案是**不记得**——除非你给了 `checkpointer`。而 `checkpointer` 到底存了什么（是整个 state，还是每个 channel 的增量？`thread_id` 是怎么变成主键的？为什么"同一 thread 的第二次 invoke 只需要传一条新消息"？），就是第 6 章要拆的。

去之前先带着这个问题：**`checkpointer` 存的是 state 快照，可 reducer 是"合并"语义——从快照恢复时，reducer 会被再执行一次吗？** 这个问题的答案决定了你能不能安全地在 reducer 里做副作用（比如发通知）。

> 想先歇一下的话，也可以跳到第 11 章看测试——`InMemorySaver` 那一节会先把"state 怎么被存"讲一半。但顺序上，第 6 章更顺。
