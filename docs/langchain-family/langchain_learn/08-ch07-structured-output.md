# 第 7 章 · 结构化输出：`response_format` 如何把 Pydantic 模型变成模型的输出约束

> **官方锚点**：`/oss/python/langchain/structured-output`
> **源码落点**：`langchain/agents/structured_output.py`（472 行）+ `langchain/agents/factory.py`（2164 行，`response_format` 的消费方）
> **前置**：第 1 章（一次 invoke 的真相）、第 2 章（Message 解剖）、第 3 章（Model 层 `bind_tools`）
> **本地版本**：`langchain 1.4.6` / `langchain-core 1.6.12`
> **源码文件位置（本机）**：`.venv/Lib/site-packages/langchain/agents/structured_output.py`
> **诚实边界**：本章所有行号、代码片段、函数签名均来自上机安装包的实际读取；**"你应该看到"的输出是按源码推导的期望值，本章代码未经实机运行验证**（本机沙箱拒绝了 `pwsh` 执行权限）。凡推导处均标注 `[源码推导]`。

---

## ① 本章你会得到什么

学完这一章，你应当能回答下面五个问题，**不用回头看源码**：

1. `create_agent(response_format=SomeModel)` 里那个 `SomeModel`，最终是以什么形式出现在发给模型的 HTTP 请求体里的？
2. `ToolStrategy` 和 `ProviderStrategy` 的差别到底是什么——是"两种写法"还是"两条完全不同的物理通道"？
3. 模型吐了一个不合 schema 的 JSON，框架是**重试**、**抛异常**，还是**悄悄给你个错对象**？三种策略各自怎么反应？
4. `AutoStrategy` 为什么是一个"什么方法都没有"的类？它的存在意义是什么？
5. `response_format={"type": "object", ...}` 传原始 dict 时，为什么 `handle_errors=True` 会**静默失效**？

### 验收清单（做完打勾）

- [ ] 能画出「schema → `_SchemaSpec` → 两条分支（`OutputToolBinding` / `to_model_kwargs`）→ 请求体」这条链路
- [ ] 能手写 `_SchemaSpec(Schema).json_schema` 并解释它和 `Schema.model_json_schema()` 的关系
- [ ] 能在不看源码的情况下说出 `ToolStrategy` 生成的"假工具"叫什么名字、`args_schema` 是什么
- [ ] 能解释 `AutoStrategy` → `ToolStrategy` → `ProviderStrategy` 的"先建工具后丢弃"的浪费行为为什么是必要的
- [ ] 亲手做完全部 4 个破坏实验，并写下你的"我原本以为…实际上…"

---

## ② 源码解剖

### 2.1 门面：`ResponseFormat` 就是三个类的联合类型

文件最后两行（L471-472）是整个模块的结论：

```python
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
```

`create_agent(response_format=...)` 接受的合法形态只有这四种：

| 你传的东西 | 框架内部归一成 | 谁决定的 |
|---|---|---|
| `None` | `None`（不做结构化输出） | 直接 |
| `ToolStrategy(...)` / `ProviderStrategy(...)` / `AutoStrategy(...)` | 原样保留 | 你 |
| 一个 Pydantic 模型类 / dataclass / TypedDict / dict | `AutoStrategy(schema=你传的)` | factory L1074 |

第四行是关键：**绝大多数人传裸的 Pydantic 类，框架就把它包成 `AutoStrategy`**。所以"自动"不是玄学，是这个包装动作。

### 2.2 `SchemaKind` 与 `_SchemaSpec`：所有翻译的起点

```python
SchemaT = TypeVar("SchemaT")

SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]
```

四种 schema 形态，对应四种 `SchemaKind`。注意最后那个叫 `json_schema` 而不是 `dict`——源码里统一用 `json_schema` 这个词。

`_SchemaSpec`（L106-192）是整个模块的**中央车站**：不管哪条策略，第一步都是把你给的 schema 压成它。它有 5 个字段：

```python
schema: type[SchemaT] | dict[str, Any]   # 原始 schema
name: str                                 # 工具名 / json_schema 的 name
description: str                          # 工具描述 / schema 描述
schema_kind: SchemaKind                   # 四种之一
json_schema: dict[str, Any]               # 真正的 JSON Schema
strict: bool | None = None                # 仅 ProviderStrategy 用得上
```

`__init__`（L139-192）干了三件事，**顺序不能乱**：

**第一步，定 name**（L160-165）：

```python
if name:
    self.name = name
elif isinstance(schema, dict):
    self.name = str(schema.get("title", f"response_format_{str(uuid.uuid4())[:4]}"))
else:
    self.name = str(getattr(schema, "__name__", f"response_format_{str(uuid.uuid4())[:4]}"))
```

优先用显式 `name=`；dict 用 `title` 字段；类用 `__name__`；**都没有就用 uuid 前 4 位兜底**。
→ 这解释了一个常见困惑：为什么你传 `dict` schema 又没写 `title` 时，日志里的工具名是 `response_format_a3f1` 这种东西。

**第二步，定 description**（L167-171）：

```python
self.description = description or (
    schema.get("description", "")
    if isinstance(schema, dict)
    else getattr(schema, "__doc__", None) or ""
)
```

→ **类（Pydantic/dataclass/TypedDict）的 `description` 来自 `__doc__`，也就是你写的 docstring**。
这就是为什么官方文档反复强调"给 schema 写 docstring"——它不是装饰，它是最终发给模型的那段工具描述。

**第三步，判 kind + 生成 json_schema**（L175-192），是个 if/elif 瀑布：

```python
if isinstance(schema, dict):
    self.schema_kind = "json_schema"
    self.json_schema = schema
elif isinstance(schema, type) and issubclass(schema, BaseModel):
    self.schema_kind = "pydantic"
    self.json_schema = schema.model_json_schema()
elif is_dataclass(schema):
    self.schema_kind = "dataclass"
    self.json_schema = TypeAdapter(schema).json_schema()
elif is_typeddict(schema):
    self.schema_kind = "typeddict"
    self.json_schema = TypeAdapter(schema).json_schema()
else:
    msg = (
        f"Unsupported schema type: {type(schema)}. "
        f"Supported types: Pydantic models, dataclasses, TypedDicts, and JSON schema dicts."
    )
    raise ValueError(msg)
```

三条要点：

1. **Pydantic 走 `model_json_schema()`**（Pydantic 自己的方法）；
2. **dataclass / TypedDict 走 `TypeAdapter(...).json_schema()`**（Pydantic 的通用适配器）；
3. **dict 原样不动**——它就是 JSON Schema，不再生成。

注意 `is_dataclass(schema)` 和 `is_typeddict(schema)` 都要求 `schema` 是**类型对象**，传实例会直接掉进最后的 `ValueError`。

### 2.3 `_parse_with_schema`：回程校验的唯一入口

L77-103，全模块**唯一**做"数据 → Python 对象"转换的函数：

```python
def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any], schema_kind: SchemaKind, data: dict[str, Any]
) -> SchemaT | dict[str, Any]:
    if schema_kind == "json_schema":
        # Raw JSON schema has no corresponding Python type to instantiate.
        return data
    try:
        adapter = TypeAdapter[SchemaT](schema)
        return adapter.validate_python(data)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        msg = f"Failed to parse data to {schema_name}: {e}"
        raise ValueError(msg) from e
```

**第一句就是全章最大的坑**：

```python
if schema_kind == "json_schema":
    return data
```

传原始 dict 时，**数据原样返回，不做任何校验**。所以传 dict schema + `handle_errors=True`，等于什么防护都没有。

这段话在 `ToolStrategy.handle_errors` 的 docstring 里被专门写成了 warning（L223-230）：

```
!!! warning "Raw JSON schema dicts are not validated"
    When `schema` is a raw JSON schema `dict` ..., the model's tool-call arguments
    are returned **as-is without validation** against the schema.

    As a result, `handle_errors` is effectively inert for dict schemas. To get
    validation and automatic retries, express the schema as a Pydantic model,
    `dataclass`, or `TypedDict` instead.
```

非 dict 的情况走 `TypeAdapter(schema).validate_python(data)`——**Pydantic v2 的通用校验器，对 Pydantic 模型、dataclass、TypedDict 都统一**。校验失败包成 `ValueError("Failed to parse data to <Name>: ...")`。

还有一个细节：`TypeAdapter[SchemaT](schema)` 用了**下标语法**（`TypeAdapter[X]` 而不是 `TypeAdapter(X)`）。这是给类型检查器看的，运行期 `TypeAdapter.__class_getitem__` 会返回 `TypeAdapter` 本身，所以 `TypeAdapter[X](schema)` 等价于 `TypeAdapter(schema)`。

### 2.4 三个异常类：错误是有名字的

L35-74，三个异常构成一条继承链：

```python
class StructuredOutputError(Exception):
    ai_message: AIMessage

class MultipleStructuredOutputsError(StructuredOutputError):
    # 模型一次回了多个结构化工具调用，但只允许一个
    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None: ...

class StructuredOutputValidationError(StructuredOutputError):
    # 工具调用参数不符合 schema
    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
        super().__init__(f"Failed to parse structured output for tool '{tool_name}': {source}.")
```

两个子类都**把触发的 `AIMessage` 存下来了**（`self.ai_message`）。这不是随手记，是给上层重试逻辑用的——重试时要把这条模型消息 + 一条错误 ToolMessage 一起塞回对话。

- `MultipleStructuredOutputsError`：对应 `ToolStrategy` 展开 Union 后产生**多个**假工具，模型贪心调了两个。
- `StructuredOutputValidationError`：对应 `_parse_with_schema` 抛的 `ValueError` 被包了一层。

### 2.5 `ToolStrategy`：把 schema 做成"假工具"

L195-267。先看 docstring 里 `handle_errors` 的六种取值（L210-221）：

| 取值 | 语义 |
|---|---|
| `True` | 捕获所有错误，用默认错误模板 |
| `str` | 捕获所有错误，用你给的自定义消息 |
| `type[Exception]` | 只捕获这个异常类型，用默认消息 |
| `tuple[type[Exception], ...]` | 只捕获这几种，用默认消息 |
| `Callable[[Exception], str]` | 自定义函数，返回错误消息 |
| `False` | 不重试，异常直接向上抛 |

构造函数（L241-251，`@dataclass(init=False)` 所以手写 `__init__`）：

```python
def __init__(
    self,
    schema: type[SchemaT] | UnionType | dict[str, Any],
    *,
    tool_message_content: str | None = None,
    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    ) = True,
) -> None:
```

注意 `schema` 类型里那个 **`UnionType`**——这是 `ToolStrategy` 独有的能力：**允许 A|B 联合 schema**。

它靠内部函数 `_iter_variants`（L253-265）实现：

```python
def _iter_variants(schema: Any) -> Iterator[Any]:
    if get_origin(schema) in {UnionType, Union}:
        for arg in get_args(schema):
            yield from _iter_variants(arg)
        return
    if isinstance(schema, dict) and "oneOf" in schema:
        for sub in schema.get("oneOf", []):
            yield from _iter_variants(sub)
        return
    yield schema
```

两层递归展开：

1. **Python 的 `A | B`**（`types.UnionType`）或 `typing.Union[A, B]` → 拆成 A、B；
2. **JSON Schema 的 `{"oneOf": [...]}`** → 拆成每个子 schema。

然后：

```python
self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]
```

**一个 Union → 多个 `_SchemaSpec` → 多个"假工具"。** 这就是为什么会有 `MultipleStructuredOutputsError`：模型面前摆着两个工具，它有可能两个都调。

> **对比**：`ProviderStrategy` 不接受 UnionType（它的 `schema` 类型只有 `type[SchemaT] | dict`）。原生 JSON Schema 通道一次只能表达一个 schema。

### 2.6 `OutputToolBinding`：假工具长什么样

L317-369，`@dataclass`（普通 dataclass，有自动 `__init__`）：

```python
schema: type[SchemaT] | dict[str, Any]
schema_kind: SchemaKind
tool: BaseTool
```

工厂方法 `from_schema_spec`（L337-355）：

```python
@classmethod
def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> OutputToolBinding[SchemaT]:
    return cls(
        schema=schema_spec.schema,
        schema_kind=schema_spec.schema_kind,
        tool=StructuredTool(
            args_schema=schema_spec.json_schema,
            name=schema_spec.name,
            description=schema_spec.description,
        ),
    )
```

**这里就是"假工具"诞生的地方**：一个 `StructuredTool`，`args_schema` 直接吃 `json_schema`（原始 JSON Schema dict），`name` 和 `description` 来自 `_SchemaSpec`。**它没有 `func`、没有 `coroutine`**——因为永远不会被真正执行，它只是给模型看的一个"函数签名贴纸"。

回程解析（L357-369）：

```python
def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
    return _parse_with_schema(self.schema, self.schema_kind, tool_args)
```

模型以 `tool_call` 的形式回数据 → `tool_call["args"]` 就是 `tool_args` → 交给 `_parse_with_schema`。

### 2.7 `ProviderStrategy`：不建工具，直接改请求体

L270-314，`@dataclass(init=False)`：

```python
@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_spec: _SchemaSpec[SchemaT]

    def __init__(self, schema: type[SchemaT] | dict[str, Any], *, strict: bool | None = None) -> None:
        self.schema = schema
        self.schema_spec = _SchemaSpec(schema, strict=strict)
```

**只有 `schema` 和 `schema_spec` 两个字段，没有 `tool`。** 构造时只做一次 `_SchemaSpec`，连工具都不建。

核心方法 `to_model_kwargs()`（L295-314）：

```python
def to_model_kwargs(self) -> dict[str, Any]:
    json_schema: dict[str, Any] = {
        "name": self.schema_spec.name,
        "schema": self.schema_spec.json_schema,
    }
    if self.schema_spec.strict:
        json_schema["strict"] = True
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": json_schema,
    }
    return {"response_format": response_format}
```

产出的 dict 直接喂给 `model.bind(**kwargs)`，最终拼进 HTTP 请求体：

> **实测修正（1.4.6）**：调用点是 `request.model.bind_tools(final_tools, **bind_kwargs)`，不是 `model.bind(**kwargs)`——provider 通道也走 `bind_tools`，只是 `final_tools` 里没有假工具。`to_model_kwargs()` 的产物本身仍如上一段所述。

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "Person",
      "schema": { "type": "object", "properties": {...}, "required": [...] },
      "strict": true
    }
  }
}
```

这就是 OpenAI 的 **Structured Outputs** 协议。注意三点：

1. `strict` 只在**真值**时出现（`if self.schema_spec.strict:`），`None` 或 `False` 时**整个字段不出现**——不是发 `"strict": false`。
2. `name` 是必填的，来自 `_SchemaSpec.name`。
3. **`response_format` 是模型 provider 的参数，不是 Agent 参数**。它和 `tool_choice` / `temperature` 平级。

### 2.8 `ProviderStrategyBinding`：从 `AIMessage` 里抠出 JSON

L372-454。

```python
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> ProviderStrategyBinding[SchemaT]:
        return cls(schema=schema_spec.schema, schema_kind=schema_spec.schema_kind)
```

**注意对比**：`OutputToolBinding.from_schema_spec` 建 `StructuredTool`；`ProviderStrategyBinding.from_schema_spec` **只存 `schema` 和 `schema_kind`，不建任何工具**。

回程 `parse(self, response: AIMessage)`（L404-430）三步：

1. `raw_text = self._extract_text_content_from_message(response)` —— 从消息里抽出纯文本
2. `data = json.loads(raw_text)` —— 解析 JSON，失败抛 `ValueError(f"Native structured output expected valid JSON for {schema_name}, but parsing failed: {e}.")`
3. `return _parse_with_schema(self.schema, self.schema_kind, data)` —— 复用同一个校验入口

`_extract_text_content_from_message`（L432-454）是静态方法，逻辑是：`content` 是 `str` 就直接返回；否则遍历 parts，`dict` 且 `type == "text"` 取 `text`，`dict` 且有字符串 `content` 取 `content`，其余 `str(part)` 兜底；最后 `"".join(parts)`。

→ 这一步的存在说明：**ProviderStrategy 的回程是"读文本"，不是"读 tool_calls"**。这是两条通道最本质的差别。

### 2.9 `AutoStrategy`：一个只有字段的类

L457-468，几行就完了：

```python
class AutoStrategy(Generic[SchemaT]):
    def __init__(self, schema: type[SchemaT] | dict[str, Any]) -> None:
        self.schema = schema
```

**普通 class，不是 dataclass，没有任何方法。** 它不建工具、不产 kwargs、不解析——它唯一的作用是**在 `factory.py` 里当一个"待决"标记**。

为什么需要它？因为**决定用哪条通道要看模型能力，而看模型能力必须等到运行期**（`model.profile` 是 bind 之后才知道的）。所以创建期只能先把状态记成"未定"，运行期再落地。

### 2.10 `factory.py` 是怎么消费它的：创建期 + 运行期

`create_agent` 处理 `response_format` 分两个阶段。

**阶段一：创建期（factory L1063-1088）**

```python
if response_format is None:
    initial_response_format = None
elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
    initial_response_format = response_format
else:
    # Raw schema - wrap in AutoStrategy to enable auto-detection
    initial_response_format = AutoStrategy(schema=response_format)

tool_strategy_for_setup: ToolStrategy[Any] | None = None
if isinstance(initial_response_format, AutoStrategy):
    tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
elif isinstance(initial_response_format, ToolStrategy):
    tool_strategy_for_setup = initial_response_format

structured_output_tools: dict[str, OutputToolBinding[Any]] = {}
```

**关键动作**：`AutoStrategy` 被"预支"成一个 `ToolStrategy`，用来**提前把假工具建出来**。

注释写得很直白（L1076-1077）：

```python
# For AutoStrategy, convert to ToolStrategy to setup tools upfront
# (may be replaced with ProviderStrategy later based on model)
```

→ 所以 `AutoStrategy` 的实际执行顺序是：**先按 ToolStrategy 把工具注册好 → 运行期发现模型支持原生结构化输出 → 换成 ProviderStrategy → 那批工具白建了**。

这不是浪费，是**必然**：因为 `create_agent` 要在编译期就把节点、边、图建好，而图的形状里包含"结构化输出工具可用于 tool_choice"。等到运行期才知道不需要它们，已经改不了图了。

**阶段二：运行期 `_get_bound_model`（factory L1399-1490）**

```python
response_format = request.response_format
if response_format is not None and not isinstance(
    response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)
):
    response_format = AutoStrategy(schema=response_format)

if isinstance(response_format, AutoStrategy):
    if model_profile_supports_structured_output:   # 见 L604 model_profile.get("structured_output")
        effective_response_format = ProviderStrategy(schema=response_format.schema)
    elif response_format is initial_response_format and tool_strategy_for_setup is not None:
        effective_response_format = tool_strategy_for_setup
    else:
        effective_response_format = ToolStrategy(schema=response_format.schema)
else:
    effective_response_format = response_format
```

三路决策：

1. **模型 profile 说支持 `structured_output`** → 换成 `ProviderStrategy(schema=同一个 schema)`
2. **不支持，且这是创建期那个 AutoStrategy** → 复用创建期已经建好的 `tool_strategy_for_setup`（工具不白建）
3. **不支持，且是运行期新来的 AutoStrategy** → 现场建一个 `ToolStrategy`

然后按策略分别改请求：

**ProviderStrategy 分支（L1443-1455）**：

```python
kwargs = effective_response_format.to_model_kwargs()
model_ = model_.bind(**kwargs)
```

**ToolStrategy 分支（L1458-1480）**：

```python
for tc in effective_response_format.schema_specs:
    if tc.name not in structured_output_tools:
        raise ValueError(
            f"ToolStrategy specifies tool '{tc.name}' "
            ...
        )

tool_choice = "any" if structured_output_tools else request.tool_choice
model_ = model_.bind_tools(tools, tool_choice=tool_choice)
```

关键两点：

- **`tool_choice="any"`**：强制模型**必须**调用至少一个工具。这就是"假工具"能被稳定触发的机制。
- **校验 schema_specs 都在 `structured_output_tools` 里**：如果 middleware 把一个 Union 缩小到了某个子集，这里能接住，因为创建期已经把所有变体都注册了。

### 2.11 `_handle_model_output`：回程三分支

factory L1246-1348。

**ProviderStrategy 分支（L1257-1272）**：建一个 `ProviderStrategyBinding.from_schema_spec(effective_response_format.schema_spec)`，调 `.parse(output)`，成功就把结果塞进 `structured_response`。

**ToolStrategy 分支（L1276-1310+）**：

```python
structured_tool_calls = [
    tc for tc in output.tool_calls if tc["name"] in structured_output_tools
]
```

把 `tool_calls` 里**名字属于假工具集**的挑出来，然后：

- **0 个** → 没触发结构化输出（正常回答），`structured_response` 不设
- **多个** → 抛 `MultipleStructuredOutputsError`
- **1 个** → 取 `structured_output_tools[tc["name"]].parse(tc["args"])`，成功则设 `structured_response`，并**追加一条 `ToolMessage`**（内容来自 `effective_response_format.tool_message_content`）——因为 OpenAI 协议要求 tool_call 必须有对应 tool 回复

**统一错误处理 `_handle_structured_output_error`（L646-660）**：

```python
if not isinstance(response_format, ToolStrategy):
    ...   # ProviderStrategy 直接不重试
handle_errors = response_format.handle_errors
...
```

→ **ProviderStrategy 没有重试机制**。它的解析失败会直接往上抛（除非外层 middleware 兜）。
→ 只有 `ToolStrategy` 会读 `handle_errors`，把错误按六种取值翻译成一条错误消息，然后**把错误写成 `ToolMessage` 塞回消息列表，让 agent 循环再问一次模型**。

> **重试次数的诚实说明**：factory 里没有硬编码的"最多重试 N 次"常量；重试靠 agent 循环驱动。实际能重试几轮，由 langgraph 的 `recursion_limit`（默认 25）兜底。这一点是 `[源码推导]`，未实测。

---

## ③ 配套代码

文件：`langchain_learn/07_structured_output.py`

A–E 段**零 token 消耗**（不调模型，纯读源码/构造对象），F–G 段才发真实请求。

```bash
# 只看离线部分
python langchain_learn/07_structured_output.py

# 含真实调用
python langchain_learn/07_structured_output.py --real
```

### 段落地图

| 段 | 内容 | 需要模型 |
|---|---|---|
| A | 手工构造 `_SchemaSpec`，看四种 schema 的 `json_schema` 差异 | ✗ |
| B | `ToolStrategy` 的 `schema_specs`：单 schema vs Union 展开 | ✗ |
| C | `OutputToolBinding` 造出的"假工具"：name/description/args_schema | ✗ |
| D | `ProviderStrategy.to_model_kwargs()` 的产物直接打印 | ✗ |
| E | `AutoStrategy` 只有一个字段的证据 | ✗ |
| F | 真实调用：`response_format=Schema` → `result["structured_response"]` | ✓ |
| G | 破坏实验：`handle_errors` 三种取值的不同反应 | ✓ |

附录对照代码（A–H 零 token，I 段 `--real`）：

```bash
python langchain_learn/07_official_structured_output.py
python langchain_learn/07_official_structured_output.py --real
```

### 手敲时的三个提醒

1. **`_SchemaSpec`、`OutputToolBinding` 这些带下划线的是私有 API**。本章用它们只是为了"看见内部"，**生产代码不要 import**。生产只用 `ToolStrategy` / `ProviderStrategy` / 裸 schema。
2. **`structured_response` 不一定存在**。它只在模型真的给出了结构化输出时才被塞进 state。取值请用 `result.get("structured_response")`，不要 `result["structured_response"]`。
3. **`response_format` 传类，不传实例**。传 `Person` 不是 `Person()`——传实例会一路掉到 `_SchemaSpec` 的 `ValueError: Unsupported schema type`。

---

## ④ 破坏实验

每个实验都先**写下你的预测**，再改代码，再看结果。

### 实验 1：把 `handle_errors` 关掉，看异常长什么样

把 F 段的 `response_format=Person` 改成 `response_format=ToolStrategy(schema=Person, handle_errors=False)`，同时把 schema 的某个字段改成模型很难满足的类型（比如 `age: int = Field(ge=999)`）。

**预测**：______（先写）

**看什么**：异常类型是不是 `StructuredOutputValidationError` → 里面 `.source` 是 Pydantic 的 `ValidationError`，`.ai_message` 是那条坏消息。

**要点**：`handle_errors=False` 不是"不校验"，是"校验失败就抛"。

### 实验 2：用原始 dict schema，验证 `handle_errors` 失效

```python
BROKEN = {"title": "Person", "type": "object", "properties": {"name": {"type": "string"}}}
result = agent.invoke({"messages": [...]}, config)  # response_format=ToolStrategy(schema=BROKEN, handle_errors=True)
```

让模型返回 `{"name": 123}`（类型不对）。

**预测**：______

**看什么**：`structured_response` 是不是 `{"name": 123}` **原样**。这就是 §2.3 那句 `if schema_kind == "json_schema": return data` 的现场。

### 实验 3：Union schema 引入 `MultipleStructuredOutputsError`

```python
response_format=ToolStrategy(schema=Person | Company)
```

问一个两者都沾边的模糊问题，并显式提示"如果同时符合就都给我"。

**看什么**：能不能逼出 `MultipleStructuredOutputsError`。注意——**能不能逼出来取决于模型**，逼不出来也算实验成功（记录"我的模型没有贪心"）。

### 实验 4：`strict=True` 有没有真的进请求体

把 `ProviderStrategy(schema=Person, strict=True)` 和 `strict=None` 两次分别跑，用 `agent.get_graph()` 无关，关键是**用 `model.bind` 打出来的 kwargs**：

```python
print(ProviderStrategy(schema=Person, strict=True).to_model_kwargs())
print(ProviderStrategy(schema=Person, strict=None).to_model_kwargs())
```

**预测**：第二个是不是 `"strict": false`？

**看什么**：第二个**根本没有 `strict` 这个 key**。`if self.schema_spec.strict:` 只认真值。

---

## ⑤ 自检题（答案内联，别折叠）

**Q1.** `create_agent(response_format=Person)` 其中 `Person` 是 Pydantic 模型。运行期如果模型不支持原生结构化输出，最终发出去的是 `bind_tools` 还是 `bind(response_format=...)`？

**A1.** `bind_tools`。`AutoStrategy` 在 `_get_bound_model` 里落到 `tool_strategy_for_setup`（创建期预建的 `ToolStrategy`），走 ToolStrategy 分支：`bind_tools(tools, tool_choice="any")`。**只有 `_supports_provider_strategy` 判真时才走 provider 分支**；provider 分支在本机 1.4.6 也是 `bind_tools(final_tools, response_format=...)`，不是 `bind(response_format=...)`（见附录 A7.2）。

**Q2.** `ToolStrategy` 生成的假工具，`args_schema` 的类型是 `BaseModel` 子类吗？

**A2.** 不是。是**原始 JSON Schema dict**——`StructuredTool(args_schema=schema_spec.json_schema, ...)`。LangChain 的 `StructuredTool` 允许 `args_schema` 直接吃 dict，但这样就没有 Pydantic 校验，这也是 §2.3 warning 的根源。

**Q3.** `AutoStrategy` 为什么会"先建工具后丢弃"？能不能优化掉？

**A3.** 不能。`create_agent` 是**编译期建图**：节点、边、`tool_choice` 都要在建图时确定。而"模型是否支持原生结构化输出"要等到 `model` 被 bind、读到 `model.profile` 才知道（运行期）。所以只能先把工具建好（ToolStrategy 形态），运行期再决定用不用。丢弃的那批工具只是 `StructuredTool` 对象，不产生额外网络开销。

**Q4.** `ProviderStrategyBinding.parse` 和 `OutputToolBinding.parse` 的入参类型有什么本质不同？为什么？

**A4.** `OutputToolBinding.parse(tool_args: dict[str, Any])` 吃的是**已经解析好的 dict**（来自 `tool_call["args"]`）；`ProviderStrategyBinding.parse(response: AIMessage)` 吃的是**整条消息**，内部自己 `_extract_text_content_from_message` + `json.loads`。因为两条通道的回收形态不同：一个是结构化字段，一个是纯文本。

**Q5.** 如果模型既没调假工具、也没给 JSON，`structured_response` 会是什么？

**A5.** **`structured_response` 存在，值是 `None`**（1.4.6 实测，修正原先"不存在这个 key"的说法）。`_build_commands(has_structured_output=True)` 会显式写 `None`，理由是防止读到上一轮 checkpoint 里的旧值。所以 `result.get(...)` 安全，但 `"structured_response" in result` 恒为真、不能当判据——要判就判 `is not None`。见附录 A7.7 ②。

---

## ⑥ 交给你自己

1. 把 §2.5 的 `_iter_variants` 抄一遍（手抄，不复制），然后**只用注释**写出它对 `A | B | C` 的递归轨迹。
2. 打开 `.venv/Lib/site-packages/langchain/agents/factory.py`，搜 `structured_output_tools`，数一数它被读了几次、分别在哪一层。你会发现"假工具"的生命周期贯穿了建图、bind、回程解析三个阶段。
3. 写一句话总结「`ToolStrategy` 和 `ProviderStrategy` 的根本差别」。参考："**前者把 schema 伪装成一个函数让模型调用，后者把 schema 塞进请求头让模型遵守。**"

---

## ⑦ 延伸阅读 / 诚实边界

**官方**
- Structured output 概念页：`/oss/python/langchain/structured-output`
- `with_structured_output`（旧 API，`BaseChatModel` 上的方法）——它和本章的 `response_format` 是**两套**东西：前者是单次调用的便捷封装，后者是 Agent 级别的状态字段。别混。

**源码**
- `langchain/agents/structured_output.py` —— 472 行，全章主体
- `langchain/agents/factory.py` L1063-1100（创建期）、L1246-1348（回程）、L1399-1490（bind 期）
- `langchain_core/utils/function_calling.py` 的 `convert_to_openai_tool` —— 第 3 章讲过，ToolStrategy 的假工具最终也走它

**诚实边界**
- 正文（②–⑦）的 `print` 仍是 `[源码推导]`；**附录 A7.1–A7.7 是实测**，代码为 `langchain_learn/07_official_structured_output.py`（`langchain 1.4.6 / core 1.6.12`，零 token 假模型，`--real` 才联网）。
- 「重试次数由 `recursion_limit` 兜底」已在附录 A7.7 ③ 实测：`tool_choice="any"` 下模型不调工具会撞 `GraphRecursionError`（`recursion_limit=6` 时模型被 bind 6 次）。
- 附录实测修正了正文三处：provider 通道也是 `bind_tools`（非 `model.bind`）、`structured_response` 缺失时是 `None`（非缺 key）、`ProviderStrategy` 不会回退。
- 「能否逼出 `MultipleStructuredOutputsError`」强依赖具体模型，不作保证。
- 附录 I 段（`--real`）本机**未跑通**：写这一节时 LLM 服务没起（`OpenAIAPIError: upstream connect failed ... 10061`），所以附录里没有任何真实模型输出。A–H 的结论不依赖它——全部来自离线假模型 + 直接调用判定函数。
- `model.profile` 里 `structured_output` 这个 key 的具体来源（`langchain_core` 的 model profile 机制）本章未展开，属于第 3 章 Model 层的延伸。

**下一章**：ch08 = 中间件（middleware）—— `create_agent` 的 `middleware=[...]` 如何插进 `model` / `tools` / `exit` 三类节点前后。

---

## 附录 · 官方写法对照（structured-output 页）

> 官方锚点：<https://docs.langchain.com/oss/python/langchain/structured-output>
> 代码：`langchain_learn/07_official_structured_output.py`（A–H 全离线，假模型只记 `bind_tools` 收到了什么；`--real` 才真发请求）

正文拆的是 `langchain/agents/structured_output.py` 那 472 行怎么实现；这一节换成官方视角：**官方页只讲"怎么选、怎么写、错了怎么办"，一行内部实现都不给**。但它有几条结论在本机 `langchain 1.4.6` 上对不上——下面每一条都是实测，不是推演。

| 官方小节 | 它回答的问题 | 代码段 |
|---|---|---|
| Response format | 四种取值各自变成什么 | A |
| 自动选择 / 模型 profile | 什么时候走 provider、什么时候走工具 | A |
| Provider strategy | 请求体长什么样 | B |
| Tool calling strategy | 请求体长什么样、为什么模型必须调工具 | B |
| Custom tool message content | 那条 ToolMessage 到底说什么 | D |
| Error handling | 六种 `handle_errors` 各走哪条路 | E / F |
| Schema 支持 | 四种 schema 各返回什么 | C |
| （官方没写） | 裸 dict、随机 name、`None`、死循环 | G |

### A7.1 "自动选择"不是一个类，是一次运行期判定

官方签名只有四个分支，实测归一化后是：

```text
  None                   → NoneType
  裸 Pydantic 类           → AutoStrategy
  裸 JSON Schema dict     → AutoStrategy
  ToolStrategy           → ToolStrategy
  ProviderStrategy       → ProviderStrategy
```

也就是说 `AutoStrategy` 只在"你传了裸 schema"时出现一次，随后由 `_supports_provider_strategy(model, tools)` 决定落哪边。这个函数是纯查表，可以离线跑——下面七行全是实测：

```text
  profile                                模型名              有工具    判定
  None                                   scripted         False  False
  {'structured_output': True}            scripted         False  True
  {'structured_output': False}           scripted         False  False
  None                                   gpt-5.5          False  True
  {'structured_output': False}           gpt-5.5          False  True
  {'structured_output': True}            gemini-2.5-pro   True   False
  {'structured_output': True}            gemini-3-pro     True   True
```

三行是官方没写的：

- **`profile` 为 `None` 时还有模型名白名单兜底**（`factory.py` 的 `FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT`）。`gpt-4.1 / 4o / 5.x`、`claude-sonnet-4-5/4-6`、`grok-4` 都在里面。你把 base_url 指向兼容网关、profile 为空，只要 `model_name` 像这些名字，框架就认为它支持原生结构化输出。
- **`profile` 显式写 `False` 也挡不住白名单**。判定代码是 `profile and profile.get("structured_output")`，假值直接掉到名字匹配——第 5 行就是这么拿到 `True` 的。想强制走工具通道，得传明确的 `ToolStrategy`，别指望 `profile=False`。
- **gemini 2.x 带工具就判 `False`**（不能同时用工具和原生结构化输出），3 系才放行。这是官方页注释里提过一句、但很容易漏的例外。

### A7.2 两条通道的差别，全在 `bind_tools` 的参数里

同一份 `Person`、同一个业务工具 `get_weather`，两条通道的绑定参数：

```text
  【ToolStrategy】   tools = ['get_weather', 'Person']   tool_choice = 'any'
  【ProviderStrategy】tools = ['get_weather']            tool_choice = None
                    response_format = {'type': 'json_schema', 'json_schema': {'name': 'Person', ...}}
```

- `ToolStrategy`：假工具**进** `tools` 列表，靠 `tool_choice="any"` 逼模型必须调一个；
- `ProviderStrategy`：假工具**不进** `tools`，schema 走 `response_format` 参数。

还有一处和正文不同：**1.4.6 里两条通道都用 `bind_tools(...)`**，provider 通道是 `bind_tools(final_tools, response_format=...)`，不是 `model.bind(**kwargs)`。正文 2.7 引的 `model_.bind(**kwargs)` 是基于更早的源码读的；行为等价，但你在本机设断点会看到 `bind_tools`。

### A7.3 四种 schema 的返回物：官方表有一格是错的

官方原话是"Pydantic → 实例；dataclass / TypedDict / JSON Schema → dict"。实测：

| schema | `schema_kind` | 返回类型 | 例子 |
|---|---|---|---|
| Pydantic | `pydantic` | `Person` 实例 | `Person(name='王五', age=41)` |
| dataclass | `dataclass` | **`Order` 实例** | `Order(order_id='A-1', amount=9.9)` |
| TypedDict | `typeddict` | `dict` | `{'title': '沙丘', 'year': 2021}` |
| JSON Schema | `json_schema` | `dict` | `{'city': '北京', 'celsius': 26}` |

**dataclass 返回的是实例，不是 dict。** 原因是回程走同一个 `_parse_with_schema`：只有 `json_schema` 在第一行 `return data`，其余三种都过 `TypeAdapter(...).validate_python(data)`——dataclass 有类型，于是被造成了对象。TypedDict 没有可实例化的类，只能返回 dict。

这条直接影响你的类型标注：写 `def f(x: dict) -> ...` 接 dataclass 的 `structured_response`，线上会在第一次属性访问时炸。

另外两条边界：

- **JSON Schema 完全不校验**（`return data` 在 `try` 之前）。类型不对也原样返回，`handle_errors` 对它是摆设。
- **JSON Schema 的 `name` 来自 `title`，没有 `title` 就现编**：`response_format_bb1e`（uuid 前 4 位，每次运行都不一样）。这个名字会进 `tools[].name` 和 `response_format.json_schema.name`，随机就意味着不可复现、不可评测。官方说"必须含 title 和 description"，真正的原因在这里，不在报错上。

### A7.4 `tool_message_content`：不填时那句话是框架拼的

```text
  tool_message_content=None           → "Returning structured response: name='赵六' age=18"
  tool_message_content='已记录到会议纪要！'  → '已记录到会议纪要！'
```

默认文案在 `factory.py` 里是 `f"Returning structured response: {structured_response}"`——把**解析后的对象 repr** 塞进 ToolMessage。两件事要注意：

1. 这个对象会进消息历史，**下一轮还会发给模型**。字段多的时候是白送 token。
2. 它是英文模板 + Python repr，用户在前端看到的是 `name='赵六' age=18` 这种半截代码。产品里一律自己填。

### A7.5 `handle_errors`：六种写法归一成一个二元组

官方给了六种写法，没给判定顺序。本机归一化函数是 `_handle_structured_output_error(exception, response_format) -> (should_retry, message)`。拿一个 `StructuredOutputValidationError` 去喂：

```text
  handle_errors                      should_retry  回给模型的话
  True                               True          Error: Failed to parse structured output for...
  "请重试，age 必须 >= 0"              True          请重试，age 必须 >= 0
  ValueError                         False         （空）
  (ValueError, KeyError)             False         （空）
  custom_handler（可调用）             True          自定义处理：StructuredOutputValidationError
  False                              False         （空）
```

三个反直觉，都会在真实项目里咬人：

- **`handle_errors=ValueError` 对校验失败无效。** 抛出来的是 `StructuredOutputValidationError`，MRO 是 `(StructuredOutputError, Exception)`，`issubclass(..., ValueError)` 为 `False`。官方示例里那句 `handle_errors=ValueError  # Only retry on ValueError` 会让人以为它能接住 schema 校验——接不住。要接就写 `StructuredOutputValidationError`，或直接 `True`。
- **传 `False` 和传"没命中的类型"结果一样**：`(False, "")`，不重试、原样抛。区别只在语义，不在行为。
- **`ProviderStrategy` 一律 `(False, "")`**。它连 `handle_errors` 字段都没有——provider 通道没有"把错误塞回对话让模型改"这条路。要重试只能在 middleware 层自己包。

### A7.6 两类错误的现场：错误是当 ToolMessage 塞回去的

`ToolStrategy(Person | Order)`，剧本让模型第一轮两个都调、第二轮只调 `Person`：

```text
  轨迹：HumanMessage → AIMessage → ToolMessage → ToolMessage → AIMessage → ToolMessage
  第一轮 2 条错误回执：Error: Model incorrectly returned multiple structured responses (Person, Order)...
  重试后 structured_response = Person(name='张三', age=30)
```

**重试不是"重新问一遍"**，是把错误文本当成 ToolMessage 追加到同一条对话里，让模型在上下文里看到"我上一步错在哪"。这也解释了为什么重试会花 token：每一轮都在变长。

`handle_errors=False` + `age=-5` 时则是另一条路：

```text
  抛 StructuredOutputValidationError，.tool_name='Person'
  .source = ValueError：Failed to parse data to Person: 1 validation error for Person.age...
  .ai_message 是那条坏消息
```

`handle_errors=False` 不是"不校验"，是"校验失败就抛"。校验永远发生——它是 `_parse_with_schema` 的固定动作，不受 `handle_errors` 影响。

### A7.7 官方没写、本机会踩的四个坑

**① 官方说"JSON Schema 必须用显式策略包裹"，本机裸传也能跑。**

```text
  裸 dict → structured_response={'city': '北京', 'celsius': 26}，bind 到的工具=['Weather']
```

1.4.6 里裸 dict 会被包成 `AutoStrategy`，再落到 `ToolStrategy`。能跑，但拿到的是**未校验的 dict**。官方那句话的意图是对的：想要校验和重试，就别裸传——但别指望它会报错提醒你。

**② 没有产出结构化输出时，`structured_response` 是 `None`，不是"没有这个 key"。**

```text
  这一轮写回 state 的键 = ['messages', 'structured_response']
  update['structured_response'] = None
```

`_build_commands` 在 `has_structured_output=True` 时会显式写 `None`，为的是不让你读到**上一轮的旧值**（这个字段会被 checkpointer 存下来，第 6 章讲过）。所以 `result.get(...)` 安全，但 `if "structured_response" in result` 不能当判据。

**③ `tool_choice="any"` 是硬约束：模型不调工具，图会一直转。**

```text
  GraphRecursionError: Recursion limit of 6 reached without hitting a stop condition.
```

模型不肯调工具 → `model` 节点没有出口 → 撞 `recursion_limit` 才停。真模型很少这样，但假模型、小模型、被 system prompt 教坏的模型会。线上"结构化输出这一段特别慢/超时"，先看这里，别先怀疑网络。

**④ 显式写 `ProviderStrategy` 不会回退。**

官方说"若提供商不支持，agent 会回退到工具调用"。实测：这句只适用于**传裸 schema**（走 `AutoStrategy`）。显式 `ProviderStrategy` 时 `effective_response_format` 原样保留，即使 `profile=None` 也照发 `response_format` 参数：

```text
  profile=None 的模型 + 显式 Provider → bind kwargs=['response_format']
  structured_response=Person(name='张三', age=30)（没有 tool_choice，也没建假工具）
```

想要"不支持就自动退回工具调用"，就传裸 schema 让 `AutoStrategy` 去判。

### 认知修正

> 我原本以为"结构化输出"是一条路：schema 进去、对象出来，顶多换个通道。实际上它是**两条完全独立的管线**——ToolStrategy 把 schema 变成模型必须调用的假工具（`tool_choice="any"`，错误可以塞回对话重试），ProviderStrategy 把 schema 变成请求参数（没有假工具，没有重试）。`AutoStrategy` 既不是第三条路也不是兜底，它只是"把选择推迟到能读到 `model.profile` 的那一刻"。**先问"这次走哪条管线"，再问"schema 怎么写"——顺序反了，报错会指错方向。**

> 还有一条：官方文档的"返回 dict"和"会回退"这两句，在本机 1.4.6 上都不成立。**教材可以引官方，但判断要自己 print 一遍。**
