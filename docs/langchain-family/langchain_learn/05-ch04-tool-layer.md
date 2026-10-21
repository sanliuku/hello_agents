# 第 4 章 · Tool 层：从 `@tool` 到一个 JSON Schema

> 本章目标：**能画出 `@tool` 到 JSON Schema 的完整流水线，并独立诊断"模型为什么不调我的工具"**。预计 90 分钟。

---

## ① 本章你会得到什么

1. 能说出 `@tool` 装饰器背后的 **5 步调用链**（`tool()` → `_tool_factory` → `StructuredTool.from_function` → `create_schema_from_function` → `_create_subset_model`）
2. 能解释**两个 schema 的区别**：`get_input_schema()`（含注入参数，用于校验）与 `tool_call_schema`（剔除注入参数，发给模型）
3. 能解释 `description` 的**四级 fallback 顺序**，知道为什么"删掉 docstring 就炸"
4. 能让工具**优雅失败**（`ToolException` + `handle_tool_error`）而不是炸掉整个 agent 循环
5. 能用 `response_format="content_and_artifact"` 把大结果拆成"模型看的摘要"+"下游用的原始数据"（接第 2 章）
6. 能用 `ToolRuntime` / `InjectedToolCallId` 拿到运行时信息，且**它们不会出现在发给模型的 schema 里**
7. 拥有一张**"模型不调工具"的四类根因排查表**，以后不用猜

第 2、3、7 条是分水岭：绝大多数"agent 不工作"的问题都落在 tool 层，而 tool 层的问题 90% 是 schema 问题——不是模型笨。

---

## ② 源码解剖

本章只碰三个文件，全在 `libs/core/langchain_core/tools/`：

| 文件 | 里面是什么 | 一句话 |
|---|---|---|
| `base.py` | `BaseTool`、`create_schema_from_function`、`InjectedToolArg`、`_format_output` | 工具能"被调用"的全部机制 |
| `convert.py` | `tool()` 装饰器 | 决定"函数怎么变成工具"的分发器 |
| `structured.py` | `StructuredTool`、`from_function` | 真正承载"多参数函数"的工具类 |

### 2.0 先看类层次（这决定了你 `isinstance` 判断写错就会漏判）

```
BaseTool(RunnableSerializable)        ← 抽象基类，定义 run/invoke/_run/_parse_input
├── Tool                              ← simple.py，只支持 str → str
└── StructuredTool                    ← structured.py，支持任意参数
```

`@tool` 默认给你的是 **`StructuredTool`**；只有当你写 `infer_schema=False` 时才退回 `Tool`（简单字符串工具）。这个分支在 `convert.py` 里是一行 `if`，下面 §2.2 会看到。

`BaseTool` 继承 `RunnableSerializable` 这点很重要：**工具本身就是一个 Runnable**，所以它能 `.invoke()`、能 `.batch()`、能进 LCEL 管道、能被 `bind_tools` 之外的方式组合。你在第 1 章看到的 `ToolNode` 之所以能直接跑工具，就是因为这个继承关系。

### 2.1 `BaseTool` 的 12 个字段：哪些是给模型看的，哪些是给框架用的

源码 `base.py` 逐字摘录（只留字段声明与关键字注释）：

```python
class BaseTool(RunnableSerializable[str | dict[str, Any] | ToolCall, Any]):
    name: str
    """The unique name of the tool that clearly communicates its purpose."""

    description: str
    """Used to tell the model how/when/why to use the tool.
    You can provide few-shot examples as a part of the description."""

    args_schema: Annotated[ArgsSchema | None, SkipValidation()] = Field(default=None, ...)

    return_direct: bool = False
    """Setting this to `True` means that after the tool is called, the `AgentExecutor` will
    stop looping."""

    verbose: bool = False
    callbacks: Callbacks = Field(default=None, exclude=True)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    handle_tool_error: bool | str | Callable[[ToolException], ToolOutput] | None = False
    handle_validation_error: bool | str | Callable[[ValidationError | ValidationErrorV1], str] | None = False

    response_format: Literal["content", "content_and_artifact"] = "content"

    extras: dict[str, Any] | None = None
    """Optional provider-specific extra fields for the tool.
    Example: Anthropic-specific fields like `cache_control`, `defer_loading`, or `input_examples`."""
```

按"归谁管"分类，这是全章最省事的记忆表：

| 字段 | 谁消费 | 用途 |
|---|---|---|
| `name` | **模型 + 框架** | 模型返回的 `tool_calls[].name` 靠它匹配；也进 schema |
| `description` | **模型** | "什么时候该用我"——最重要的一行文本 |
| `args_schema` | **模型 + 校验** | 参数 schema；也是 `_parse_input` 的校验依据 |
| `return_direct` | 框架（图装配） | `True` 则工具结果直接结束循环（详见 §2.7 的边界） |
| `response_format` | 框架（`_format_output`） | `content` 还是 `(content, artifact)` 二元组 |
| `handle_tool_error` / `handle_validation_error` | 框架（`run()` 的 except） | 出错后"抛"还是"变成观测结果" |
| `callbacks` / `tags` / `metadata` / `verbose` | 框架（tracing） | 观测，与模型无关 |
| `extras` | **供应商适配层** | 塞供应商私有字段（Anthropic 的 `cache_control` 等） |

记住一句话：**前 3 个字段决定模型会不会、能不能正确调你；后 9 个决定框架怎么跑、怎么记录、怎么报错。**

还有一个隐藏检查：`__init_subclass__` 会在**定义子类时**检查 `args_schema` 的类型注解。你写

```python
class MyTool(BaseTool):
    args_schema: BaseModel = SchemaClass     # 错了
```

会在**类创建那一刻**抛 `SchemaAnnotationError`，提示必须是 `Type[BaseModel]`。这是框架里少见的"编译期"检查，专门防一个常见手误。

### 2.2 `@tool` 装饰器：四种写法、一个分发器

`convert.py` 里 `tool()` 有 4 个 `@overload`，对应 4 种用法：

| 写法 | 走哪条分支 | 结果类型 |
|---|---|---|
| `@tool` | `name_or_callable` 是 callable，取 `__name__` | `StructuredTool` |
| `@tool("my_name")` | `name_or_callable` 是 str → 返回 `_create_tool_factory` | `StructuredTool` |
| `@tool(parse_docstring=True)` | 无参调用 → 返回 `_partial` | `StructuredTool` |
| `@tool("name", runnable)` | `runnable is not None` | `StructuredTool`（包 Runnable） |
| `@tool(infer_schema=False)` | `_tool_factory` 里的 `if infer_schema or args_schema is not None` 为假 | **`Tool`**（str→str） |

最后一行是唯一会改变返回类型的开关，源码注释说得很直白：

```python
# If someone doesn't want a schema applied, we must treat it as
# a simple string->string function
tool_description = tool_description or dec_func.__doc__
if tool_description is None:
    msg = (
        "Function must have either a docstring or description "
        "when infer_schema is False."
    )
    raise ValueError(msg)
```

**注意 `infer_schema=False` 时连 `args_schema` 都不用了**——模型只能给你一个字符串，工具内部自己解析。除非你在对接一个天然单输入的东西（比如 SQL 查询串），否则别关它。

装饰器里还有一段 `!! warning "Reserved argument names"`，这是本章最有价值的一份"避坑文书"，逐字摘录：

> Avoid naming tool arguments `config`, `run_manager`, or `callbacks`. These collide with values LangChain injects into tools at call time. A `config` argument is shadowed by the injected `RunnableConfig`, so the tool fails at call time with `TypeError: ... missing 1 required positional argument: 'config'`, while `run_manager` and `callbacks` are silently dropped from the generated schema. To access runtime information (state, context, store, config), add a `ToolRuntime` parameter instead.

三种参数名、三种死法：`config` **运行时炸**，`run_manager` / `callbacks` **静默消失**。静默消失那种最毒——你的 schema 里少了一个字段，模型永远不知道要传它，你还以为是模型不配合。

### 2.3 `StructuredTool.from_function`：拼装的最后一步

`structured.py` 的 `from_function` 干了三件事：**定 schema、定 description、造对象**。

```python
name = name or source_function.__name__
if args_schema is None and infer_schema:
    args_schema = create_schema_from_function(
        name,
        source_function,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
        filter_args=_filter_schema_args(source_function),
    )
```

`_filter_schema_args` 就是你以后排查"我的参数为什么没进 schema"的第一现场：

```python
def _filter_schema_args(func: Callable[..., Any]) -> list[str]:
    filter_args = list(FILTERED_ARGS)          # ("run_manager", "callbacks")
    if config_param := _get_runnable_config_param(func):
        filter_args.append(config_param)       # 叫 config 或 runtime 的那个参数
    # filter_args.extend(_get_non_model_params(type_hints))
    return filter_args
```

`FILTERED_ARGS = ("run_manager", "callbacks")` 定义在 `base.py` 第 78 行。**这两个名字从 schema 里被无条件剔除**——这就是上面"静默消失"的代码级原因。

接着是 description 的**四级 fallback**，源码顺序就是优先级：

```python
description_ = description
if description is None and not parse_docstring:
    description_ = source_function.__doc__ or None
if description_ is None and args_schema:
    if isinstance(args_schema, type) and is_basemodel_subclass(args_schema):
        description_ = args_schema.__doc__
        if description_ and "A base class for creating Pydantic models" in description_:
            description_ = ""
        elif not description_:
            description_ = None
    elif isinstance(args_schema, dict):
        description_ = args_schema.get("description")
    else:
        msg = "Invalid args_schema: expected BaseModel or dict, got ..."
        raise TypeError(msg)
if description_ is None:
    msg = "Function must have a docstring if description not provided."
    raise ValueError(msg)
if description is None:
    # Only apply if using the function's docstring
    description_ = textwrap.dedent(description_).strip()
```

读出来的规则：

1. 显式 `description="..."` 最优先
2. 否则用**函数 docstring**（且仅当没开 `parse_docstring`）
3. 否则用 `args_schema` 的类 docstring（并特判 pydantic 自动生成的那句 "A base class for creating Pydantic models"，会当空串）
4. 否则 `args_schema` 是 dict 时读 `["description"]`
5. 全空 → **`ValueError: Function must have a docstring if description not provided.`**

注意第 3 条的特判：**框架知道 pydantic 会塞一句无意义的话，专门为它写了一个 `in` 判断**。这类"防御性字符串匹配"在 LangChain 里出现很多次，读到就知道这地方历史上踩过坑。

还有一个容易忽略的不对称：`tool()` 装饰器的 `error_on_invalid_docstring` 默认 **`True`**，而 `StructuredTool.from_function` 的默认是 **`False`**。所以你用 `@tool(parse_docstring=True)` 时，docstring 写错会**当场 ValueError**；直接调 `from_function` 时反而宽容。这是两个不同层级的"严格程度"。

### 2.4 `create_schema_from_function`：本章的心脏

这个函数有 8 个步骤，是全章最该逐行读的一段。核心逻辑：

```python
def create_schema_from_function(
    model_name: str,
    func: Callable[..., Any],
    *,
    filter_args: Sequence[str] | None = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> TypeBaseModel:
    sig = inspect.signature(func)

    if _function_annotations_are_pydantic_v1(sig, func):
        validated = validate_arguments_v1(func, config=_SchemaConfig)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
            validated = validate_arguments(func, config=_SchemaConfig)

    # Let's ignore `self` and `cls` arguments for class and instance methods
    in_class = bool(func.__qualname__ and "." in func.__qualname__)
    ...
    inferred_model = validated.model
    ...
    description, arg_descriptions = _infer_arg_descriptions(
        func, parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
    # Pydantic adds placeholder virtual fields we need to strip
    valid_properties = []
    for field in get_fields(inferred_model):
        if not has_args and field == "args":        continue
        if not has_kwargs and field == "kwargs":    continue
        if field == "v__duplicate_kwargs":          continue
        if field not in filter_args_:
            valid_properties.append(field)

    return _create_subset_model(
        model_name, inferred_model, list(valid_properties),
        descriptions=arg_descriptions, fn_description=description,
    )
```

七个值得停下来看的点：

**(1) 它用的是 pydantic 的 `validate_arguments`，而且明确知道这是弃用 API。**

```python
# We are using deprecated functionality here.
# This code should be re-written to simply construct a Pydantic model
# using inspect.signature and create_model.
```

这句注释是教材里最诚实的一行：**LangChain 自己也在等技术债还完**。你读源码时看到这种注释，别照它的实现抄，但要理解它的意图。

**(2) 生成的模型配置是 `extra="forbid"`。**

```python
class _SchemaConfig:
    extra: str = "forbid"
    arbitrary_types_allowed: bool = True
```

`forbid` 意味着：**模型多传一个没声明的参数 → 校验直接失败**。这是好事（能立刻抓到模型乱编参数），但也解释了你在第 2 章见过的 `handle_validation_error` 为什么有用——模型确实会多传参数。

**(3) `in_class` 靠 `__qualname__` 里有没有 `.` 判断。**

```python
in_class = bool(func.__qualname__ and "." in func.__qualname__)
```

装饰器定义的函数是 `outer.<locals>.my_tool`（也含 `.`），所以这个判断会把闭包函数误判成"类方法"。不过后面只有在参数列表**首项真的叫 `self` 或 `cls`** 时才会剔除，所以实际无害——**但它解释了为什么你的工具函数里别把第一个参数起名 `self`**。

**(4) 三个 pydantic 虚拟字段必须手工剥掉：`args` / `kwargs` / `v__duplicate_kwargs`。**

源码注释 `# Pydantic adds placeholder virtual fields we need to strip`。这是"用别人的校验器生成 schema"的代价：pydantic 为 `*args` / `**kwargs` 造了占位字段，LangChain 得手工擦掉，否则它们会出现在发给模型的 schema 里。

**(5) `include_injected=True` 是默认值，理由写得很清楚。**

```python
"""Defaults to `True`, since we want to include them in the schema when *validating* tool inputs."""
```

**这就是"两个 schema"的根**：校验时需要注入参数（因为运行时要真的把 `tool_call_id` 填进去），发给模型时必须剔除。下一节展开。

**(6) docstring 解析在 `_infer_arg_descriptions` 里，且会校验。**

```python
def _validate_docstring_args_against_annotations(...):
    ...
    msg = f"Arg {arg} in docstring not found in function signature."
    raise ValueError(msg)
```

原文措辞就是 `Arg {arg} in docstring not found in function signature.`。你以后看到这个报错，别怀疑框架——是 docstring 里写了函数里不存在的参数名。

**(7) 最终产物是 `_create_subset_model(...)`，不是 `validate_arguments` 那个模型本身。**

它是一个**子集模型**：只保留 `valid_properties` 里的字段，并把 `descriptions`（参数描述）和 `fn_description`（函数描述）塞进去。所以你 `print(tool.args_schema.model_json_schema())` 看到的 `description` 字段，来自这里。

### 2.5 两个 schema：本节是整章最重要的认知

这是新手最容易糊的地方，也是"模型收不到参数"问题的高发区。

```python
@override
def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
    """The tool's input schema."""
    if self.args_schema is not None:
        if isinstance(self.args_schema, dict):
            return super().get_input_schema(config)
        return self.args_schema
    return create_schema_from_function(self.name, self._run)
```

```python
@property
def tool_call_schema(self) -> ArgsSchema:
    """Get the schema for tool calls, excluding injected arguments."""
    if isinstance(self.args_schema, dict):
        if self.description:
            return {**self.args_schema, "description": self.description}
        return self.args_schema

    if (memo := self._tool_call_schema_memo) is not None:
        return memo

    full_schema = self.get_input_schema()
    fields = []
    if (is_pydantic_v2_subclass(full_schema)
            and not full_schema.__pydantic_complete__):
        full_schema.model_rebuild()

    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        if not _is_injected_arg_type(type_):
            fields.append(name)
    subset_model = _create_subset_model(
        self.name, full_schema, fields, fn_description=self.description
    )
    _patch_json_schema_cache(subset_model)
    self._tool_call_schema_memo = subset_model
    return subset_model
```

对照表（请直接背下来）：

| | `get_input_schema()` | `tool_call_schema` |
|---|---|---|
| 含注入参数（`ToolRuntime` / `InjectedToolArg`） | **含** | **不含** |
| 谁用 | `_parse_input` 校验、你自己调试 | **发给模型**（`bind_tools`）、`.args` 属性 |
| 缓存 | 不缓存 | 实例级 memo + JSON schema 字典级缓存 |
| 类型 | `Type[BaseModel]` | `Type[BaseModel]` 或 `dict` |

一句话：**「校验要全，发出去要瘦」**。你运行时需要 `tool_call_id` 才能回填 `InjectedToolCallId`，但模型绝不该知道有这么个参数——它凭什么自己编一个 call id？

这也解释了一个常见困惑：**为什么 `param: ToolRuntime` 没出现在 `bind_tools` 的 payload 里，但你的函数能收到它？** 因为 `_is_injected_arg_type` 在 `tool_call_schema` 里把它滤了，而运行时由 `_parse_input` + `_to_args_and_kwargs` 注入。

### 2.6 三个性能补丁：为什么"每次模型调用都重建 schema"是陷阱

agent 循环里，每一轮都要把工具 schema 序列化发给模型。如果不缓存，8 个工具就是 8 次 `model_json_schema()`。源码里为此写了三层缓存，注释给的数据很具体：

**(1) `structured.py` 的 `_model_json_schema`，`lru_cache(maxsize=256)`：**

```python
@functools.lru_cache(maxsize=256)
def _model_json_schema(args_schema: TypeBaseModel, _rebuild_token: object) -> Any:
    """...
    Cached because pydantic does not memoize this per class and it dominates the
    dump -- 1.10 ms against 0.02 ms for an eight-tool payload -- which tracing
    pays on every run.

    `_rebuild_token` is the class's validator, which `model_rebuild()` replaces.
    Keying on it retires the entry for a rebuilt schema...
    """
```

**1.10 ms vs 0.02 ms，55 倍。** 这个数字值得记住：schema 生成不是免费的，它在每轮循环里都发生。

注意缓存键里塞了 `_rebuild_token`（`__pydantic_validator__`）——这是个很讲究的设计：`model_rebuild()` 会换掉 validator，键一变，旧缓存自然失效。**用"对象的身份"当版本号**，比手工清缓存可靠。

**(2) `base.py` 的 `_patch_json_schema_cache`：给动态生成的类打猴子补丁。**

```python
"""Patch `model_json_schema` (or `schema` for pydantic v1) to cache.

Pydantic regenerates the full JSON-schema dict on every `model_json_schema()` call
— there is no per-class cache.
"""
def _cached_json_schema(cls, *args, **kwargs):
    if not args and not kwargs:
        cached = cls.__dict__.get("_json_schema_cache")
        if cached is not None:
            return cached
    result = orig(*args, **kwargs)
    if not args and not kwargs:
        cls._json_schema_cache = result
    return result
```

**只有全默认参数调用才走缓存**——带参数的调用照常生成。这是"缓存但不骗人"的写法：默认路径快，特殊路径仍然正确。

**(3) `_tool_call_schema_memo` + `_TOOL_CALL_SCHEMA_FIELDS` 失效机制。**

```python
_TOOL_CALL_SCHEMA_FIELDS = frozenset({"name", "description", "args_schema"})
"""Fields the memoized `tool_call_schema` is built from; reassignment clears it."""

@override
def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
    if name in _TOOL_CALL_SCHEMA_FIELDS and self.__pydantic_private__ is not None:
        self._tool_call_schema_memo = None
```

改 `name` / `description` / `args_schema` 任何一个 → memo 立刻失效。`model_copy(update=...)` 也要专门清（因为它绕过 `__setattr__` 直接写 `__dict__`），源码在 `model_copy` 里还特意写了原因注释。

**这三个补丁给你两个可迁移的工程认知：**

1. **缓存键要包含"能改变结果的东西"**。`name`/`description`/`args_schema` 变了，schema 就变了，memo 必须失效。漏一个就是脏读。
2. **同一份数据被不同层以不同代价重新计算时，逐层加缓存**。这里三层各自解决一个层次的重复：生成类（memo）、生成 dict（patch）、跨实例复用（lru_cache）。

### 2.7 执行流水线：`invoke` → `run` 的 11 步

```
invoke(input)                      # input 可以是 str / dict / ToolCall
  └─ _prep_run_args                 # ToolCall 拆成 (args.copy(), call_id)
       └─ run(tool_input, ...)
           ① CallbackManager.configure(...)           # 上级 callbacks + 自身 tags/metadata 合并
           ② _filter_injected_args(tool_input)        # 从回调输入里剔掉注入参数
           ③ on_tool_start(...)                       # 拿 run_manager
           ④ patch_config + set_config_context        # 子配置进 contextvar
           ⑤ _to_args_and_kwargs(tool_input, call_id) # → (args, kwargs)，内部走 _parse_input
           ⑥ 若 _run 签名有 run_manager → 注入
           ⑦ 若 _run 签名有 config 类参数 → 注入
           ⑧ response = context.run(self._run, ...)   # 真正调用你的函数
           ⑨ content/artifact 分流（看 response_format）
           ⑩ 异常三分支：ValidationError / ToolException / 其它
           ⑪ _format_output(...) → on_tool_end → return
```

`_prep_run_args` 是关键的一小段，它决定了"模型给的 ToolCall 怎么变成 Python 实参"：

```python
def _prep_run_args(value, config, **kwargs):
    config = ensure_config(config)
    if _is_tool_call(value):
        tool_call_id: str | None = cast("ToolCall", value)["id"]
        tool_input = cast("ToolCall", value)["args"].copy()   # 注意 .copy()
    else:
        tool_call_id = None
        tool_input = cast("str | dict[str, Any]", value)
    return tool_input, dict(..., tool_call_id=tool_call_id, config=config, **kwargs)
```

`.copy()` 是为了后面能安全地往 `tool_input` 里塞注入值（`_parse_input` 里会 `tool_input[k] = tool_call_id`），不污染调用方手里的原始 dict。

`_parse_input` 里有三段值得单独拎出来：

**(a) 字符串输入 + 结构化 schema = 直接报错：**

```python
if isinstance(tool_input, str):
    if input_args is not None:
        if isinstance(input_args, dict):
            msg = (
                "String tool inputs are not allowed when "
                "using tools with JSON schema args_schema."
            )
            raise ValueError(msg)
        key_ = next(iter(get_fields(input_args).keys()))
        if issubclass(input_args, BaseModel):
            input_args.model_validate({key_: tool_input})
```

注意那个 `key_ = next(iter(...))`：**单参数工具允许你传裸字符串**，框架把它包成 `{第一个字段名: 你传的串}` 再校验。这就是"为什么单参数工具能 `tool.run("杭州")`"。

**(b) `InjectedToolCallId` 缺 id 时的报错，措辞极其啰嗦但有用：**

```python
msg = (
    "When tool includes an InjectedToolCallId "
    "argument, tool must always be invoked with a full "
    "model ToolCall of the form: {'args': {...}, "
    "'name': '...', 'type': 'tool_call', "
    "'tool_call_id': '...'}"
)
raise ValueError(msg)
```

含义：**你这工具只能被 agent 调，不能手动 `.invoke({"x": 1})`**。

**(c) 校验后的输入是"provided + 别名 + 有默认值的字段"三者的并集：**

```python
for k in result_dict:
    if k in tool_input:             validated_input[k] = getattr(result, k)   # 输入里给了
    elif k in provided_fields:      validated_input[k] = getattr(result, k)   # 通过别名给了
    elif k in field_info and k not in {"args", "kwargs"}:
        fi = field_info[k]
        has_default = (not fi.is_required() if hasattr(fi, "is_required")
                       else not getattr(fi, "required", True))
        if has_default:             validated_input[k] = getattr(result, k)   # 有默认值
```

这段解释了：**为什么 `Field(default=1)` 的参数，模型不传也能进你的函数；而没默认值的必填参数不传就 ValidationError。**

### 2.8 `_format_output`：返回值怎么变成 `ToolMessage`

```python
def _format_output(content, artifact, tool_call_id, name, status):
    if (isinstance(content, list) and content
            and all(isinstance(item, ToolOutputMixin) for item in content)):
        return content                                  # ① 已经是消息列表，原样返回
    if isinstance(content, ToolOutputMixin) or tool_call_id is None:
        return content                                  # ② 自己造了 ToolMessage / 手动调用
    normalized_content = _normalize_message_content(content)
    content = _stringify(content) if normalized_content is None else normalized_content
    return ToolMessage(content, artifact=artifact, tool_call_id=tool_call_id,
                       name=name, status=status)        # ③ 包成 ToolMessage
```

三个出口各对应一种用法：

| 出口 | 触发条件 | 你什么时候用它 |
|---|---|---|
| ① 原样返回 | 返回 `list[ToolOutputMixin]` | 一个工具要回多条 `ToolMessage`（多结果） |
| ② 原样返回 | 返回的本身就是 `ToolMessage`，或 `tool_call_id is None` | 你要自己控制 `artifact`/`content`；或手动调 `tool.invoke(...)` 而非 agent 调 |
| ③ 包成 `ToolMessage` | 其它所有情况 | 默认路径 |

`_stringify` 只有两行，但是全章最该记住的一行：

```python
def _stringify(content: Any) -> str:
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)
```

**`json.dumps(..., ensure_ascii=False)`** —— 两个后果：中文不会变成 `\uXXXX`（省 token，且模型读得更准）；但 dict 会变成**字符串**，模型看到的是 JSON 文本而不是结构化对象。这就是第 2 章结论的代码级证据。

`_normalize_message_content` 则负责"哪些返回值得当内容块"：

```python
if isinstance(obj, str):
    return obj
if isinstance(obj, Sequence) and all(_is_message_content_block(e) for e in obj):
    return list(obj)
return None
```

其中 `_is_message_content_block` 只认两类：字符串，或 `type` 在 `TOOL_MESSAGE_BLOCK_TYPES` 里的 dict（`text` / `image_url` / `image` / `json` / `search_result` / `custom_tool_call_output` / `document` / `file`）。**所以工具返回 `[{"foo": 1}]` 不会被当成内容块，而是会被 `_stringify` 成字符串。**

### 2.9 错误处理：让工具"失败得体面"

`run()` 的异常处理是三分支结构：

```python
except (ValidationError, ValidationErrorV1) as e:
    if not self.handle_validation_error:
        error_to_raise = e
    else:
        content = _handle_validation_error(e, flag=self.handle_validation_error)
        status = "error"
except ToolException as e:
    if not self.handle_tool_error:
        error_to_raise = e
    else:
        content = _handle_tool_error(e, flag=self.handle_tool_error)
        status = "error"
except (Exception, KeyboardInterrupt) as e:
    error_to_raise = e

if error_to_raise:
    run_manager.on_tool_error(error_to_raise, tool_call_id=tool_call_id)
    raise error_to_raise
output = _format_output(content, artifact, tool_call_id, self.name, status)
```

三种失败的命运：

| 异常类型 | `handle_*` 未设（默认） | `handle_*` 已设 |
|---|---|---|
| `ValidationError`（参数不合 schema） | **抛出**，agent 崩 | 变成观测结果，`status="error"` |
| `ToolException`（你主动抛） | **抛出** | 变成观测结果，`status="error"` |
| 任何其它异常（`ValueError`/`HTTPError`…） | **永远抛出** | **永远抛出**（无开关） |

最后一行是本章最实用的一个认知：**普通异常没有开关**。所以工具里该抛 `ToolException` 而不是 `ValueError`——前者可以被"喂回模型让它重试"，后者一定炸穿。

`handle_tool_error` 的四种取值（源码 docstring 原话）：

> If `False`, the exception is re-raised. If `True`, the exception message is returned as tool output. If a string is passed, that string is returned as tool output. If a callable is passed, it receives the exception and its return value is used as the tool output.

`True` 最省事、**也最危险**：它把异常的 `str()` 原样喂给模型。如果异常里带了内部路径或 SQL，你就等于把内部细节写给模型看。生产上宁可用 callable 做过滤。

### 2.10 `return_direct` 的真实边界

很多人以为 `return_direct=True` 就"工具跑完直接结束"。源码里这个字段的注释确实这么写（`the AgentExecutor will stop looping`），但**真正的图装配逻辑在 `factory.py`**（第 1 章 §2.5 已解剖）：`_make_tools_to_model_edge` 只有在**全部**待执行的工具都是 `return_direct=True` 时才回 `END`。

这意味着：

- 模型一次调 2 个工具，1 个 `return_direct` 1 个普通 → **不会提前结束**，都跑完再回模型
- 想"工具结果直接当最终答案"→ 让模型只调那一个，或用 middleware 的 `jump_to="end"`

这条边界在文档里几乎不提，只能从源码读出来。**这是本章一个"读源码 > 读文档"的实例。**

---

## ③ 你敲这段

新建 `langchain_learn/04_tool_layer.py`。**A–E 段全部离线**（不花 token），F–G 段才发请求。

```python
"""第 4 章：Tool 层。A–E 离线，F–G 才发请求。

对应教材：docs/langchain-family/langchain_learn/05-ch04-tool-layer.md
运行：在项目根目录 python langchain_learn/04_tool_layer.py
"""

import json
from typing import Annotated

from langchain_core.tools import (
    InjectedToolCallId,
    StructuredTool,
    ToolException,
    tool,
)
from pydantic import BaseModel, Field

# ============ A. @tool 的四种写法，看它到底给你什么 ============
print("=" * 64)
print("A. 四种写法 → 什么类型")
print("=" * 64)


@tool
def t_bare(city: str) -> str:
    """查天气。"""
    return f"{city} 晴"


@tool("renamed_tool")
def t_named(city: str) -> str:
    """查天气。"""
    return f"{city} 晴"


@tool(parse_docstring=True)
def t_doc(city: str) -> str:
    """查天气。

    Args:
        city: 城市名，例如 杭州。
    """
    return f"{city} 晴"


@tool(infer_schema=False)
def t_raw(payload: str) -> str:
    """吃什么给什么。"""
    return payload


for t in (t_bare, t_named, t_doc, t_raw):
    print(f"{t.name:<16} {type(t).__name__:<16} args={list(t.args.keys())}")

# ============ B. 一个函数 → 三份 schema ============
print("\n" + "=" * 64)
print("B. 三份 schema 的差异（本章最重要的一屏）")
print("=" * 64)


class WeatherIn(BaseModel):
    """查天气的入参。"""

    city: str = Field(description="城市名")
    days: int = Field(default=1, description="预报天数")


@tool("weather", args_schema=WeatherIn)
def weather(city: str, days: int = 1) -> str:
    """查询指定城市天气。"""
    return f"{city} 未来 {days} 天：晴"


print("1) .args（最瘦，只有 properties）:")
print(json.dumps(weather.args, ensure_ascii=False, indent=2))

print("\n2) tool_call_schema（发给模型的）:")
tcs = weather.tool_call_schema
print(json.dumps(tcs.model_json_schema(), ensure_ascii=False, indent=2)
      if hasattr(tcs, "model_json_schema") else tcs)

print("\n3) get_input_schema()（校验用的）:")
print(json.dumps(weather.get_input_schema().model_json_schema(),
                 ensure_ascii=False, indent=2))

print("\n4) 直接 .run 单参数 → 裸字符串也能过（_parse_input 的 first-field 包装）:")
print("   ", weather.run("杭州"))

# ============ C. 注入参数：模型看不到，但你能拿到 ============
print("\n" + "=" * 64)
print("C. InjectedToolCallId：schema 里没有它")
print("=" * 64)


@tool
def echo_with_id(
    text: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """回显文本，并把 tool_call_id 一起带回。"""
    return f"{text} | call_id={tool_call_id}"


print("tool_call_schema 字段:", list(
    echo_with_id.tool_call_schema.model_json_schema()["properties"].keys()))
print("get_input_schema 字段:", list(
    echo_with_id.get_input_schema().model_json_schema()["properties"].keys()))

print("\n手动 invoke（会炸，因为它需要完整 ToolCall）:")
try:
    echo_with_id.invoke({"text": "hi"})
except Exception as e:
    print("   ", type(e).__name__, ":", str(e)[:120], "...")

print("\n用完整 ToolCall invoke（成功）:")
call = {"type": "tool_call", "name": "echo_with_id",
        "args": {"text": "hi"}, "id": "call_abc"}
print("   ", echo_with_id.invoke(call))

# ============ D. 三种失败，三种命运 ============
print("\n" + "=" * 64)
print("D. ToolException vs ValueError")
print("=" * 64)


@tool
def fail_tool_exc(x: int) -> str:
    """抛 ToolException。"""
    raise ToolException(f"业务失败：x={x} 不合法")


@tool
def fail_value(x: int) -> str:
    """抛 ValueError。"""
    raise ValueError(f"内部错误：x={x}")


@tool(handle_tool_error="工具暂时不可用，请稍后重试")
def fail_handled() -> str:
    """已配错误处理。"""
    raise ToolException("真实原因：DB 连接池耗尽（不该给模型看）")


for name, t, inp in [
    ("ToolException + 无 handler", fail_tool_exc, {"x": 1}),
    ("ValueError    + 无 handler", fail_value, {"x": 1}),
    ("ToolException + handler=字符串", fail_handled, {}),
]:
    try:
        print(f"[{name}] → {t.invoke(inp)!r}")
    except Exception as e:
        print(f"[{name}] → 抛出 {type(e).__name__}: {e}")

print("\n参数类型错误（ValidationError）→ handle_validation_error 未设 → 抛出:")
try:
    fail_tool_exc.invoke({"x": "不是整数"})
except Exception as e:
    print("   ", type(e).__name__, str(e)[:100], "...")

# ============ E. content vs artifact ============
print("\n" + "=" * 64)
print("E. response_format='content_and_artifact'")
print("=" * 64)


@tool(response_format="content_and_artifact")
def run_tests(path: str) -> tuple[str, dict]:
    """跑测试，返回摘要 + 完整日志。"""
    full_log = "line1\nline2\nFAILED tests/test_api.py:42"
    return f"{path}：2000 行，FAILED 3 处，首条 tests/test_api.py:42", {
        "stdout": full_log, "failed": ["tests/test_api.py:42"]
    }


msg = run_tests.invoke(
    {"type": "tool_call", "name": "run_tests",
     "args": {"path": "tests/"}, "id": "call_t1"}
)
print("type     :", type(msg).__name__)
print("content  :", msg.content)
print("artifact :", json.dumps(msg.artifact, ensure_ascii=False))
print("模型只看到 content；artifact 留在对象上给下游用。")

# ============ F. 真实模型：它为什么（不）调工具 ============
print("\n" + "=" * 64)
print("F. 真实模型：description 决定它调不调")
print("=" * 64)

from _shared import dump_messages, get_model  # noqa: E402


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单状态。仅当用户明确给出订单号时使用，返回状态与金额。"""
    return f"订单 {order_id}：已发货，金额 299.00"


@tool
def vague_tool(q: str) -> str:
    """处理数据。"""
    return "ok"


model = get_model()
bound = model.bind_tools([query_order, vague_tool])

print("发给模型的 tools payload（只看 name/description/parameters）:")
for t in bound.kwargs.get("tools", []):
    fn = t.get("function", t)
    print("  -", fn.get("name"), "|", str(fn.get("description"))[:40], "...")

r1 = bound.invoke("帮我查下订单 A10086 到哪了")
print("\n问『查订单 A10086』→ tool_calls:")
print("   ", r1.tool_calls)

r2 = bound.invoke("帮我处理一下数据")
print("\n问『处理一下数据』→ tool_calls:")
print("   ", r2.tool_calls)
print("（观察：模糊描述的工具容易被乱选，或两个都不选）")

dump_messages([r1, r2])

# ============ G. 破坏实验（见教材 §⑤，先跑完再改） ============
print("\n" + "=" * 64)
print("G. 完整往返：工具真的被执行")
print("=" * 64)

first = bound.invoke("订单 A10086 状态？")
if first.tool_calls:
    call = first.tool_calls[0]
    result = {"query_order": query_order, "vague_tool": vague_tool}[
        call["name"]].invoke(call)
    dump_messages([first, result])
    final = model.invoke(
        [*[first, result], ("user", "用一句话总结")]
    )
    print("\n最终回答:", final.content)
```

---

## ④ 你应该看到

> **先预测，再运行。** 下面每一段都写下你的预期，再对照。

**A 段**：四行，注意最后一行类型不同。

```
t_bare           StructuredTool   args=['city']
renamed_tool     StructuredTool   args=['city']
t_doc            StructuredTool   args=['city']
t_raw            Tool             args=[]          ← infer_schema=False 退化成 Tool
```

`t_raw` 的 `args` 是 **空列表**——这就是"模型没法知道要传什么"的直观样子。

**B 段**：三份 schema 逐层变胖。

- `.args` 只有 `city` / `days` 两个 property
- `tool_call_schema` 多出 `title`、`description`（来自你的 `WeatherIn` docstring + 字段 `Field(description=...)`）
- `get_input_schema()` 在这题里**看起来一样**（因为没有注入参数）——**这正是关键：只有 C 段才看得出差别**
- 第 4 行输出 `杭州 未来 1 天：晴`（`days` 走了默认值）

**C 段**：

```
tool_call_schema 字段: ['text']              ← 只有 text
get_input_schema 字段: ['text', 'tool_call_id']  ← 多出 tool_call_id
```

这就是 §2.5 那张表的实证。手动 invoke 的报错原文以 `When tool includes an InjectedToolCallId argument...` 开头。

**D 段**：

```
[ToolException + 无 handler] → 抛出 ToolException: 业务失败：x=1 不合法
[ValueError    + 无 handler] → 抛出 ValueError: 内部错误：x=1
[ToolException + handler=字符串] → '工具暂时不可用，请稍后重试'    ← 没抛，变成返回值
参数类型错误（ValidationError）→ 抛出 ValidationError: 1 validation error for fail_tool_exc
```

第三行是重点：**异常变成了工具输出**。真实原因 `DB 连接池耗尽` 被你的字符串替换掉了——这就是"别用 `True`"的理由。

**E 段**：

```
type     : ToolMessage
content  : tests/：2000 行，FAILED 3 处，首条 tests/test_api.py:42
artifact : {"stdout": "line1\nline2\nFAILED tests/test_api.py:42", "failed": ["tests/test_api.py:42"]}
```

**F 段**：你会看到两个工具的 payload 都被打印出来。`vague_tool` 的 description 只有"处理数据。"四个字——这是它被乱选（或不被选）的根本原因。两次 `tool_calls` 的对比就是本章教学目的。

**G 段**：`dump_messages` 会打出 3 条消息：`AIMessage(tool_calls)`、`ToolMessage`、`AIMessage(最终回答)`。这是第 2 章那四条消息的**真实生成版**。

> **诚实声明**：以上期望值是按源码逻辑推导的，我没有在你的机器上实机运行（本机 pwsh 被执行环境拦住）。若某处不符，先查版本：`python -c "import langchain_core; print(langchain_core.__version__)"`。

---

## ⑤ 三个破坏性实验

**每做完一个，写下"我原本以为…实际上…"。**

### 实验 A：删掉 docstring

把 `query_order` 的 docstring 整行删掉，其余不动，重跑。

预期：**导入阶段（decorator 执行时）就炸**，不是等到 invoke：

```
ValueError: Function must have a docstring if description not provided.
```

**要理解的是"炸的时间点"**：`@tool` 在**定义时**就构造完了整个 `StructuredTool`（含 schema），所以 schema 错误永远是 import 期错误，不是运行期错误。这条能解释你以后遇到的一类怪现象：**改了工具代码但报错在 import 行**。

### 实验 B：docstring 里写个不存在的参数

```python
@tool(parse_docstring=True)
def bad_doc(city: str) -> str:
    """查天气。

    Args:
        banana: 这是个不存在的参数。
    """
    return "ok"
```

预期：

```
ValueError: Arg banana in docstring not found in function signature.
```

再做一件更有价值的事：**把 `parse_docstring=True` 删掉，再跑一次**。你会发现不报错了——因为默认 `parse_docstring=False`，docstring 只被当"整段描述"，不做参数级解析。**这就是 §2.3 里那个 `error_on_invalid_docstring` 默认值不对称的实际后果。**

### 实验 C：参数名叫 `config`

```python
@tool
def collides(config: str, x: int) -> str:
    """故意撞名。"""
    return f"{config}-{x}"
```

先 `print(collides.args)` 看 schema，再 `collides.invoke({"config": "a", "x": 1})`。

预期：schema 里**没有 `config`**（被 `_filter_schema_args` 剔了），invoke 报

```
TypeError: collides() missing 1 required positional argument: 'config'
```

再把参数名换成 `run_manager`：schema 里同样没有它，但 **invoke 不会报错**——因为它被当成"框架注入项"，压根没传给你的函数，你函数里拿到的会是默认值或直接 TypeError（取决于你有没有给默认值）。

**这两个名字的差别（一个炸得响、一个静默）是本章最值得记住的坑。**

### 实验 D（加分）：让 `_stringify` 现形

```python
@tool
def return_dict() -> dict:
    """返回 dict。"""
    return {"a": 1, "中文": "值"}

msg = return_dict.invoke({"type": "tool_call", "name": "return_dict",
                          "args": {}, "id": "c1"})
print(type(msg.content), repr(msg.content))
```

预期：`<class 'str'> '{"a": 1, "中文": "值"}'`——**是 str，不是 dict；中文没被转义**。这两点分别对应 `_stringify` 的 `json.dumps` 和 `ensure_ascii=False`。

---

## ⑥ 自检题

1. `tool_call_schema` 和 `get_input_schema()` 差在哪？为什么必须有两个？
2. 你的工具参数在发给模型的 schema 里消失了。按什么顺序排查？给出至少 4 个可能原因。
3. 工具里抛 `ValueError` 和抛 `ToolException`，agent 循环的行为有什么不同？为什么这个区别重要？
4. `@tool` 报 `ValueError: Function must have a docstring...` 时，是 import 期还是 invoke 期？为什么？
5. 源码给 schema 生成加了**三层缓存**。如果只加一层（比如只做 `tool_call_schema` 的 memo），会漏掉什么？

### 参考答案

> 先自己做，再对答案。每条后面的 **延伸** 是超出题面的部分，用来连到后面的章。

**1. `tool_call_schema` 和 `get_input_schema()` 差在哪？为什么必须有两个？**

`get_input_schema()` 返回**完整** schema（含 `ToolRuntime` / `InjectedToolArg` / `InjectedToolCallId` 这类注入参数），用于运行时**校验和回填**；`tool_call_schema` 返回**剔除注入参数后**的子集模型，用于**发给模型**。

必须有两个的理由：注入参数的值只有框架知道（`tool_call_id` 是模型自己吐的 id，`ToolRuntime` 里的 state/store 是图运行时的东西），**模型无从提供，也不该看到**。如果把它们放进发给模型的 schema，模型会试图自己编一个 `tool_call_id`，直接破坏第 2 章讲的"请求-结果配对"。

> **延伸**：这条给你一个立刻可用的调试动作——当"模型不传某个参数"，先 `print(tool.tool_call_schema.model_json_schema()["properties"].keys())`。**如果那个参数不在里面，问题在 schema 生成阶段，跟模型无关。** 90% 的"模型不听话"到这一步就结案了。

**2. 参数在发给模型的 schema 里消失了。排查顺序？**

按"离源头从近到远"排：

| # | 原因 | 怎么确认 |
|---|---|---|
| 1 | 参数名是 `run_manager` / `callbacks` | 在 `FILTERED_ARGS` 里，无条件剔除 |
| 2 | 参数名是 `config`（或叫 `runtime` 且被识别为配置参数） | `_get_runnable_config_param` 会把它加入 filter |
| 3 | 参数被标了注入类型 | `InjectedToolArg` / `InjectedToolCallId` / `ToolRuntime` → `_is_injected_arg_type` 为真 |
| 4 | `infer_schema=False` | 返回的是 `Tool` 而不是 `StructuredTool`，`args` 为空 |
| 5 | 你显式传了 `args_schema`，但那个模型里没这个字段 | 打印 `tool.args_schema.model_fields.keys()` |
| 6 | 参数是 `*args` / `**kwargs` | `create_schema_from_function` 只保留声明过的具名参数 |

第 1、2 条是**静默**的（没有警告、没有报错），所以必须靠这份清单，不能靠猜。

> **延伸**：第 3 条有个反向用法——**如果你确实想让某个参数"框架给值、模型看不见"，就故意标成注入类型**，而不是从函数签名里删掉。这是 middleware 和 `ToolRuntime` 那套设计的入口。

**3. `ValueError` vs `ToolException`，行为差别？**

`ToolException` 是**可拦截**的：`run()` 里 `except ToolException` 分支会看 `handle_tool_error`，配了就把错误信息变成 `ToolMessage(status="error")` 喂回模型，**循环继续**，模型看到错误可以换个参数重试。

`ValueError`（以及任何非 `ToolException` 的异常）走最后一个 `except (Exception, KeyboardInterrupt)` 分支，**无条件 `raise`**，整个 agent 运行崩掉。

这个区别重要，因为**工具失败是常态，不是异常**：API 限流、参数格式不对、目标不存在——这些都应该让模型知道并重试，而不是让你的服务 500。一句话原则：**"我预料到的业务失败"抛 `ToolException`，"我没想到的 bug"让它抛原异常。**

> **延伸**：`handle_tool_error=True` 会把异常 `str()` 原样喂给模型。生产上别用 `True`，用 callable 把内部细节（路径、SQL、堆栈）过滤掉，只回"可操作的信息"。这既是安全边界，也是 prompt 质量边界。

**4. import 期还是 invoke 期？**

**import 期**（更准确说：decorator 执行、即模块加载时）。因为 `@tool` 在定义那一刻就调了 `StructuredTool.from_function`，而 `from_function` 里 `description_ is None` 就直接 `raise ValueError`。整个 schema 构造是**定义时完成的**，不是调用时算的。

所以：**任何 schema/description 错误都不可能活到运行期**。这条能解释"我只改了一个工具的 docstring，为什么整个服务起不来"。

> **延伸**：反过来说，`_run` 内部抛的错才是运行期。所以看到一个 `ValueError`，先看它出自哪个阶段——`from_function` / `create_schema_from_function` 是定义期，`_run` / `_parse_input` 是运行期。这两个阶段的修法完全不同（前者改签名/docstring，后者改业务逻辑）。

**5. 只加一层 memo 会漏掉什么？**

三层各解决一个层次的重复：

1. **`_tool_call_schema_memo`（实例级）**：重复访问 `tool_call_schema` 时不再重建"子集模型类"
2. **`_patch_json_schema_cache`（类级）**：pydantic 每次 `model_json_schema()` 都重算完整 dict，patch 后同参数默认调用直接返回缓存
3. **`_model_json_schema` 的 `lru_cache(256)`（跨实例）**：同一个 `args_schema` 被多个工具实例复用时共享

只做第 1 层，你省掉了"造类"，但每次序列化仍在算 dict（源码给的数字：8 工具 1.10 ms vs 0.02 ms）；只做第 3 层，不同工具实例各自造一遍类，类里塞的 `fn_description` 还不一样，无法共享。

> **延伸**：这三层里最值得学的是**失效策略**。`_TOOL_CALL_SCHEMA_FIELDS = {"name","description","args_schema"}` 精确声明"哪些字段能改变结果"；`_rebuild_token` 用 validator 对象身份当版本号。**缓存的难点从来不是"存"，而是"什么时候必须丢"。**

---

## ⑦ 承上启下

你现在能把一个函数变成一份精确的 schema，也知道它会怎么被执行、怎么失败、怎么回来。

但还有一条线没接上：**这份 schema 是怎么被送进 HTTP 请求体的？** 你在 F 段打印过 `bound.kwargs["tools"]`，可 `bind_tools` 到底做了什么？`ChatOpenAI` 把 `BaseTool.tool_call_schema` 翻译成 OpenAI 的 `{"type":"function","function":{...}}` 时，改了哪些字段？为什么 Anthropic 和 OpenAI 的 payload 形状不同？

**下一章（第 3 章 Model 层）就拆这个**：`init_chat_model` 如何解析 `"openai:gpt-4o"`、`bind_tools` 往 `RunnableBinding` 里塞了什么、以及"工具 schema 里的 description 到底从哪个字段读"。

> 想先歇一下的话，另一条合理路线是跳到**第 5 章 State 与 Reducer**——从"单个工具"上升到"工具之间共享的状态"。两条路都能接上，选你现在更想解决的那个问题。
