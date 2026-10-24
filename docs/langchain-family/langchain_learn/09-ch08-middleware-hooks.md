# 第 8 章 · Middleware 上：钩子地图

> 官方锚点：`/oss/python/langchain/middleware`（Middleware 总览）
> 源码落点：`langchain/agents/middleware/types.py`（2201 行）、`langchain/agents/factory.py`（编图段 1091–1894）
> 本机实证版本：**langchain 1.4.6 / langchain-core 1.6.12**（解释器：`.venv\Scripts\python.exe`）
> 前置：第 1 章（`invoke()` 真相）、第 5 章（state 与 reducer）、第 7 章（structured output）
> 诚实边界：本章行号与签名来自**实读源码**；所有"运行输出"标注为**源码推导**——本机沙箱拒绝执行 python（见 §7.1），未做实跑。

---

## 0. 先纠正一个流行误解

大多数人第一次看 Middleware，脑子里装的是这样的模型：

```
agent.run()
  ├─ before_hook()      # 像 Flask 的 before_request
  ├─ model()
  └─ after_hook()       # 像 Flask 的 teardown
```

于是自然推论：钩子是"在一个跑着的 agent 里被顺手调用一下的回调函数"。

**这个模型在 langchain v1 里是错的。** 真相是：

> 中间件的钩子**不是被调用的，而是被编译成图节点的**。你在 `create_agent(middleware=[...])` 里写的类，会在编译期被读一遍——读的不是它的实例，是**它的类**——然后 `factory.py` 用这些钩子**改写了整张 StateGraph 的节点表和边表**。

一句话记住本章：

> **Middleware 是"代码生成代码"：你写的是类，编译器读的是 `__class__`，产出的是节点和边。**

这一章不教你某个内置中间件怎么用（那是第 9 章）。这一章给你**钩子地图**：哪 12 个方法可以覆盖、每个方法在图里占什么位置、谁先谁后、哪些根本不成节点。

---

## ① 本章你会得到什么

读完并手敲完本章，你应该能回答这 6 个问题：

1. `AgentMiddleware` 一共几个可覆盖的钩子？为什么是"6 对 12 个"？同步版和异步版分别叫什么？
2. 为什么 `create_agent` 判断"你有没有覆盖某个钩子"用的是 `is not` 类比较，而不是 `hasattr`？
3. 三个 `before_*` 钩子按列表**正序**串，三个 `after_*` 按列表**逆序**串——为什么必须是这个方向？
4. 为什么 `wrap_model_call` **不**产生节点，却仍然是"洋葱"结构？它和 `after_model` 的本质区别是什么？
5. 在 `before_model` 里改 `request.messages` 为什么会 `DeprecationWarning`？正确写法是什么？
6. `jump_to` 为什么存在？它解决的是"顺序"问题还是"拓扑"问题？

验收清单（做完这 6 件事才算过）：

- [ ] 能默写 12 个方法名，并说出哪些是"节点"、哪些是"包装器"
- [ ] 能画出 4 个端点（entry / loop_entry / loop_exit / exit）的判定规则
- [ ] 能解释 `after_model` 逆序的因果，而不是背结论
- [ ] 能用 `override()` 正确改 `ModelRequest`
- [ ] 跑通配套脚本 A–F 段（零 token），说出每段在验证哪一条
- [ ] 能自己写一个"必须在第 2 个位置"的中间件，并解释为什么它不能任意插队

---

## ② 源码解剖

### 2.1 一个类，四个属性

`types.py:385` 起：

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the following methods to customize the agent's behavior:
    - before_agent: Called before the agent starts (once per invocation)
    - before_model: Called before each model call
    ...
    """
```

它**几乎是个空类**。全部可配置面只有 4 个类属性 + 1 个 property：

| 位置 | 声明 | 作用 |
|---|---|---|
| `types.py:397` | `state_schema: type[StateT] = _DefaultAgentState` | 这个中间件读写哪些 state 字段（会被合并进 agent 的 state） |
| `types.py:400` | `tools: Sequence[BaseTool] = ()` | 中间件**自带**的工具（会被融合进 agent 的工具列表） |
| `types.py:403` | `trace_policy: TracePolicy \| None = None` | 这个中间件的内部动作是否进 trace |
| `types.py:413` | `transformers: Sequence[TransformerFactory] = ()` | 消息历史变换工厂（压缩/裁剪用） |
| `types.py:423` | `name` (property) | 返回**类名**——这就是图里节点名的前缀 |

注意 `state_schema` 的默认值 `_DefaultAgentState`（`types.py:381`），它继承自 `AgentState`：

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Annotated[list[AnyMessage], add_messages]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromSchema]]
```

这就是第 5 章的产物：中间件能看到的 state，默认只有这三把钥匙。想多一把，就覆盖 `state_schema`。

### 2.2 12 个方法 = 6 个钩子 × 2

钩子的完整清单（`types.py` 行号是**方法定义行**）：

| 钩子 | 同步版 | 异步版 | 签名要点 |
|---|---|---|---|
| 进场 | `before_agent` `:431` | `abefore_agent` `:442` | `(state, runtime) -> dict \| None` |
| 每次模型前 | `before_model` `:455` | `abefore_model` `:466` | `(state, runtime) -> dict \| None` |
| 每次模型后 | `after_model` `:479` | `aafter_model` `:490` | `(state, runtime) -> dict \| None` |
| 模型包装 | `wrap_model_call` `:503` | `awrap_model_call` `:598` | `(request, handler) -> ModelResponse \| AIMessage` |
| 出场 | `after_agent` `:650` | `aafter_agent` `:661` | `(state, runtime) -> dict \| None` |
| 工具包装 | `wrap_tool_call` `:674` | `awrap_tool_call` `:756` | `(request, handler) -> ToolMessage \| Command` |

三个观察：

1. **所有方法默认实现都是 `raise NotImplementedError`**——不是 `pass`/`return None`。这个设计非常关键，见 §2.3。
2. 同步/异步**成对出现**。你只写同步版，异步调用链里它就"没被覆盖"。
3. 返回 `dict` 的四个钩子（`before_*` / `after_*`）是**state 增量**，会走第 5 章讲的 reducer。

以 `before_agent` 为例（`types.py:431-441`）：

```python
def before_agent(
    self,
    state: StateT,
    runtime: Runtime[ContextT],
) -> dict[str, Any] | None:
    """Logic to run before the agent starts.

    Args:
        state: The current agent state.
        runtime: The runtime context.

    Returns:
        Optional dict to update the state with.
    """
    raise NotImplementedError
```

### 2.3 覆盖检测：为什么必须是 `__class__` 比较

`factory.py:1163` 起，四段几乎一模一样的代码：

```python
middleware_w_before_agent = [
    m for m in middleware
    if m.__class__.before_agent is not AgentMiddleware.before_agent
    or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
]
```

**逐字读这句话**：只有"子类里真的重定义了那个函数对象"才算覆盖。

为什么不用 `hasattr`？因为父类**本来就有**这个方法名（只是会 raise）。`hasattr` 对任何子类都返回 `True`，等于所有中间件都"覆盖了所有钩子"——图里会给每个中间件生成 12 个空节点。`is not` 比的是**函数对象身份**，只有真正换掉了实现才算数。

这里还藏着一个易错点：`or` 的两边。**只实现异步版 `abefore_agent` 也算覆盖**，此时同步图里那个节点会抛 `NotImplementedError`……除非你的 agent 全程 `ainvoke`。这就是为什么仓库里每个钩子都有成对的"友好报错"：

```python
raise NotImplementedError(
    f"Middleware {self.name} implements 'abefore_agent' but not 'before_agent'. ..."
)
```

（`types.py:586-596` / `:744-754`，报错文案里直接给了三种修法：实现同步版、改用 `ainvoke`、或继承 `AgentMiddleware` 覆写。）

> **动手即得**：覆盖检测读的是 `m.__class__`（类），不是 `m`（实例）。所以中间件**不能**在 `__init__` 里动态塞 `self.before_model = ...` 来指望生效——编译期看的是类。这点和第 3 章讲的 callback handler 恰好相反（那边是实例方法就行）。

### 2.4 钩子怎么变成节点：`factory.py` 的四组判定

`factory.py:1163-1186` 收集出四组中间件（顺序 = 你传进列表的顺序）：

```python
middleware_w_before_agent   # 覆盖了 before_agent 或 abefore_agent
middleware_w_before_model   # ...
middleware_w_after_model    # ...
middleware_w_after_agent    # ...
```

然后 `factory.py:1611` 之后开始建节点：

```python
graph.add_node("model", _execute_model_sync)   # :1611 附近
graph.add_node("tools", tool_node)             # :1615 附近
for m in middleware_w_before_agent:            # :1636 附近
    graph.add_node(f"{m.name}.before_agent", ...)
for m in middleware_w_before_model:            # :1660 附近
    graph.add_node(f"{m.name}.before_model", ...)
for m in middleware_w_after_model:             # :1684 附近
    graph.add_node(f"{m.name}.after_model", ...)
for m in middleware_w_after_agent:             # :1708 附近
    graph.add_node(f"{m.name}.after_agent", ...)
```

节点名规则一句话：**`{中间件的类名}.{钩子名}`**。

你以后在 trace / 图可视化 / 报错栈里看到 `SummarizationMiddleware.before_model`，就知道它来自哪个类、挂在哪个钩子上。

### 2.5 四个端点与它们的判定规则

`factory.py:1715-1741`，这是全章最该背下来的一段：

```python
entry_node = (
    middleware_w_before_agent[0].name + ".before_agent"
    if middleware_w_before_agent
    else (middleware_w_before_model[0].name + ".before_model"
          if middleware_w_before_model else "model")
)

loop_entry_node = (
    middleware_w_before_model[0].name + ".before_model"
    if middleware_w_before_model else "model"
)

loop_exit_node = (
    middleware_w_after_model[0].name + ".after_model"
    if middleware_w_after_model else "model"
)

exit_node = (
    middleware_w_after_agent[-1].name + ".after_agent"   # 注意 [-1]！
    if middleware_w_after_agent else END
)
```

四个端点，四种"第一个/最后一个"策略：

| 端点 | 规则 | 直觉 |
|---|---|---|
| `entry_node` | 第一个 `before_agent`；否则第一个 `before_model`；否则 `model` | 从图的最外面进场 |
| `loop_entry_node` | 第一个 `before_model`；否则 `model` | **每圈**回到这里 |
| `loop_exit_node` | 第一个 `after_model`；否则 `model` | 从模型出来后先撞它 |
| `exit_node` | **最后**一个 `after_agent`；否则 `END` | 出场链的末端 |

`loop_*` 是"每圈都走"，`entry/exit` 是"一次调用只走一次"。这个区分就是 `before_agent` 和 `before_model` 的本体差别——不是语义差别，是**拓扑位置**差别。

### 2.6 连边：三个 before 正序，三个 after 逆序

这是全章最容易背错、也最值得理解的地方。

**`before_*`：正序串，末端指向下一层**（`factory.py:1814-1852` 一带）：

```
START → A.before_agent → B.before_agent → (entry_node)
A.before_model → B.before_model →  model
```

**`after_*`：逆序串，首端接到上一层**（`factory.py:1856-1894` 一带）：

```python
graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
for idx in range(len(middleware_w_after_model) - 1, 0, -1):
    m1 = middleware_w_after_model[idx]
    m2 = middleware_w_after_model[idx - 1]
    graph.add_edge(f"{m1.name}.after_model", f"{m2.name}.after_model")
```

于是：

```
model → B.after_model → A.after_model → (exit)
```

**为什么必须反过来？** 把中间件想成**嵌套的括号**，而不是一排管道：

```
A(before)  B(before)  [ model ]  B(after)  A(after)
```

- **进场**：A 先见（外层先动手）→ 正序
- **出场**：B 先收（内层先收口）→ 逆序

如果 after 也正序，你得到的是 `A.after → B.after`：A 跑完自己的收尾，B 才跑——**外层比内层先收口**，嵌套就翻过来了。所有"成对"的语义都会崩：A 在 before 里压入的栈帧，A 的 after 会先弹出，而那时 B 的 before 还没结束。

一句话记住：

> **before 正序进，after 逆序出；before 是"穿衣服"，after 是"脱衣服"，永远后穿先脱。**

`before_agent` / `after_agent` 同理，只是它们夹在最外层（一次调用一次），而 `before_model` / `after_model` 夹住**每一次**模型调用（每圈都夹）。

### 2.7 三个 `wrap_*`：不成节点，是洋葱

到这里你可能期待 `wrap_model_call` 也变成 `{name}.wrap_model_call` 节点。**它没有。**

`factory.py:1187-1224` 一带，它被**组合成函数**，而不是编成节点：

```python
sync_handlers = [
    traceable(name=f"{m.name}.wrap_model_call", **_wrap_trace_kwargs(m))(
        m.wrap_model_call
    )
    for m in middleware_w_wrap_model_call
]
wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)
```

然后在 `model` 节点**内部**调用（`factory.py:1543` 同步 / `:1602` 异步一带）：

```python
result = wrap_model_call_handler(request, _execute_model_sync)
```

`_chain_model_call_handlers` 的 docstring 一句话说明语义（原文）：

> Compose multiple `wrap_model_call` handlers into single middleware stack.
> **Multiple middleware compose with first in list as outermost layer.**

所以 wrap 的洋葱是：

```
A.wrap_model_call( request, handler=λ req_a:
    B.wrap_model_call( req_a, handler=λ req_b:
        _execute_model_sync(req_b) ) )
```

**节点 vs 包装器的判别口诀**：

| 形态 | 是否产生节点 | 谁调用它 | 能改什么 |
|---|---|---|---|
| `before_*` / `after_*` | ✅ 是节点 | 图调度器 | state（返回 dict） |
| `wrap_model_call` | ❌ 否 | `model` 节点内部 | **整个 request 和整个 response** |
| `wrap_tool_call` | ❌ 否 | `ToolNode` 内部（`factory.py:1091-1146`） | request + 工具结果，可**短路**不调工具 |

为什么 wrap 不做成节点？因为节点只能"前后加料"，而 wrap 要能**决定调不调**（重试、缓存、降级）。`handler` 是"下一层/内层/真正的模型调用"，你不调它，模型就不会被调用。

> 这正是 `factory.py:1772` 那个奇怪条件的来源：`if response_format or loop_exit_node != "model" or middleware_w_wrap_model_call:` ——如果存在 wrap_model_call，就要允许 `model` 节点**回到 loop_entry**，因为 wrap 可能动态改了工具集（`request.override(tools=[...])`），改完得再走一圈。

### 2.8 `ModelRequest`：为什么不能 `setattr`

`types.py:88` 起的 `ModelRequest` 是 `@dataclass(init=False)`，字段：

```
model / messages / system_message / tool_choice / tools
/ response_format / state / runtime / model_settings
```

两个坑：

**坑 1：`system_prompt` 和 `system_message` 互斥。** `types.py:139-141` 直接抛：

```python
raise ValueError("Cannot specify both system_prompt and system_message")
```

**坑 2：不能直接赋值。** `types.py:169-201` 的 `__setattr__` 会拦截：

```python
def __setattr__(self, name, value):
    warnings.warn(
        "Directly mutating ModelRequest attributes is deprecated. "
        "Use `request.override(...)` instead.",
        DeprecationWarning,
    )
```

正确写法（`types.py:203` 的 `override`，返回新对象）：

```python
new_request = request.override(system_prompt="You are terse.")
response = handler(new_request)
```

**为什么这么设计？** 因为 request 要在栈里被多层传递。如果允许原地改，A 层的修改会污染 B 层看到的对象，而且重试时无法回滚到"原始 request"。`override` 造新对象，等于**每层都有自己的副本**——这是洋葱能安全套多层的前提。

顺带记住 `messages` 字段的坑：它**不含 system message**（system 单独在 `system_message` 字段）。你在 `before_model(state, runtime)` 里拿到的 state 里的 `messages` 是**含** SystemMessage 的完整历史。两处口径不同，最容易写出重复 system 的 bug。

### 2.9 `jump_to`：绕过串行链的逃生门

`before_*` / `after_*` 返回的 dict 里可以带一个特殊键：

```python
return {"jump_to": "end"}     # 或 "tools" / "model"
```

`JumpTo` 字面量在 `types.py:68`：`"tools" | "model" | "end"`。

解析逻辑两处：`factory.py:1955` 的 `_resolve_jump` 把字符串映射成目标节点，`:2000-2003` 和 `:2053-2056` 处理**优先级**——state 里的 `jump_to` 优先于静态的 `__can_jump_to__`。

而 `__can_jump_to__` 是通过装饰器/钩子自动推断的（`factory.py:541-575` 的 `_get_can_jump_to`：先看同步方法上的 `__can_jump_to__` 属性，没有再看异步版本）。

>`_add_middleware_edge`（`factory.py:2120` 一带）只有在 `can_jump_to` 非空时才建**条件边**，否则建普通边。

**为什么需要它？** 因为串行链是"无逃逸"的：一旦 `after_model` 返回了值，编译器只会把它连向下一个 `after_model`。但有些中间件要"提前结束"（护栏拦截）或"回到工具"（自愈重试）。`jump_to` 就是给串行链开的**条件边**——它不是顺序问题，是**拓扑问题**：

| 需求 | 用什么 |
|---|---|
| 想调整"谁先谁后" | 调整列表顺序（before 正序 / after 逆序自动处理） |
| 想改变"下一步去哪" | `jump_to` + `__can_jump_to__` |
| 想改变"调不调" | `wrap_*`（不调 handler） |

### 2.10 装饰器：把普通函数升级成单钩子中间件

`types.py` 后半段全是装饰器，每个钩子三个重载（裸用 / 带参 / 带 kwargs）：

| 装饰器 | 行号 |
|---|---|
| `hook_config` | `:879` |
| `before_model` | `:934 / :940 / :952` |
| `after_model` | `:1123 / :1129 / :1141` |
| `before_agent` | `:1300 / :1306 / :1318` |
| `after_agent` | `:1512 / :1518 / :1530` |
| `dynamic_prompt` | `:1690 / :1696 / :1704` |
| `wrap_model_call` | `:1846 / :1852 / :1864` |
| `wrap_tool_call` | `:2019 / :2025 / :2037` |

三种最常用的写法：

```python
# 1) 裸装饰器：隐式中间件，名字取自函数名
@before_model
def add_context(state, runtime):
    return {"messages": [...]}

# 2) 带参数：显式声明 state_schema / 工具 / 能否跳转
@after_model(state_schema=MyState, can_jump_to=["end"])
def guard(state, runtime):
    return {"jump_to": "end"} if bad(state) else None

# 3) 动态 system prompt（最常用）
@dynamic_prompt
def pick_prompt(request: ModelRequest) -> str:
    return "You are a SQL expert." if "sql" in request.system_prompt else "..."
```

`dynamic_prompt` 本质是 `wrap_model_call` 的语法糖，它只改 `system_prompt` 然后调 `handler`。官方文档里"动态提示"那一节讲的就是它。

### 2.11 46 个导出名 = 内置中间件地图

`langchain/agents/middleware/__init__.py` 的 `__all__` 有 46 项。按用途分四类（完整的放在第 9 章，这里只给地图）：

| 类别 | 代表 | 用哪个钩子实现 |
|---|---|---|
| 上下文治理 | `SummarizationMiddleware`、`ContextEditingMiddleware`、`ClearToolUsesEdit` | `before_model` + `wrap_model_call` |
| 工具治理 | `ToolRetryMiddleware`、`ToolCallLimitMiddleware`、`LLMToolSelectorMiddleware`、`ToolEmulatorMiddleware` | `wrap_tool_call` / `wrap_model_call` |
| 人在环 | `HumanInTheLoopMiddleware` | `after_model` + interrupt |
| 安全/观测 | `PIIMiddleware`、`ModelCallLimitMiddleware`、`ModelFallbackMiddleware` | `before_model` / `wrap_model_call` |
| 规划/沙箱 | `TodoListMiddleware`、`FilesystemMiddleware`、`ShellToolMiddleware`、`CodexSandboxExecutionPolicy` | 多个钩子 + `tools` |

> ⚠️ 一个常见误判：`middleware/_execution.py` **不是**钩子执行引擎，它是 `ShellToolMiddleware` 的**执行策略**（Docker / Host / Codex 三种沙箱）。别在讲"钩子怎么执行"时引它。

---

## ③ 配套代码

`langchain_learn/08_official_middleware_hooks.py`，六段，**全部零 token 离线**：

| 段 | 验什么 | 关键观察点 |
|---|---|---|
| A | 钩子盘点 | 12 个方法名 + 默认实现是 raise 而非 pass |
| B | 覆盖检测 | `__class__.x is not AgentMiddleware.x` 的真值表；实例赋值不生效 |
| C | 洋葱顺序 | 用纯函数模拟 `_chain_model_call_handlers`，打印嵌套括号 |
| D | wrap_tool_call | handler 可短路；`request.override` 造新对象 |
| E | 装饰器 | `@before_model` 生成的类名与 `name` 属性 |
| F | `--real` | 只有这段需要真模型 |

手敲提醒：

1. **先跑 A 段**，对着打印出来的 12 个名字，回 §2.2 的表格逐个对。
2. B 段有个"陷阱分支"——在 `__init__` 里给实例赋值 `self.before_model = fn`，然后打印覆盖检测结果。你会看到 `False`，这就是 §2.3 讲的那条。
3. C 段的输出应该长这样（嵌套深度 = 中间件个数）：

```
A.wrap_model_call( handler=λ1:
  B.wrap_model_call( handler=λ2:
    C.wrap_model_call( handler=λ3:
      ⚙️ real model call
    )
  )
)
```

---

## ④ 三个破坏实验

### 实验 1：把 `after_model` 的中间件顺序反过来传，看谁先跑

传 `[A, B]` 再传 `[B, A]`，观察 after 的**实际执行顺序**。

- 期望（源码推导）：`[A,B]` → after 顺序是 `B → A`；`[B,A]` → `A → B`。
- **关键**：不是"顺序没变"，而是"永远和列表反着来"。如果两次打印结果相同，说明你把 before 和 after 弄混了。

### 实验 2：只实现异步版的中间件，放进同步调用链

写一个只有 `abefore_model` 的类，用 `create_agent(...).invoke(...)`。

- 期望：抛 `NotImplementedError`，报错文案里明确写"implements 'abefore_model' but not 'before_model'"。
- 然后改成 `ainvoke`，同样的中间件就通了。
- 这条实验的价值：记住**同步/异步是两套图**，不是一套图跑两种函数。

### 实验 3：在 `before_model` 里直接 `request.messages = ...`

- 期望：`DeprecationWarning`（`types.py:169-201` 的 `__setattr__`）。
- 改成 `handler(request.override(messages=...))` 后警告消失。
- 进阶：故意同时传 `system_prompt=` 和 `system_message=`，撞 `ValueError`（`types.py:139-141`）。

### 实验 4（拓扑级）：让 `after_model` 返回 `{"jump_to": "end"}`

- 不加 `can_jump_to=["end"]`：编译期不会建条件边，`jump_to` 被忽略（**静默失效**，这是最坑的）。
- 加上 `can_jump_to=["end"]`：护栏生效，直接跳到出口。
- 结论：`jump_to` 不是运行时魔法，它**需要编译期先建好边**。

---

## ⑤ 自检题

**Q1.** `AgentMiddleware` 有几个可覆盖的钩子方法？为什么说"6 对 12 个"？

<details><summary>答案</summary>
6 个语义钩子（before_agent / before_model / after_model / wrap_model_call / after_agent / wrap_tool_call），每个都有同步/异步两份实现，共 12 个方法名。覆盖检测是"覆盖任一版本即算覆盖"，所以列表长度按 6 个算。
</details>

**Q2.** 为什么覆盖检测用 `m.__class__.before_agent is not AgentMiddleware.before_agent`，而不是 `hasattr(m, "before_agent")`？

<details><summary>答案</summary>
因为父类本来就有这个方法（实现是 raise NotImplementedError），`hasattr` 对所有子类恒为真，会让每个中间件都生成 12 个空节点。`is not` 比较的是函数对象身份，只有真正替换了实现才算覆盖。
</details>

**Q3.** `[A, B, C]` 三个中间件同时实现 `before_model` 和 `after_model`，模型前后的实际顺序是什么？

<details><summary>答案</summary>
before：`A → B → C → model`（正序）；after：`model → C → B → A`（逆序）。合起来是嵌套括号 `A( B( C( model ) C ) B ) A`。
</details>

**Q4.** 为什么 `wrap_model_call` 不做成图节点？

<details><summary>答案</summary>
节点只能"在前/在后加料"，无法表达"调不调下一层"。wrap 的 handler 就是下一层（或真正的模型调用），只有做成函数包装才能实现重试、缓存、降级、短路。它是被 model 节点内部调用的，`_chain_model_call_handlers` 把它们组合成单一 handler。
</details>

**Q5.** `exit_node` 用的是 `middleware_w_after_agent[-1]` 还是 `[0]`？为什么？

<details><summary>答案</summary>
`[-1]`，最后一个。因为 after 链是逆序连的：`... → C.after_agent → B.after_agent → A.after_agent → END`，出场的"末端"是列表第一个，所以端点要取最后一个。`loop_exit_node` 取 `[0]`，因为它是从 model 出来的**入口**。
</details>

**Q6.** 你在 `before_model` 里想让 agent 直接结束。写 `return {"jump_to": "end"}` 却没生效，最可能漏了什么？

<details><summary>答案</summary>
漏了 `@before_model(can_jump_to=["end"])`（或类里声明 `__can_jump_to__`）。`jump_to` 需要编译期先建出条件边，没声明就是普通边，值会被静默丢弃。
</details>

---

## ⑥ 交给你自己

1. 写一个 `TimingMiddleware`：在 `before_model` 记开始时间，`after_model` 打印耗时。要求它必须处在**最内层**——怎么传列表？
2. 写一个 `TerseMiddleware`：用 `wrap_model_call` 把 `system_prompt` 追加一句"回答不超过 20 字"，不许用 `@dynamic_prompt`。
3. 写一个 `NoDoubleCallMiddleware`：用 `wrap_model_call` 做 5 秒内相同 request 的缓存，第二次直接返回缓存的 `AIMessage`（提示：`handler` 不被调用时，模型根本不会跑）。
4. 画一张图：两个中间件 `[Guard(只 after_model + can_jump_to), Tool(retry, wrap_tool_call)]` 编出来的 StateGraph 长什么样？标出四个端点。
5. 解释：为什么 `middleware_w_after_model[0]` 是 `loop_exit_node`，而 `[−1]` 是模型出来的第一个节点？

---

## ⑦ 诚实边界与延伸

### 7.1 本章的"实测"边界

- ✅ **实读**：`types.py` 全 2201 行的类定义、12 个钩子签名、装饰器段、`factory.py` 的 1163–1186（覆盖检测）/1091–1224（wrap 组合）/1611–1741（节点与端点）/1743–1894（连边）/1955 与 2000–2053（jump 解析）。
- ⚠️ **源码推导，未实跑**：本章所有"期望输出"。本机 `pwsh` 被 Windows 沙箱 ACL 拦死（`SetNamedSecurityInfoW failed (Win32 5): grantWrite(...)`），`python` 无法启动。**你手敲时若看到不同输出，以真实输出为准，并回来改本文**。
- ❌ **不覆盖**：内置中间件逐个用法（第 9 章）、`HumanInTheLoopMiddleware` 的 interrupt 细节（第 14 章）、`transformers` 消息压缩工厂（第 9 章）。

### 7.2 承上启下

- 第 7 章告诉你"结构化输出是策略对象"；本章告诉你"这些策略对象被谁包装"——答案就是 `ModelRequest.response_format` + `wrap_model_call`。
- 第 5 章的 reducer 在本章第一次承受真实压力：`before_*` / `after_*` 返回 dict → 直接喂给 reducer。**多个中间件返回同一个 key，谁赢**？→ 第 9 章讲中间件排序时的冲突处理。
- 第 9 章会把这 12 个钩子"用满"：`SummarizationMiddleware` 怎么在 `before_model` 里改消息、`ToolRetryMiddleware` 怎么在 `wrap_tool_call` 里 try/except、`HumanInTheLoopMiddleware` 怎么用 `jump_to` 退出。

---

## 附录 · 官方写法对照（A8.x）

### A8.1 重试（wrap_model_call 最小可用版）

```python
class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_attempts: int = 3):
        super().__init__()
        self.max_attempts = max_attempts

    def wrap_model_call(self, request, handler):
        for attempt in range(self.max_attempts):
            try:
                return handler(request)
            except Exception:
                if attempt == self.max_attempts - 1:
                    raise
        raise AssertionError("unreachable")
```

要点：`handler` 可以调多次；`request` 不需要改，直接复用（它是不可变的）。

### A8.2 缓存（handler 不调 = 模型不跑）

```python
def wrap_model_call(self, request, handler):
    cached = self.cache.get(key_of(request))
    if cached is not None:
        return cached                      # 短路：模型不会被调用
    response = handler(request)
    self.cache[key_of(request)] = response
    return response
```

### A8.3 动态改工具

```python
def wrap_model_call(self, request, handler):
    if should_add_calculator(request.state["messages"][-1]):
        request = request.override(tools=[*request.tools, CALCULATOR])
    return handler(request)
```

注意：动态加工具后 `model` 节点可能需要回到 `loop_entry`（§2.7 讲的那条条件边）。

### A8.4 只改 system prompt

```python
@dynamic_prompt
def style(request: ModelRequest) -> str:
    return (request.system_prompt or "") + "\n务必先给结论，再给理由。"
```

---

## 认知修正

| # | 之前的错误认知 | 修正后 | 触发点 |
|---|---|---|---|
| 1 | 中间件是回调，被 agent 顺手调用 | 钩子被编译成图节点 | `factory.py:1611-1741` |
| 2 | 用 `hasattr` 判断覆盖 | 必须 `__class__.x is not Base.x` | `factory.py:1163` |
| 3 | before/after 都按列表顺序跑 | before 正序、after **逆序**（穿脱衣服） | `factory.py:1856-1894` |
| 4 | `wrap_*` 也是一种钩子节点 | 不成节点，是 model/ToolNode 内部的洋葱 | `factory.py:1187-1224`、`_:1091-1146` |
| 5 | `request.messages = ...` 可以改 | `__setattr__` 拦截，必须 `override()` | `types.py:169-201` |
| 6 | `return {"jump_to": "end"}` 就能提前结束 | 需编译期 `can_jump_to` 才建边，否则静默失效 | `factory.py:2120`、`_:541-575` |
| 7 | `_execution.py` 是钩子执行引擎 | 是 ShellTool 的沙箱执行策略 | 包内文件名误判 |

---

*本章配套代码：`langchain_learn/08_official_middleware_hooks.py`（A–F 段，零 token；`--real` 才发请求）。*
