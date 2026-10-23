# 第 3 章 · Model 层：字符串变成模型，工具变成 JSON

> 本章目标：**能手画一次 `invoke` 的 6 层调用链，并能说出"我绑的工具"到底在哪一行进入 HTTP 请求体**。预计 80 分钟。

---

## ① 本章你会得到什么

1. 能说出 `init_chat_model("openai:gpt-4o")` 这一行到底经过了哪 4 个阶段，以及为什么它**可能根本不实例化模型**
2. 能画出 `invoke` 的 6 层链路，并指出**哪一层才是供应商代码**（只有一层）
3. 能解释"我明明调的是 `invoke`，为什么它在流式"——三条触发条件 + 一个开关
4. 能说清 `bind_tools` 只干三件事，且**它返回的不是模型**
5. 能自己写一个 60 行的假模型，把整条链路跑通（不发一次网络请求）
6. 能说清缓存键包含什么，从而解释"我加了工具，缓存为什么全失效"
7. 能诊断 `strict=True` 带来的一个隐蔽副作用

第 5 条和本章的分水岭：**能自己实现 `_generate`，你就不再以为"模型"是个黑盒**。

---

## ② 源码解剖

本章碰四个文件：

| 文件 | 里面是什么 |
|---|---|
| `libs/langchain_v1/langchain/chat_models/base.py` | `init_chat_model`、`_BUILTIN_PROVIDERS`、`_ConfigurableModel` |
| `libs/core/langchain_core/language_models/chat_models.py` | `BaseChatModel`、`invoke`、`_generate_with_cache`、`_should_stream`、`_get_llm_string` |
| `libs/core/langchain_core/utils/function_calling.py` | `convert_to_openai_tool` |
| `libs/partners/*/chat_models.py` | 真实 provider 的 `bind_tools` / `_generate`（本章举 Mistral、OpenAI） |

### 2.1 `init_chat_model`：先解析字符串，再决定要不要造对象

真实签名（前三个是重载，实际实现只有一个）：

```python
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
```

函数体只有四个判断，按顺序：

```python
if not model and not configurable_fields:
    configurable_fields = ("model", "model_provider")   # ① 没给 model → 自动变成"可配置"
config_prefix = config_prefix or ""
if config_prefix and not configurable_fields:
    warnings.warn(...)                                   # ② 给了前缀却没给可配置字段 → 警告
if not configurable_fields:
    return _init_chat_model_helper(model, model_provider=model_provider, **kwargs)  # ③ 固定模型：立即实例化
if model: kwargs["model"] = model
if model_provider: kwargs["model_provider"] = model_provider
return _ConfigurableModel(...)                           # ④ 可配置模型：只存配置，不实例化
```

**这一段的全部意义**：`init_chat_model` 不是一个"构造函数"，它是一个**分流器**。同一次调用，可能给你一个真模型，也可能给你一个"还不知道自己是什么模型"的中间对象。

### 2.2 注册表：字符串 → 类

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "openai":    ("langchain_openai", "ChatOpenAI", _call),
    "deepseek":  ("langchain_deepseek", "ChatDeepSeek", _call),
    "ollama":    ("langchain_ollama", "ChatOllama", _call),
    # …共约 29 项，完整清单见源码
}
```

三元组的含义：`(module_path, class_name, creator_func)`。第三项不是摆设，它处理"构造签名不一样"的 provider：

```python
"huggingface": ("langchain_huggingface", "ChatHuggingFace",
                lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs)),
"ibm":         ("langchain_ibm", "ChatWatsonx",
                lambda cls, model, **kwargs: cls(model_id=model, **kwargs)),
```

取工厂函数的过程被 `lru_cache(maxsize=len(_BUILTIN_PROVIDERS))` 缓存——**避免每次建模型都重新 import 一遍模块**。

import 失败时给的报错是特意写过的：

```python
pkg = module.split(".", maxsplit=1)[0].replace("_", "-")
msg = f"Initializing {class_name} requires the {pkg} package. Please install it with `pip install {pkg}`"
```

所以看到 `Initializing ChatDeepSeek requires the langchain-deepseek package` 时，不要怀疑代码，去 `pip install langchain-deepseek -i https://pypi.org/simple`（这台机器默认源不通，理由见第 0 章）。

### 2.3 解析：`_parse_model` 只有三步，但有两个坑

```python
def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    # 第一步：剥前缀（注意 not model_provider 这个前置条件）
    if (not model_provider and ":" in model
            and model.split(":", maxsplit=1)[0] in _BUILTIN_PROVIDERS):
        model_provider = model.split(":", maxsplit=1)[0]
        model = ":".join(model.split(":")[1:])

    # 第二步：猜
    model_provider = model_provider or _attempt_infer_model_provider(model)

    # 第三步：猜不出来就报错；猜出来就归一化
    if not model_provider:
        raise ValueError(f"Unable to infer model provider for {model=}. ...")
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider
```

**坑 1：传了 `model_provider` 就不剥前缀。** 所以

```python
_parse_model("openai:gpt-4o", None)              # → ("gpt-4o",     "openai")
_parse_model("openai:gpt-4o", "azure_openai")    # → ("openai:gpt-4o", "azure_openai")  ← 冒号留在模型名里
```

第二种写法会把 `"openai:gpt-4o"` 原样当模型名发给 Azure，通常直接 404。**结论：用前缀式，或者用 `model_provider=` 时模型名里不要带冒号。**

**坑 2：推断是 best-effort。** 推断规则（`_attempt_infer_model_provider`，全部小写比较）是一串前缀匹配：

| 模型名前缀 | 推断结果 |
|---|---|
| `gpt-` / `o1` / `o3` / `chatgpt` / `text-davinci` | `openai` |
| `claude` | `anthropic` |
| `command` | `cohere` |
| `accounts/fireworks` | `fireworks` |
| `gemini` | `google_vertexai`（**会发 DeprecationWarning**，下个大版本改 `google_genai`） |
| `amazon.` / `anthropic.` / `meta.` | `bedrock` |
| `mistral` / `mixtral` | `mistralai` |
| `deepseek` / `grok` / `sonar` / `solar` | `deepseek` / `xai` / `perplexity` / `upstage` |
| 其它 | `None` → `ValueError` |

第三步的归一化 `replace("-", "_").lower()` 不是小事：`"Azure-OpenAI"` 和 `"azure_openai"` 都能用，就是这个原因。

### 2.4 `_ConfigurableModel`：一个"还没出生的模型"

它是本章最值得读的 40 行。核心是 `__getattr__`：

```python
_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")

def __getattr__(self, name: str) -> Any:
    if name in _DECLARATIVE_METHODS:
        def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
            queued_declarative_operations = list(self._queued_declarative_operations)
            queued_declarative_operations.append((name, args, kwargs))
            return _ConfigurableModel(...)          # 返回一个新对象，把操作记进队列
        return queue
    # 非声明式方法：有默认模型就转发，否则报错
    ...
```

然后真正动手的是 `_model()`：

```python
def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
    params = {**self._default_config, **self._model_params(config)}
    model = _init_chat_model_helper(**params)          # ← 到这里才真的实例化
    for name, args, kwargs in self._queued_declarative_operations:
        model = getattr(model, name)(*args, **kwargs)  # ← 按顺序补上 bind_tools 等
    return model
```

三个必须记住的后果：

1. **`init_chat_model(temperature=0)` 不建模型**。它在等你在 `config={"configurable": {...}}` 里告诉它用哪个 provider。
2. **`configurable_model.bind_tools([...])` 也不绑工具**。它只是往队列里塞了一条记录。真正 `bind_tools` 发生在每次 `invoke` 里的 `_model(config)`。
3. **每次 `invoke` 都重新实例化一次模型**（`_model` 没有缓存）。所以可配置模型比固定模型慢，别在热路径上用。

还有一个安全警告，源码里写得很重：

> `configurable_fields="any"` means fields like `api_key`, `base_url`, etc., can be altered at runtime, potentially redirecting model requests to a different service/user.

**接收外部传入的 config 时，永远枚举 `configurable_fields=(...)`，不要用 `"any"`。**

#### 可配置模型到底什么时候用

先把两件事分开，因为提问里最容易把它们混成一件：

| 你想做的事 | 该用的机制 |
|---|---|
| 下一次 `invoke` 换一个 provider / key / base_url | `configurable_fields` + config |
| agent 循环**内部**第 3 步和第 5 步用不同模型 | middleware 的 `wrap_model_call` 改 `request.model` |
| 只是自己换个模型跑跑 | 什么机制都不用，重新 `get_model(...)` 再传进 `create_agent` |

**"每轮对话后切 config"这个说法只对一半。** config 不是"对话结束后写进去"的状态，它是**每次 `invoke` 的入参**，沿 Runnable 树往下传。一次 `agent.invoke(..., config=A)` 内部可能跑 5 轮 model→tools→model，这 5 轮共享**同一个** A。你把 A 换成 B，影响的是**下一次 `invoke`**，不是这一次的第 5 轮。

真正让"循环内部每步换模型"成立的是另一条路：`create_agent` 的 model 节点每次执行都走 `_get_bound_model(request)`，而 middleware 的 `wrap_model_call` 能改 `request.model`。那条路完全不需要 `configurable_fields`。

**它的真实主场，按使用频率排：**

1. **同一个图，不同租户/用户用不同 key 或 base_url。** config 由外层（HTTP 请求头、DB 里的用户配置）注入，图本身一行不改。这是唯一非它不可的场景——你没法让一张已经 `compile()` 好的图去读请求上下文。
2. **dev / CI / 生产靠 config 切 provider。** 代码零改动，只在部署配置里换 `model` 字段。
3. **工厂函数不让模型在编译期锁死。** `create_agent(model=init_chat_model())` 这种写法，运行时才定模型。
4. **多智能体里给不同角色不同模型。** 靠 `config_prefix` 拆成 `primary_model` / `critic_model` 两组 key。

**代价要算清**：`_model()` 无缓存，每次 `invoke` 重建一次客户端。放进 agent 循环，就是一次对话重建 N 次（N = model 节点被进入的次数）。所以第 3 条"工厂不锁死"听着优雅，代价是每次调用都多一次 `importlib` 查找加对象构造。

**什么时候不该用它**：如果模型在你进程生命周期里是固定的，直接构造好 `ChatOpenAI(...)` 传进去。可配置模型解决的是"**谁**来决定用哪个模型"这个权限问题，不是"**怎么**换模型"这个技术问题——后者有更便宜的解。

> 上面 4 个场景与"每轮重建"的代价，是**从源码推的**（`_ConfigurableModel._model` 无缓存 + `_init_chat_model_helper` 每次构造），没有实测计时。第一个场景（多租户注入 config）也没有在本机跑通，因为 `pwsh` 被沙箱 ACL 挡住。

### 2.5 `BaseChatModel` 把方法分成两张表

类文档串里就有这张对照表，直接抄下来当心智模型：

| 声明式（造新 Runnable，不调模型） | 干什么 |
|---|---|
| `bind_tools` | 造一个"会调工具"的模型 |
| `with_structured_output` | 造一个"按 schema 出结构化结果"的包装 |
| `with_retry` / `with_fallbacks` | 加重试 / 加兜底模型 |
| `configurable_fields` / `configurable_alternatives` | 运行时可换参数 / 换模型 |

| 命令式（真发请求） | 输出 |
|---|---|
| `invoke` / `ainvoke` | `AIMessage` |
| `stream` / `astream` | `Iterator[BaseMessageChunk]` |
| `batch` / `abatch` | `list[BaseMessage]` |
| `astream_events` | `on_chat_model_start` / `on_chat_model_stream` / `on_chat_model_end` |

**声明式方法全部返回 Runnable，不返回模型。** 这是本章最重要的一句话，2.8 会用上。

### 2.6 一次 `invoke` 的 6 层

```python
# 第 1 层：统一输入
def invoke(self, input, config=None, *, stop=None, **kwargs) -> AIMessage:
    config = ensure_config(config)
    return self.generate_prompt(
        [self._convert_input(input)],       # ← str → StringPromptValue；list → ChatPromptValue
        stop=stop,
        callbacks=config.get("callbacks"),
        ...
    ).generations[0][0].message              # ← ChatResult → ChatGeneration → AIMessage

# 第 2 层：PromptValue → 消息列表
def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs) -> LLMResult:
    prompt_messages = [p.to_messages() for p in prompts]
    return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

# 第 3 层：generate → 逐个调 _generate_with_cache（这里还有 callback 管理、批处理）
# 第 4 层：_generate_with_cache（见 2.7，五个关卡都在这里）
# 第 5 层：self._generate(messages, stop=stop, **kwargs)   ← 抽象方法，供应商唯一必须实现的
# 第 6 层：供应商里 _generate → 构造 payload → 发 HTTP → 解析回 ChatResult
```

`_convert_input` 的三种输入形态（记住这个，能少写很多 try）：

```python
if isinstance(model_input, PromptValue):  return model_input
if isinstance(model_input, str):          return StringPromptValue(text=model_input)
if isinstance(model_input, Sequence):     return ChatPromptValue(messages=convert_to_messages(model_input))
raise ValueError(...)
```

#### callbacks：不是"钩子开关"，是旁路观察通道

先把因果说正：**你不是用 `callbacks` 去"设置" `on_chat_model_start`**。这两个事件的位置是框架在 `generate()` 里写死的，你提供的是"事件发生时该通知谁"。

```python
def generate(self, messages, stop=None, callbacks=None, *, tags=None, metadata=None, run_name=None, **kwargs) -> LLMResult:
    params = self._get_invocation_params(stop=stop, **kwargs)
    callback_manager = CallbackManager.configure(
        callbacks, self.callbacks, self.verbose,   # 继承来的 / 本地默认 / verbose
        tags, self.tags,                            # 继承 tags / 本地 tags
        metadata, self.metadata,                    # 继承 metadata / 本地 metadata
    )
    run_managers = callback_manager.on_chat_model_start(   # ← 触发点 1
        {**self._identifying_params},
        messages,
        invocation_params=params,        # ← 本次实际要发的参数（model/temperature/stop…）
        options={"stop": stop},
    )
    results = [
        self._generate_with_cache(m, stop=stop, run_manager=run_managers[i] if run_managers else None, **kwargs)
        for i, m in enumerate(messages)
    ]
    output = LLMResult(
        generations=[r.generations for r in results],
        llm_output=self._combine_llm_outputs([r.llm_output for r in results]),
    )
    for run_manager in run_managers:
        run_manager.on_llm_end(output)                         # ← 触发点 2
    return output
```

`run_managers` 是 **list**，因为 `messages` 是 `list[list[BaseMessage]]`：`invoke` 只塞一条，`batch` 塞 N 条，每条 prompt 一个独立 run。这也解释了为什么调试 batch 时你会看到 N 组 start/end。

**chat model 这一层只有 4 个事件：**

| 事件 | 谁触发 | 你能拿到什么 |
|---|---|---|
| `on_chat_model_start` | `generate()` 开头 | `messages` 全量 + `invocation_params`（真正要发出去的参数）|
| `on_llm_new_token` | `_stream` / `_astream` 每吐一个 chunk | 增量 token（非流式调用时根本不触发）|
| `on_llm_end` | `generate()` 末尾 | `LLMResult`（含 token usage）|
| `on_llm_error` | 异常路径 | 异常对象 |

再往上一层还有 `on_chain_start/end/error`、`on_tool_start/end/error`、`on_retriever_*`、`on_retry`。`astream_events` 不是新东西，它就是把这些回调翻译成统一事件流的消费者。

**四个注入途径**（`CallbackManager.configure` 会合并；注意「合并」在 handler / tags / metadata 上规则**各不相同**，见下面的实测表）：

```python
model.invoke(msgs, config={"callbacks": [my_handler]})   # ① 最常用：只影响这一次调用及其子树
ChatOpenAI(..., callbacks=[my_handler])                  # ② 构造期默认 handler
ChatOpenAI(..., verbose=True)                            # ③ 装一个打印到 stdout 的 debug handler
model.with_config(callbacks=[my_handler])                # ④ 绑在 Runnable 上，跟着它走
```

**三个容易看错的地方：**

1. **start/end 的条数 ≠ HTTP 请求数。** 这两个触发点在 `generate()` 里，不在 `_generate_with_cache` 里。所以缓存命中时你照样收到 start + end，而 `on_llm_new_token` 一条都没有。
2. **`_generate` 签名里没有 `run_manager`，就永远收不到 token 事件。** 基类用 `_signature_parameters(self._generate).get("run_manager")` 判断（就是 2.7 关卡 3 那一行），没有就当你不支持回调，静默不传。这是自定义 provider "流式不生效"最常见的原因。
3. **handler 里抛异常默认不会中断模型调用。** `BaseCallbackHandler.raise_error` 默认 `False`，事件分发时捕获并记日志；想让它抛，得在 handler 类上写 `raise_error = True`。

#### 实跑记录：上面四条落成数字

环境：`langchain 1.4.6` / `langchain-core 1.6.12` / `langchain-openai 1.7.2`，模型 `qwen3:latest`（本地 ollama，`base_url=http://localhost:11434/v1`）。
对应代码：`langchain_learn/03_callbacks_demo.py`（A/A2/B 段零 token）与 `langchain_learn/03_callback_params_trace.py`（B–F 段）。

**① `configure` 的合并规则，三种字段三种行为**（`03_callbacks_demo.py` B 段）：

```python
cm = CallbackManager.configure(
    inheritable_callbacks=[TraceHandler()], local_callbacks=[TraceHandler()],
    inheritable_tags=["from-config"],       local_tags=["from-self"],
    inheritable_metadata={"layer": "config"}, local_metadata={"layer": "self"},
)
```

实测：`handlers 数量 = 2`（**拼接，不去重**）｜`tags = ['from-config', 'from-self']`（**拼接**）｜`metadata = {'layer': 'self'}`（**字典覆盖，本地赢**）。

一句话记住：**handler 和 tags 做加法，metadata 做覆盖。** 这条差异有实际后果——你往上挂 `{"user_id": ...}`，里层同名键会把外层悄悄盖掉；tags 则永远不会被吃掉，所以"给某次请求打标"用 tags 比用 metadata 可靠。

**② 构造函数里的 callbacks 不会自动变成"可继承"**（A 段）：

```text
configure(inheritable_callbacks=None, local_callbacks=[TraceHandler()])
  handlers            = ['TraceHandler']
  inheritable_handlers= []          # ← 注意这里
```

`handlers` 是"这次通知谁"，`inheritable_handlers` 是"往下传谁"。只从 `local_callbacks` 进来的 handler 进 `handlers` 但**不进** `inheritable_handlers`——真正沿 tree 下钻的继承是 `Runnable.with_config` / `config` 那条链在做，不是构造函数。所以"为什么我 `ChatOpenAI(callbacks=[h])` 收不到 agent 里 tool 的事件"的答案是：它本来就不负责那个。

**③ `run_id.hex[:8]` 不是唯一标识——别拿它当 id 用**（A2 段，一次 batch 两条 prompt）：

```text
run_manager[0].run_id = 01a1619c-d56d-7c10-a353-fa4968d7391a
run_manager[1].run_id = 01a1619c-d56d-7c10-a353-fa56fdbe5abc
                         ^^^^^^^^ 前 8 位一模一样
```

langsmith 的 `run_id` 是 **UUIDv7**（时间戳在前），同一毫秒内产生的两个 run 前缀相同。日志里只截 8 位，会把两次不同调用打成同一个 id——排障时尤其致命。要截就截**尾部** 8 位，或者干脆打全。

**④ temperature 的三处写法，谁盖谁（实测 `invocation_params` 里的值）**

| 写法 | 实测 `temperature` | 结论 |
|---|---|---|
| `ChatOpenAI(temperature=0.0)` + `invoke(...)` | `0.0` | 构造值作为默认 |
| `ChatOpenAI(temperature=0.0)` + `invoke(..., temperature=0.9)` | `0.9` | **调用期 kwargs 赢** |
| `ChatOpenAI(temperature=0.0).bind(temperature=0.7)` + `invoke(...)` | `0.7` | **bind 赢构造，但输给调用期 kwargs** |

优先级：**`invoke` 的 kwargs > `bind` 的预置 > 构造参数**。

**⑤ 真发出去的 payload 里，没有 callbacks / tags / metadata**（D 段，靠覆写 `_get_request_payload` 偷看）：

```text
[wire payload] {'model': 'qwen3:latest', 'stream': False, 'temperature': 0.9, 'max_completion_tokens': 32}
[payload 里有 callbacks 吗] False
```

两个顺带的细节：`invoke(max_tokens=32)` 在请求体里叫 **`max_completion_tokens`**（langchain-openai 按新 OpenAI 规范改名，老字段 `max_tokens` 你写、它转）；`meta_keys` 里那堆 `ls_provider` / `ls_model_name` / `lc_versions` 不是你传的，是 langchain 自己注入的 tracing 元数据。

**⑥ 流式下的两个"意外"**（`03_callbacks_demo.py` C3 段）

```text
[token] ''   × 31     ← qwen3 的思考段，content 为空，但事件照样发
[token] '一' [token] '、' [token] '二' ...
[end] usage=None      ← 流式结束拿不到 usage_metadata
```

同一份 handler，非流式 `invoke` 的 `on_llm_end` 能拿到完整 `{'input_tokens':15,'output_tokens':433,'total_tokens':448}`，流式却是 `None`——因为 `stream_usage` 默认不开。想流式也要 usage，构造模型时加 `stream_usage=True`。

而前面那 31 个空 token 是**思维链阶段**：模型在 reasoning 字段里吐字，`content` 还是空的。做打字机效果时这些空 token 会让前端"先卡住 30 帧"，得靠 `additional_kwargs` 里的 reasoning 字段分流。

**⑦ 进了 agent，事件流长这样**（`03_callback_params_trace.py` E 段，一次"北京天气"问答，共 10 条事件）

```text
chain_start   name=LangGraph  root=True   tags=['agent-demo']
chain_start   name=model      root=False  tags=['graph:step:1','agent-demo']
chat_model_start  msgs_per_prompt=[1]  meta_keys=[..., 'langgraph_node', 'langgraph_step', 'langgraph_triggers', 'langgraph_path', 'checkpoint_ns']
llm_end
chain_start   name=tools      root=False  tags=['graph:step:2','agent-demo']
tool_start    name=get_weather  input_str={'city': '北京'}   parent_is_node=True
tool_end      output=content='北京：晴，26℃' ...
chain_start   name=model      root=False  tags=['graph:step:3','agent-demo']
chat_model_start  msgs_per_prompt=[3]     ← 第二轮带上了 AI 的 tool_calls + ToolMessage
llm_end
```

三个能直接拿去排障的点：

1. **节点名就是 `chain_start.name`**：`LangGraph`（整图）→ `model` → `tools` → `model`。你在 LangSmith 里看到的那些框，就是这三个。
2. **`langgraph_step` / `langgraph_node` 在 metadata 里**——想知道"这次模型调用是第几步、哪个节点发的"，不用猜，metadata 直接给。
3. **`tool_start` 的 `input_str` 是 dict，不是 JSON 字符串**。基类签名写的是 `input_str: str`，但 v1 的 `ToolNode` 直接把参数字典传进来了。你要打日志就得自己 `json.dumps`，别指望它已经是字符串——按类型标注去写 `input_str[:60]` 会抛 `TypeError`。

**⑧ `raise_error` 的默认值**：`langchain_core/callbacks/base.py` 里 `BaseCallbackHandler.raise_error = False`。这条是**读源码得到的，没有实测**——验证它正好是本章的破坏实验：把某个 handler 方法里加一句 `1/0`，跑一次 `invoke`，模型照常返回；再把类属性改成 `raise_error = True`，同一个 `1/0` 会把 `ZeroDivisionError` 抛到你的调用栈上。两步都做完，你就再也不会搞混"回调出错"和"模型出错"。

### 2.7 `_generate_with_cache`：五个关卡，顺序不能乱

```python
def _generate_with_cache(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    # 关卡 0：要不要查缓存
    llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
    check_cache = self.cache is not False
    if check_cache:
        if llm_cache is not None:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            normalized_messages = [
                msg.model_copy(update={"id": None}) if getattr(msg, "id", None) is not None else msg
                for msg in messages
            ]
            prompt = dumps(normalized_messages)
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=self._convert_cached_generations(cache_val))   # 命中即返回
        elif self.cache is None:
            pass
        else:
            raise ValueError("Asked to cache, but no cache found at `langchain.cache`.")

    # 关卡 1：限流（放在缓存之后 —— 缓存命中不该被限流）
    if self.rate_limiter:
        self.rate_limiter.acquire(blocking=True)

    # 关卡 2/3/4：v2 流式 / v1 流式 / 普通 _generate，三选一
    if self._should_use_protocol_streaming(...):  ...
    elif self._should_stream(...):                ... self._stream(...)
    elif _signature_parameters(self._generate).get("run_manager"):
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    else:
        result = self._generate(messages, stop=stop, **kwargs)

    # 关卡 5：给结果补元数据，再写缓存
    for generation in result.generations:
        generation.message.response_metadata = _gen_info_and_msg_metadata(generation)
    if check_cache and llm_cache is not None:
        llm_cache.update(prompt, llm_string, result.generations)
```

四个可以直接拿来 debug 的事实：

- **缓存键 = 消息（去掉 `id`）+ `llm_string`。** 去掉 `id` 是为了"同一条消息重发也能命中"。
- **`llm_string` 里包含本次调用的 `kwargs`。** 所以 `bind_tools` 换了工具集 → 缓存键变了 → 旧缓存全部不命中。这就是"我加了个工具，缓存好像没了"的答案。
- **限流在缓存之后。** 命中缓存不消耗配额，这是刻意的。
- **`_generate` 有没有 `run_manager` 参数会被检查。** 你的自定义模型带不带 `run_manager` 都能跑，但带了才有 token 级回调。

紧接着看 `_get_llm_string`，它解释了缓存键的另一半：

```python
def _get_llm_string(self, stop: list[str] | None = None, **kwargs: Any) -> str:
    if self.is_lc_serializable():
        params = {**kwargs, "stop": stop}
        param_string = str(sorted(params.items()))
        serialized_repr = self._serialized
        _cleanup_llm_representation(serialized_repr, 1)
        return json.dumps(serialized_repr, sort_keys=True) + "---" + param_string
    params = {**self._get_invocation_params(stop=stop, **kwargs)}
    params = {**params, **kwargs}
    return str(sorted(params.items()))
```

**能序列化的模型，缓存键是"模型全量配置 + 本次参数"；不能序列化的，只剩"调用参数"。** 后者意味着：两个不同实例只要参数相同，缓存会互相串。自己写模型时想清楚要不要 `is_lc_serializable()` 返回 `True`。

### 2.8 为什么你调 `invoke` 却在流式

`_should_stream` 的判定方向是"**流式是默认，非流式是例外**"：

```python
def _streaming_disabled(self, **kwargs: Any) -> bool:
    if self.disable_streaming is True:                        return True
    if self.disable_streaming == "tool_calling" and kwargs.get("tools"):
        return True                                           # ← 带工具就退回非流式
    if "stream" in kwargs and not kwargs["stream"]:           return True
    return "streaming" in self.model_fields_set and getattr(self, "streaming", None) is False
```

三条触发 + 四条关闭。最实用的那一条是 `disable_streaming="tool_calling"`，源码注释把意图写明了：

> `'tool_calling'` … LangChain will automatically switch to non-streaming behavior (`invoke`) only when the tools argument is provided. This offers the best of both worlds.

**如果你的供应商流式下工具调用的 JSON 会拼坏，把这个开关打开**，代价是带工具的那一次拿不到 token 级输出。

另一种隐式触发是 `astream_events()` / `astream_log()`：即使你只调了 `invoke`，只要挂了流式回调，它也走 `_stream` 再聚合成完整消息。

### 2.9 `bind_tools`：只干三件事

基类里它是空的：

```python
def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs):
    raise NotImplementedError
```

**因为工具协议是供应商特有的。** 看一个真实实现（Mistral，最干净的一份）：

```python
def bind_tools(self, tools, tool_choice=None, **kwargs):
    formatted_tools = [convert_to_openai_tool(tool) for tool in tools]      # ① 转成 JSON
    if tool_choice:
        tool_names = [...]
        if tool_choice in tool_names:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        else:
            kwargs["tool_choice"] = tool_choice                             # ② 归一化 tool_choice
    return super().bind(tools=formatted_tools, **kwargs)                    # ③ 存进 RunnableBinding
```

三件事之后它**返回一个 `RunnableBinding`，不是模型**。绑定物不会立刻发出去，而是挂在 `kwargs` 里等下一次调用。

到了调用那一刻，OpenAI 的实现把它们并进请求体：

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
    messages = self._convert_input(input_).to_messages()
    if stop is not None:
        kwargs["stop"] = stop
    payload = {**self._default_params, **kwargs}   # ← 绑定进来的 tools / tool_choice 在这一行合并
    ...
    payload["messages"] = [_convert_message_to_dict(m) for m in messages]
    return payload

def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    raw_response = self.client.with_raw_response.create(**payload)
    return self._create_chat_result(raw_response.parse())
```

**整条链闭合了**：`bind_tools` 存 kwargs → `bind` 造 RunnableBinding → `invoke` 时 kwargs 一路透传 → `_get_request_payload` 合并成 payload → HTTP。

### 2.10 `convert_to_openai_tool`：八种输入，一个出口

这是"你写的东西"到"模型看懂的东西"的唯一转换器。分派顺序（顺序本身有含义）：

```python
if isinstance(tool, dict) and tool.get("type") in _WellKnownOpenAITools:
    return tool                                   # ① 已经是内置工具（file_search / web_search / mcp …）原样返回
if isinstance(tool, Tool) and (tool.metadata or {}).get("type") == "custom_tool":
    return {"type": "custom", ...}                # ② 自定义工具（自由文本输入）
oai_function = convert_to_openai_function(tool)   # ③ 其余全部走这里
return {"type": "function", "function": oai_function}
```

`convert_to_openai_function` 内部按输入类型分支：

| 你给的东西 | 走哪条 |
|---|---|
| Anthropic 风格 `{"name","input_schema"}` | 改键名成 `parameters` |
| Bedrock Converse 风格 `{"toolSpec": ...}` | 取 `toolSpec.inputSchema.json` |
| 已经是 OpenAI function `{"name": ...}` | 只保留 `name/description/parameters/strict`（**其余键会被丢掉**） |
| 带 `title` 的 JSON Schema | `title` → `name`，`description` 提出来，剩下的当 `parameters` |
| Pydantic 类 | `model_json_schema()` → `_rm_titles` → 见下 |
| `TypedDict` | 先转成 pydantic v1 模型再转 |
| `BaseTool` | 用 `tool.tool_call_schema`（**第 4 章的双 schema，注入参数在这里被排除**） |
| 普通函数 | `create_schema_from_function(..., parse_docstring=True, include_injected=False)` |

两个细节值得单独记：

**`_rm_titles`**：pydantic 生成的 schema 到处是 `"title"`，全都删掉——除了嵌在 `properties` 里的。删是因为它纯属噪声，白占 token。

**`strict=True` 的真实副作用**（`_recursive_set_additional_properties_false`）：

```python
properties = schema.get("properties")
if isinstance(properties, dict) and properties:
    schema["required"] = list(properties.keys())     # ← 所有字段都变成必填
...
schema["additionalProperties"] = False
```

OpenAI strict 模式要求"每个字段都在 required 里"，所以它**会把可选参数强行变成必填**。你的工具里有 `Optional[str] = None` 的参数时，开 `strict=True` 会让模型被迫编造这个字段。

### 2.11 回来的路：`_create_chat_result`

```python
for res in choices:
    message = _convert_dict_to_message(res["message"])       # ← 第 2 章那张表
    if token_usage and isinstance(message, AIMessage):
        message.usage_metadata = _create_usage_metadata(token_usage, service_tier)
    generation_info["finish_reason"] = res.get("finish_reason")
    generations.append(ChatGeneration(message=message, generation_info=generation_info))

llm_output = {
    "token_usage": token_usage,
    "model_provider": "openai",
    "model_name": response_dict.get("model", self.model_name),
    "system_fingerprint": response_dict.get("system_fingerprint", ""),
}
```

然后第 2.7 节里那一行会把 `llm_output` 合进 `response_metadata`：

```python
if len(result.generations) == 1 and result.llm_output is not None:
    result.generations[0].message.response_metadata = {
        **result.llm_output,
        **result.generations[0].message.response_metadata,
    }
```

**这就是为什么你能在 `response_metadata` 里看到 `model_name`、`finish_reason`、`system_fingerprint`**——它们是从 `llm_output` 合并进来的，不是供应商直接放在消息里的。

注意 `_convert_dict_to_message` 里这段，它把第 2 章接上了：

```python
if raw_tool_calls := _dict.get("tool_calls"):
    for raw_tool_call in raw_tool_calls:
        try:
            tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
        except Exception as e:
            invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
```

**坏 JSON 在这里被降级成 `invalid_tool_calls`，不抛异常。** 所以"模型说要调工具但 `tool_calls` 是空的"——先看 `invalid_tool_calls`。

---

## ③ 你敲这段

新建 `langchain_learn/03_model_layer.py`。**A–E 段完全不联网**，只有 F 段真发请求。

```python
"""第 3 章：Model 层。A–E 段离线，F 段才真发请求。

对应教材：docs/langchain-family/langchain_learn/04-ch03-model-layer.md
运行：在项目根目录 python langchain_learn/03_model_layer.py
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

# ============ A. 字符串怎么被解析 ============
print("=" * 64)
print("A. _parse_model：字符串 → (模型名, provider)")
print("=" * 64)
from langchain.chat_models.base import (
    _BUILTIN_PROVIDERS,
    _attempt_infer_model_provider,
    _parse_model,
)

print("注册表里的 provider 数量:", len(_BUILTIN_PROVIDERS))
for raw, prov in [
    ("openai:gpt-4o", None),
    ("gpt-4o", None),
    ("claude-sonnet-4-5", None),
    ("deepseek-chat", None),
    ("openai:gpt-4o", "azure_openai"),   # ← 坑 1：前缀不剥
]:
    print(f"  _parse_model({raw!r}, {prov!r}) -> {_parse_model(raw, prov)}")

try:
    _parse_model("my-private-llm", None)
except ValueError as e:
    print("  猜不出来时的报错:", str(e).split("\n")[0])


# ============ B. 固定模型 vs 可配置模型 ============
print("\n" + "=" * 64)
print("B. init_chat_model 是个分流器")
print("=" * 64)
from langchain.chat_models import init_chat_model

fixed = init_chat_model("openai:gpt-4o", api_key="sk-dummy", temperature=0)
print("  给了 model      ->", type(fixed).__name__)

cfg = init_chat_model(temperature=0)
print("  没给 model      ->", type(cfg).__name__)
print("  对它 bind_tools ->", type(cfg.bind_tools([{"name": "x", "parameters": {}}])).__name__)
print("  仍然是排队状态，没有真的绑定")


# ============ C. 一个工具变成三段 JSON ============
print("\n" + "=" * 64)
print("C. convert_to_openai_tool：你写的 → 模型看的")
print("=" * 64)


@tool
def get_weather(city: str, unit: str = "c") -> str:
    """查询指定城市当前天气。

    Args:
        city: 城市名，例如 杭州。
        unit: 温度单位，c 或 f。
    """
    return f"{city} 26{unit}"


print(json.dumps(convert_to_openai_tool(get_weather), ensure_ascii=False, indent=2))
print("\n注意三件事：没有 title（被 _rm_titles 删了）、type 是 function、parameters 里只有 city 是 required")


# ============ D. 自己写一个假模型，把链路跑通 ============
print("\n" + "=" * 64)
print("D. 最小 ChatModel：只实现 _generate 和 _llm_type")
print("=" * 64)

SEEN = []   # 记录每次 _generate 收到的 kwargs，用来证明 bind_tools 的透传


class EchoChatModel(BaseChatModel):
    """不发网络请求，把收到的消息和 kwargs 打回来。"""

    reply: str = "ok"

    @property
    def _llm_type(self) -> str:
        return "echo"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "echo", "reply": self.reply}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        SEEN.append({"n_messages": len(messages), "types": [m.type for m in messages], "kwargs": dict(kwargs)})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)


echo = EchoChatModel()
out = echo.invoke("你好")
print("  直接 invoke        ->", type(out).__name__, repr(out.content))
print("  _generate 收到     ->", SEEN[-1])

bound = echo.bind_tools([get_weather], tool_choice="get_weather")
print("  bind_tools 返回类型 ->", type(bound).__name__)
bound.invoke([HumanMessage(content="杭州天气？")])
print("  绑定后 kwargs 里多了 ->", list(SEEN[-1]["kwargs"].keys()))
print("  tools[0].function.name ->", SEEN[-1]["kwargs"]["tools"][0]["function"]["name"])
print("  tool_choice           ->", SEEN[-1]["kwargs"]["tool_choice"])


# ============ E. 缓存键：加工具 = 换缓存 ============
print("\n" + "=" * 64)
print("E. _get_llm_string：缓存键里有什么")
print("=" * 64)
k1 = echo._get_llm_string(stop=None)
k2 = echo._get_llm_string(stop=None, tools=[convert_to_openai_tool(get_weather)])
print("  无工具键长:", len(k1))
print("  有工具键长:", len(k2))
print("  两者相同吗:", k1 == k2, "  ← 不同就说明加了工具缓存必然失效")


# ============ F. 真请求（需要 .env） ============
print("\n" + "=" * 64)
print("F. 真模型：元数据从哪来")
print("=" * 64)
from _shared import get_model

real = get_model()
msg = real.invoke("用一句话说明什么是 LangChain。")
print("  type          :", type(msg).__name__)
print("  content       :", (msg.content or "")[:60])
print("  usage_metadata:", msg.usage_metadata)
print("  response_meta :", {k: msg.response_metadata.get(k) for k in
                            ("model_name", "model_provider", "finish_reason", "system_fingerprint")})
print("  msg.id        :", msg.id)

# G. 破坏实验：见教材 ⑤
```

### 逐段说清"为什么这么写"

| 段 | 为什么 |
|---|---|
| A | 直接调私有函数，是为了让你看见"解析"和"实例化"是**两件独立的事**，中间没有魔法 |
| B | `fixed` 和 `cfg` 必须是不同的两个类。你以后看到 `init_chat_model` 返回值"不能调 `.bind_tools` 的某些属性"，原因在这里 |
| C | 这一屏是全章最值钱的输出：**这就是你付 token 买到的东西** |
| D | 只实现 2 个成员就能跑。`SEEN` 用模块级列表而不是 `self.xxx`，因为 `BaseChatModel` 是 pydantic 模型，**给未声明字段赋值会直接报错**——这个坑你一定会踩 |
| E | 用同一个 `echo` 实例只改 `tools` 参数，能干净地证明"只有 kwargs 变了" |
| F | 打印 `response_metadata` 的四个键，正好对应 2.11 里 `llm_output` 的四个键 |

---

## ④ 你应该看到

> 以下是**按源码推导的期望值**，我没有在你机器上跑过（`pwsh` 被 Windows 文件权限拦住，见第 0 章）。第一次跑若有差异，先查版本，再怀疑我。

**A 段**

```
注册表里的 provider 数量: 29
  _parse_model('openai:gpt-4o', None) -> ('gpt-4o', 'openai')
  _parse_model('gpt-4o', None) -> ('gpt-4o', 'openai')
  _parse_model('claude-sonnet-4-5', None) -> ('claude-sonnet-4-5', 'anthropic')
  _parse_model('deepseek-chat', None) -> ('deepseek-chat', 'deepseek')
  _parse_model('openai:gpt-4o', 'azure_openai') -> ('openai:gpt-4o', 'azure_openai')
  猜不出来时的报错: Unable to infer model provider for model='my-private-llm'.
```

provider 数量随版本变（29 是 master 这一刻的值），**别把它当断言**。最后一行那个 `('openai:gpt-4o', 'azure_openai')` 是本段唯一需要你盯住的：**冒号没被剥掉**。

**B 段**

```
  给了 model      -> ChatOpenAI
  没给 model      -> _ConfigurableModel
  对它 bind_tools -> _ConfigurableModel
```

第三行是关键：**绑定之后类型没变**，因为它还在排队。

**C 段**

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询指定城市当前天气。",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "城市名，例如 杭州。"},
        "unit": {"type": "string", "description": "温度单位，c 或 f。", "default": "c"}
      },
      "required": ["city"]
    }
  }
}
```

三个细节和一个提醒：

- 没有 `"title"`（`_rm_titles` 删了）
- `unit` 有 `"default": "c"` 但**不在 required 里**
- `description` 是 docstring 的第一段（summary），**不含 Args 段**
- 参数描述来自 Google style docstring 的 `Args:`；如果你写成 numpy style，这里就没有参数描述。第 4 章 §2.4 讲过这条

**D 段**

```
  直接 invoke        -> AIMessage 'ok'
  _generate 收到     -> {'n_messages': 1, 'types': ['human'], 'kwargs': {}}
  bind_tools 返回类型 -> RunnableBinding
  绑定后 kwargs 里多了 -> ['tools', 'tool_choice']
  tools[0].function.name -> get_weather
  tool_choice           -> get_weather
```

注意 `_generate 收到` 里的 `'types': ['human']`——`str` 输入在 `_convert_input` 变成了 `HumanMessage`，到 `_generate` 时**已经是消息对象了**。

还有：`bind_tools 返回类型` 是 `RunnableBinding`。第 2.5 节那张表说的"声明式方法返回 Runnable"，在这里变成了实物。

**E 段**

```
  无工具键长: 60
  有工具键长: 187
  两者相同吗: False   ← 不同就说明加了工具缓存必然失效
```

长度数字取决于你的 `reply` 值，**不要求一致**；要求一致的是最后那个 `False`。

**F 段**（需要 `.env`，模型名以你的 `LLM_MODEL_ID` 为准）

```
  type          : AIMessage
  content       : LangChain 是一个用于构建大模型应用的开发框架……
  usage_metadata: {'input_tokens': 24, 'output_tokens': 31, 'total_tokens': 55}
  response_meta : {'model_name': '...', 'model_provider': 'openai', 'finish_reason': 'stop', 'system_fingerprint': '...'}
  msg.id        : lc_run-...
```

`usage_metadata` 的键名是 LangChain 统一的 `input_tokens`/`output_tokens`/`total_tokens`，**不是**供应商原始的 `prompt_tokens`/`completion_tokens`。原始值在 `response_metadata["token_usage"]` 里。

如果你用的是 DeepSeek 或其它 OpenAI 兼容端点，`system_fingerprint` 很可能是空串——**正常，别当 bug**。

---

## ⑤ 三个破坏性实验

### 实验 A：把 `_llm_type` 删掉

```python
class Broken(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])
```

**期望**：`TypeError: Can't instantiate abstract class Broken with abstract method _llm_type`（pydantic 的 ABC 在实例化时拦住）。

**你要读懂的事**：`_llm_type` 不是装饰性属性，它参与日志与追踪。`BaseChatModel` 只有这一个抽象属性 + 一个抽象方法，**这就是"最少要写多少"的答案**。

### 实验 B：`init_chat_model` 用错 provider 名

```python
init_chat_model("open-ai:gpt-4o")     # 下划线写成连字符
```

**期望**：`open-ai` 不在注册表里 → 前缀不剥（因为 `in _BUILTIN_PROVIDERS` 为假）→ 整串去猜 → `Unable to infer model provider for model='open-ai:gpt-4o'`，后面跟一长串 supported providers。

**你要读懂的事**：错误信息里那句 `Supported providers: ...` 是 `sorted(_BUILTIN_PROVIDERS)` 拼的。**遇到 provider 问题，先看这个列表里有没有你写的那串**。

### 实验 C：`strict=True` 把可选参数变必填

```python
print(json.dumps(convert_to_openai_tool(get_weather, strict=True), ensure_ascii=False, indent=2))
```

**期望**：`parameters` 里多出 `"additionalProperties": false`，并且 **`required` 变成 `["city", "unit"]`**——`unit` 那个有默认值的参数被强行required 了。

**你要读懂的事**：strict 模式是"用可选性换结构保证"。你只有 `Optional` 真的不该出现时（比如抽取任务里的日期格式），才值得开。

### 附加实验（想做就做）：让 `invoke` 走流式

```python
echo.disable_streaming = False
for chunk in echo.stream("你好"):
    print(type(chunk).__name__, repr(chunk.content))
```

**期望**：`_stream` 没实现 → 基类 `_stream` 抛 `NotImplementedError`，但 `stream()` 在 `_should_stream` 为假时会退化成"跑 `invoke` 再吐一块"。

这条实验的目的不是看输出，是让你理解 **`disable_streaming` 与 `_stream` 是否实现，是两件独立的事**。

---

## ⑥ 自检题

1. `init_chat_model("openai:gpt-4o")` 和 `init_chat_model(temperature=0)` 返回的是两个不同的类。为什么作者要这么设计？
2. `bind_tools` 返回的为什么不是模型？如果返回模型，会丢掉什么能力？
3. 你给消息加了 `id`，然后发现缓存命中率骤降。为什么？
4. `strict=True` 为什么要把所有 properties 塞进 `required`？
5. 你写了个自定义 `ChatModel`，`invoke` 正常，但 `astream_events` 收不到 token 级事件。最可能是哪一步漏了？

### 参考答案

> 先自己做，再对答案。每条后面的 **延伸** 是超出题面的部分，用来连到后面的章。

**1. 两个不同的类，为什么这么设计？**

因为它们服务两种截然不同的需求：`ChatOpenAI` 是**此刻就确定**的模型，`_ConfigurableModel` 是**运行时才确定**的模型（可能连 provider 都不知道）。

后者的存在意义写在源码的安全警告旁边：它让"同一个流程跑在 gpt / claude / 本地模型上"变成配置问题，而不是代码问题。代价是它**不是** `BaseChatModel`，所以任何依赖"我是 chat model"的代码都得走 `_model()` 先落地。

> **延伸**：`_ConfigurableModel` 每次 `invoke` 都重新实例化一次（`_model()` 无缓存），而 provider 实例化会走 `importlib.import_module`——虽然被 `lru_cache` 挡住了重复 import，但构造对象本身不便宜。**在 agent 循环里用可配置模型，等于每一轮都重建一次客户端。** 这是我在你的 `rfwp_agent` 那类工程里会优先排查的性能点。

**2. `bind_tools` 为什么返回 Runnable？返回模型会丢什么？**

因为"绑定"是**声明**，不是**状态变更**。返回 `RunnableBinding` 让你可以：

- `base.bind_tools([a]).bind_tools([b])` 层层叠加而不污染原对象
- 同一个 `base` 派生出"N 个工具集版本"，共享底层客户端
- 把 `base.bind_tools(...)` 当管道的一段：`prompt | model.bind_tools(...) | parser`

如果它返回模型，你就得 `copy.deepcopy`，客户端、连接池、回调全都得跟着复制。

> **延伸**：`RunnableBinding` 属于 Runnable 家族的"装饰器"，`with_retry`、`with_fallbacks`、`with_config` 全是同一套机制。理解了一个，你就理解了全部——这也是第 10 章流式时 `astream_events` 能穿透多层包装的原因。

**3. 加了 `id` 为什么缓存命中率骤降？**

因为缓存里存的是 `dumps(normalized_messages)`，而 `normalized_messages` 会把每条消息的 `id` 置成 `None`：

```python
normalized_messages = [
    msg.model_copy(update={"id": None}) if getattr(msg, "id", None) is not None else msg
    for msg in messages
]
```

**等一下**——置成 `None` 意味着 `id` 本该不影响命中。所以真正的答案是：如果你看到命中率下降，**不是 `id` 本身造成的，而是 `id` 让 `msg.model_copy()` 产生了新对象**，进而……不，那也不影响键。

诚实的答案是：**`id` 被显式归一化了，所以它不该影响缓存。若你观察到影响，说明你用的是自定义模型且 `is_lc_serializable()` 返回 `True`——此时键里带的是 `self._serialized` 全量序列化，而某些实现会把消息也卷进去。** 先做一件事验证：把 `_get_llm_string(stop=None)` 在两个"只有 id 不同"的调用里各打一次，看是否相同。

> **延伸**：这题的真正价值是教你一个**排查方法**：缓存不命中时不要猜，直接把 `prompt`（消息 JSON）和 `llm_string`（参数串）分别打出来对比。两者之一变了，就是它。

**4. `strict=True` 为什么要把所有 properties 塞进 required？**

因为 OpenAI 的 strict 模式要求生成的 JSON 必须**严格符合**给定 schema，而它实现这个保证的方式是：不允许"缺字段"。若一个字段可选，那它缺席时模型就"没按 schema 填"，strict 的承诺就破了。

LangChain 的做法是顺着这个约束走：**把所有 properties 变成 required**（`schema["required"] = list(properties.keys())`），并递归给每层加 `additionalProperties: false`。

> **延伸**：这解释了一个常见困惑——"我开了 strict，模型开始瞎编可选参数"。那不是模型变笨了，是你告诉它"这个字段必须有值"。真要可选，就别开 strict，改用 `Optional` + 应用层校验（第 7 章讲结构化输出时会给出替代方案）。

**5. `astream_events` 收不到 token，最可能漏了哪步？**

三选一，按概率排序：

1. **`_stream` 没实现**（只写了 `_generate`）。基类 `_stream` 直接 `raise NotImplementedError`，事件流就退化成"一次性 end 事件"。
2. **`_generate`/`_stream` 签名里没有 `run_manager`**。基类会检查 `_signature_parameters(self._generate).get("run_manager")`，没有就不传回调，`on_llm_new_token` 自然不触发。
3. **`disable_streaming` 被设成了 `True` 或 `"tool_calling"` 而这次带工具**。

> **延伸**：这三条是一个通用的调试次序——**先问"能力实现了没"，再问"回调接上了没"，最后问"开关关了没"**。这套次序在 LangGraph 的 `stream_mode` 上会原样复用（第 10 章）。

---

## ⑦ 承上启下

现在你知道了：`invoke` 把消息送到供应商，把 `AIMessage` 拿回来，中间不保存任何东西。**那多轮对话的记忆存在哪？**

`create_agent(checkpointer=...)` 里的那个 checkpointer，和 `AgentState` 里的 `messages` 字段，是两套不同的机制：前者存"图执行到哪了"，后者定义"状态长什么样、新消息怎么并进去"。

下一章拆 `State` 与 `Reducer`：你会看到 `add_messages` 为什么不是普通 list 拼接，以及**为什么两个并行节点同时往 `messages` 写不会互相覆盖**。

---

## 附录 · 官方写法对照（models 页）

> 官方锚点：<https://docs.langchain.com/oss/python/langchain/models>
> 代码：`langchain_learn/03_official_models.py`（A–E 段零 token，`--real` 才联网）

正文是"顺着 `init_chat_model` 的源码往下读"，这一节换成官方视角：**用户该知道的那五件事**。两边对照，你会发现正文里那些自研打印，恰好补上了官方页没说透的部分。

### A3.1 取模型的两条路，其实是同一个类

```text
  init_chat_model(...)  -> langchain_openai.chat_models.base.ChatOpenAI
  ChatOpenAI(...)       -> langchain_openai.chat_models.base.ChatOpenAI
```

`init_chat_model` 就是个**分发表**：`provider` 决定 import 哪个包、实例化哪个类。什么时候必须用第二条路？**要传 provider 独有参数时**（比如 OpenAI 的 `use_responses_api`）。

装错包的对应关系（这是新手最常见的 30 分钟）：

| provider | 要装的包 | extra |
|---|---|---|
| `openai` | `langchain-openai` | `langchain[openai]` |
| `anthropic` | `langchain-anthropic` | `langchain[anthropic]` |
| `google_genai` | `langchain-google-genai` | `langchain[google-genai]` |
| `bedrock_converse` | `langchain-aws` | `langchain[aws]` |

### A3.2 参数全部落在构造期

实测把本机实例的属性读出来：

```text
    model_name         = 'qwen3:latest'
    temperature        = 0.0
    max_tokens         = None
    request_timeout    = 60.0
    max_retries        = None
```

官方表里六个参数（`model` / `api_key` / `temperature` / `max_tokens` / `timeout` / `max_retries`）**都是构造期就固定的**，不是每次 invoke 再传。这条认知很重要：**想"这次用 0、下次用 1"改 temperature，得换实例，不是在消息里加参数。**

### A3.3 重试边界：`is_retryable` 是写死的，不用你猜

官方页说"自动重试网络错误、429、5xx，不重试 401/404"。本机实测把这张表打全了：

| 异常 | `is_retryable` |
|---|---|
| `ModelConnectionError` | ✅ True |
| `ModelRateLimitError`（429） | ✅ True |
| `ModelTimeoutError` | ✅ True |
| `ModelAPIError`（5xx 父类） | ✅ True |
| `ModelAuthenticationError`（401） | ❌ False |
| `ModelNotFoundError`（404） | ❌ False |
| `ModelInvalidRequestError` | ❌ False |
| `ModelPermissionDeniedError` | ❌ False |
| `ContextOverflowError` | ❌ False |

**最后一行是个提醒**：上下文超了不会因为重试而变短。它的处理方式是摘要或截断（第 12 章），不是 retry。

### A3.4 `bind_tools` 之后请求里多了什么

```text
  kwargs['tools'] 条数：1
    name        : get_weather
    description : Get the weather at a location.
    parameters  : {'properties': {'location': {'type': 'string'}},
                   'required': ['location'], 'type': 'object'}
```

`bind_tools` **只把 schema 塞进请求，不执行任何东西**。所以官方紧接着给了一段"手动工具循环"：

```python
messages = [{"role": "user", "content": "..."}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    result = tools_by_name[tool_call["name"]].invoke(tool_call)
    messages.append(result)
final = model_with_tools.invoke(messages)
```

**这 6 行就是 `create_agent` 内部 `tools` 节点的全部内容。** 正文第 1 章那张图里的循环，剥掉 LangGraph 的调度外壳，剩下的就是这个。

### A3.5 结构化输出：3 种 schema × 3 种 method，共用同一个出口

```text
  ① Pydantic
     with_structured_output(Movie) -> RunnableSequence
     必填字段：['title', 'year', 'director', 'rating']

  ② TypedDict / ③ JSON Schema
     method=function_calling   构造成功
     method=json_schema        构造成功
     method=json_mode          构造成功
```

`method` 三选一的真实含义：

| method | 怎么实现 | 代价 |
|---|---|---|
| `function_calling` | 借工具的通道传 schema，兼容性最好 | 多一份 tool schema，token 更贵 |
| `json_schema` | 用 provider 原生结构化输出 | 不是所有 provider 都支持 |
| `json_mode` | 只保证输出是合法 JSON，schema 要自己写进 prompt | 约束最弱 |

**和 `create_agent(response_format=...)` 的区别**（这条最容易被混）：

- `with_structured_output` 是**模型级**约束：一次调用就定型。
- `response_format` 是 **agent 级**约束：多轮工具调用跑完之后再定型，结果在 `result["structured_response"]`。

### A3.6 真跑一次

```text
  invoke  -> AIMessage / 温度是物体内部粒子运动剧烈程度的度量，用来表示物体的冷热程度。
  usage   -> {'input_tokens': 16, 'output_tokens': 414, 'total_tokens': 430, ...}

  stream  -> 209 个 AIMessageChunk
  拼接后  -> 一、二、三！（*鼓掌*）需要我接着做点什么吗？
```

注意 `invoke` 用时远大于 `stream` 首块出现的时间——**这就是所有前端都改成流式的原因**，跟模型快慢无关，跟"用户先看到字"有关。

### 认知修正

> 我原本以为 `max_retries` 只是"多试几次"的粗暴开关；实际上框架**已经按异常类型分好了**可重试与不可重试两类，`is_retryable` 是写死的属性。你真正该调的不是 retry 次数，而是"遇到不可重试的异常时降级到哪个模型"（第 19 章 `ModelFallbackMiddleware`）。
