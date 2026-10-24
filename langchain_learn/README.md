# langchain_learn · 教材代码区

配合 `docs/langchain-family/langchain_learn/` 下的教材使用。

## 运行约定

**在项目根目录**用脚本方式运行（不要用 `-m`）：

```powershell
python langchain_learn/00_env_check.py
python langchain_learn/01_graph_anatomy.py
python langchain_learn/02_message_anatomy.py
python langchain_learn/03_model_layer.py
python langchain_learn/03_callbacks_demo.py
python langchain_learn/03_callback_params_trace.py
python langchain_learn/04_tool_layer.py
python langchain_learn/05_state_reducer.py
python langchain_learn/06_checkpointer.py
python langchain_learn/06_lite_checkpoint.py
python langchain_learn/07_structured_output.py
python langchain_learn/08_official_middleware_hooks.py
```

原因：脚本方式下 `langchain_learn/` 会进 `sys.path[0]`，所以 `from _shared import ...` 可用；用 `-m` 反而会因模块名以数字开头而出问题。

### 官方写法对照（ch00–ch08 的附录代码）

这 9 个文件对应每章末尾的「附录 · 官方写法对照」，**默认零 token**（只有编译期 + 手工 invoke），加 `--real` 才真发请求：

```powershell
python langchain_learn/00_official_agent_equation.py
python langchain_learn/01_official_create_agent.py
python langchain_learn/02_official_messages.py
python langchain_learn/03_official_models.py
python langchain_learn/04_official_tools.py
python langchain_learn/05_official_short_term_memory.py
python langchain_learn/06_official_checkpointer.py
python langchain_learn/07_official_structured_output.py
python langchain_learn/08_official_middleware_hooks.py

# 想真跑模型那一段，就加 --real（qwen3 一次 20-40 秒，别在赶时间时开）
python langchain_learn/00_official_agent_equation.py --real
```

`_official_common.py` 是这 9 个文件共用的三件小工具：分节横幅、`real_enabled()`、`short()`。

## 文件

| 文件 | 对应章节 | 作用 |
|---|---|---|
| `_shared.py` | 全书 | 读 `.env` 造模型 `get_model()`、打印消息轨迹 `dump_messages()` |
| `00_env_check.py` | 第 0 章 | 环境自检：版本、入口、连通性、工具调用能力 |
| `01_graph_anatomy.py` | 第 1 章 | 打印 `create_agent` 编出的图；对比"有工具/无工具" |
| `02_message_anatomy.py` | 第 2 章 | 消息字段解剖；手工续写 4 条消息；流式 `tool_call_chunks` |
| `03_model_layer.py` | 第 3 章 | 字符串解析 → 实例化分流；工具转 JSON；自写假模型跑通 6 层链路；缓存键对比（A–E 离线，F 真请求） |
| `03_callbacks_demo.py` | 第 3 章 · 2.6 补充 | `CallbackManager.configure` 合并规则、批量为什么返回 list、run_id 截断陷阱、四种注入途径（A/A2/B 零 token） |
| `03_callback_params_trace.py` | 第 3 章 · 2.6 补充 | temperature 三处写法的优先级、偷看 wire payload、agent 里的 10 条事件流（B 零 token，C–E 联网） |
| `04_tool_layer.py` | 第 4 章 | `@tool` 四种写法、三份 schema、注入参数、三种失败、content/artifact |
| `05_state_reducer.py` | 第 5 章 | `AgentState` 注解、`add_messages` 五种情况、一 state 变三份 schema、有/无 reducer 对比、假模型抓"失忆"铁证、自定义 reducer（A–F 零 token，`--real` 联网） |
| `06_checkpointer.py` | 第 6 章 | 拆 `InMemorySaver` 三个字典（`storage`/`writes`/`blobs`）；`channel_values` 为什么是空的；uuid6 与 `max()` 取最新；`get_state_history` 时间轴；`update_state` 分叉；**自定义"会喊的 reducer"证明恢复时不重放**（A–G 零 token、纯图无 LLM，`--real` 才跑 `create_agent`） |
| `06_lite_checkpoint.py` | 第 6 章 · 极简版 | 114 行、只用标准库重写一遍存档：`CheckpointStore` 的 `save/load/head/history`、`(thread, 变量, 版本)` 二级键、值没变就复用旧版本。跑一次能看到 `8 帧 / 6 份内容`、以及改档长出新分支（标识符全英文，注释中文） |
| `07_structured_output.py` | 第 7 章 | `_SchemaSpec` 四种 schema；`ToolStrategy` 的 Union 展开；`OutputToolBinding` 假工具；`ProviderStrategy.to_model_kwargs`（`strict` 只在真值时出现）；`AutoStrategy` 空标记；`handle_errors` 三种反应（A–E 零 token，`--real` 跑 F/G） |
| `_official_common.py` | 附录 | 7 个官方对照示例共用：分节横幅、`--real` 判定、长对象截断 |
| `00_official_agent_equation.py` | 第 0 章附录 | overview 页命题 `Agent = Model + Harness`；`provider:model` 怎么拆（`ollama:qwen3:latest` 的冒号坑）；15 个参数分 harness / 运行时两堆；三框架判据 + 本机装了什么 |
| `01_official_create_agent.py` | 第 1 章附录 | 五层配置逐层建图对比（节点 vs 边）；`response_format` 改边不改节点；retry / PII / HITL 三种 middleware 对图的影响（PII 会加 2 个节点） |
| `02_official_messages.py` | 第 2 章附录 | 三种输入归一 + 裸字符串陷阱；四种消息手搓；`content_blocks` 抹平 anthropic/openai 原生格式（块数会变）；`artifact`；`dumpd`/`load` 的两条警告；真实 chunk 拼接 |
| `03_official_models.py` | 第 3 章附录 | `init_chat_model` 与 `ChatOpenAI` 的关系；六个构造期参数；`is_retryable` 全表实测；`bind_tools` 塞进去的 schema + 官方手动工具循环；3 种 schema × 3 种 method |
| `04_official_tools.py` | 第 4 章附录 | `@tool` 三件套；`args_schema` 的 `Literal`→enum；`runtime` 不进 schema；`ToolRuntime` 八个部件；四种返回值实测；`return_direct` 三条边界；`@wrap_tool_call` 兜错 |
| `05_official_short_term_memory.py` | 第 5 章附录 | checkpointer × thread_id 三态对照（`[1,3]` vs `[1,1]`）；`state_schema` 继承 vs 裸 TypedDict 的 output schema 差；`@before_model` 裁剪的稳态与"半对"副作用；`RemoveMessage` 删一条/全清；`SummarizationMiddleware` 触发与改写形态；工具里 `runtime.state` 读 / `Command` 写；`@dynamic_prompt` 三概念划界（A–G 零 token，`--real` 联网） |
| `06_official_checkpointer.py` | 第 6 章附录 | checkpointers 页六层：threads 主键；带工具的 invoke = 5 张快照；子图 `checkpoint_ns`；`StateSnapshot` 七字段（文档与 1.2.16 的两处差异）；history 四种检索姿势（生成器陷阱）；replay 的三条边界；`update_state` 与 `as_node`；`durability` 三档落盘数；blobs 去重 + 四种后端与官方四个坑（A–I 零 token，`--real` 联网） |
| `07_official_structured_output.py` | 第 7 章附录 | structured-output 页九件事：`response_format` 四张脸；provider 判据七行（名字白名单、`profile=False` 挡不住、gemini 2.x 带工具例外）；两条通道的 `bind_tools` 参数对照；四种 schema 返回物（dataclass 实测是**实例**）；`tool_message_content` 默认文案；`handle_errors` 六种归一化表（`ValueError` 接不住校验错）；两类错误现场；四个坑（裸 dict 能跑 / `None` 不是缺 key / `tool_choice="any"` 死循环 / 显式 Provider 不回退）（A–H 零 token，I 段 `--real`） |
| `08_official_middleware_hooks.py` | 第 8 章附录 | middleware 页六段：`AgentMiddleware` 12 个钩子（6 个语义 × 同步/异步，默认实现是 `raise`）；覆盖检测读 `__class__` 而非 `hasattr`（在 `__init__` 里塞实例方法无效）；`wrap_model_call` 是嵌套括号、first = outermost；`wrap_*` 不进图（不调 handler 就等于不执行）；三个装饰器产出的是中间件对象而非函数；`--real` 看 `before_model`/`after_model` 实际命中顺序（A–E 零 token，F 段 `--real`） |

## 这些是参考实现

教材要求**手敲**。这里的文件是给你对照用的：敲完再 diff 一次，差异处往往就是你理解错的地方。

## `recap/` —— 卡住时回来看

`recap/` 不是新章节，是**给 ch00–ch05 重新长骨架**的经典例子：
`r1_invoke` / `r2_tool_loop` / `r3_tool_schema` / `r4_state` / `r6_runtime` / `r5_quiz`。
全部离线零 token，只看输出就能"啊"一声。详见 `recap/README.md`。

感觉"学了但不懂"时，不要往下推 ch07，先跑一遍 `recap/`。


## 阶段一（LangChain）之后的目录归属

- 本目录（`langchain_learn/`）：LangChain 高层 API 与 middleware
- `langgraph_learn/`（已存在）：LangGraph 图语义、checkpoint、HITL
- `deepagents_learn/`（待建）：DeepAgents harness 与三版对照项目
