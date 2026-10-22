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
```

原因：脚本方式下 `langchain_learn/` 会进 `sys.path[0]`，所以 `from _shared import ...` 可用；用 `-m` 反而会因模块名以数字开头而出问题。

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

## 这些是参考实现

教材要求**手敲**。这里的文件是给你对照用的：敲完再 diff 一次，差异处往往就是你理解错的地方。

## 阶段一（LangChain）之后的目录归属

- 本目录（`langchain_learn/`）：LangChain 高层 API 与 middleware
- `langgraph_learn/`（已存在）：LangGraph 图语义、checkpoint、HITL
- `deepagents_learn/`（待建）：DeepAgents harness 与三版对照项目
