# langchain_learn · 教材代码区

配合 `docs/langchain-family/langchain_learn/` 下的教材使用。

## 运行约定

**在项目根目录**用脚本方式运行（不要用 `-m`）：

```powershell
python langchain_learn/00_env_check.py
python langchain_learn/01_graph_anatomy.py
```

原因：脚本方式下 `langchain_learn/` 会进 `sys.path[0]`，所以 `from _shared import ...` 可用；用 `-m` 反而会因模块名以数字开头而出问题。

## 文件

| 文件 | 对应章节 | 作用 |
|---|---|---|
| `_shared.py` | 全书 | 读 `.env` 造模型 `get_model()`、打印消息轨迹 `dump_messages()` |
| `00_env_check.py` | 第 0 章 | 环境自检：版本、入口、连通性、工具调用能力 |
| `01_graph_anatomy.py` | 第 1 章 | 打印 `create_agent` 编出的图；对比"有工具/无工具" |

## 这些是参考实现

教材要求**手敲**。这里的文件是给你对照用的：敲完再 diff 一次，差异处往往就是你理解错的地方。

## 阶段一（LangChain）之后的目录归属

- 本目录（`langchain_learn/`）：LangChain 高层 API 与 middleware
- `langgraph_learn/`（已存在）：LangGraph 图语义、checkpoint、HITL
- `deepagents_learn/`（待建）：DeepAgents harness 与三版对照项目
