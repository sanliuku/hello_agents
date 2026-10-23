# 第 6 章 · 会话持久化：checkpoint 里到底存了什么

> **官方锚点**：`/oss/python/langchain/short-term-memory`
> **源码落点**：`langgraph/checkpoint/base/__init__.py`、`langgraph/checkpoint/memory/__init__.py`、`langgraph/channels/{base,binop}.py`、`langgraph/pregel/main.py`
> **前置**：ch05（State 与 Reducer）。ch05 末尾留下的那个问题，本章正面回答。
> **版本**：本地 `langgraph 1.2.16` / `langchain 1.4.6` / `langchain-core 1.6.12`。本章源码行号全部来自你 `.venv` 里**已装的那份**，不是 GitHub master——你可以照着行号直接翻。

---

## ① 本章你会得到什么

**一句话**：你能亲手打印出 `InMemorySaver` 里那三个字典，指着它们说清楚"哪一行是会话、哪一行是消息、哪一行是断点"，并且能回答 ch05 留下的那个要命问题——

> **reducer 是合并语义。从快照恢复时，它会被再执行一次吗？**

答案是 **不会**，而且原因不在 reducer 里，在 `from_checkpoint` 里。这一章就把这行代码挖出来。

验收标准（你自己能对号入座）：

1. 能说出 `thread_id` / `checkpoint_ns` / `checkpoint_id` 三层主键各自是什么、谁生成、怎么排序。
2. 能解释为什么 `Checkpoint` 这个 dict 里 **`channel_values` 在落盘时是空的**，值去哪了。
3. 能解释"第二次 `invoke` 只传一条新消息就够"这件事，在源码里是哪一行实现的。
4. 能说出 `pending_writes` 是干什么的，以及什么情况下它非空。
5. 能解释为什么 checkpoint id 用 `uuid6` 而不是 `uuid4`——这一条是整章最漂亮的设计。

---

## ② 源码解剖

### 2.1 `thread_id` 不是"会话 ID"，是三层主键的第一层

先看官方在基类 docstring 里怎么写的。`langgraph/checkpoint/base/__init__.py`（`BaseCheckpointSaver` 的 docstring）：

```python
The `thread_id` is the primary key used to store and retrieve checkpoints. Without
it, the checkpointer cannot save state, resume from interrupts, or enable
time-travel debugging.
```

"primary key" —— 主键。但**它只是第一层**。真正的主键是三层：

```
thread_id          会话         你给（"user-42-chat-1"）
  └─ checkpoint_ns  命名空间     子图/子 agent 用，根图是 ""（空字符串）
       └─ checkpoint_id  快照    langgraph 自己生成（uuid6）
```

`InMemorySaver` 把这个结构直接写在类型注解里（`memory/__init__.py:69-73`）：

```python
# thread ID ->  checkpoint NS -> checkpoint ID -> checkpoint mapping
storage: defaultdict[
    str,
    dict[str, dict[str, tuple[tuple[str, bytes], tuple[str, bytes], str | None]]],
]
```

`storage[thread_id][checkpoint_ns][checkpoint_id]` → 三元组 `(checkpoint_bytes, metadata_bytes, parent_checkpoint_id)`。

三个要点，一次说清：

- **`thread_id` 由你决定语义。** 官方 docstring 给了两种用法：单次任务用 `uuid4()` 每次不同；对话记忆复用同一个 `thread_id`。所以"会话"这个词是**你的约定**，框架只当它是个字符串 key。
- **`checkpoint_ns` 是给子图用的。** 根图永远是 `""`。你调用子图/子 agent 时，框架会拼出 `"节点名<task_id>"` 这样的 ns（`pregel/main.py:1276-1278` 里 `task_ns = f"{task.name}{NS_END}{task.id}"`）。**这一层存在的原因**：一个子图可能被同一个父图调用多次（比如并行发三个子 agent），每次调用都需要**各自独立的**检查点序列，否则它们会互相覆盖。
- **`checkpoint_id` 不是计数器，是 uuid6。** 见 2.4，这是本章的核心设计。

> 这里有个坑要提前说：`checkpoint_ns` 不是"文件夹"。它是**命名空间字符串**，Postgres 里是普通列，`InMemorySaver` 里是嵌套 dict 的第二层 key。你在 `config` 里手动传 `checkpoint_ns` 只在"我要读某个子图的检查点"时有意义；日常对话你只传 `thread_id`。

---

### 2.2 `Checkpoint` 就 7 个字段，而且值不在里面

`base/__init__.py` 里 `Checkpoint` 是个 `TypedDict`，一共 7 个字段：

```python
class Checkpoint(TypedDict):
    v: int                    # 检查点格式版本，当前 1（但 LATEST_VERSION = 2，见下）
    id: str                   # uuid6，唯一且单调递增 → 可以直接排序
    ts: str                   # ISO 8601 时间戳
    channel_values: dict[str, Any]   # 各 channel 的「值」
    channel_versions: ChannelVersions # 各 channel 的「版本号」
    versions_seen: dict[str, ChannelVersions]  # 节点 → {channel → 它见过的版本}
    updated_channels: list[str] | None
```

先注意一个**反直觉点**：`v` 注释写"当前 1"，但文件末尾 `LATEST_VERSION = 2`，`empty_checkpoint()` 用的是 `LATEST_VERSION`。所以**实际写进去的是 2**，注释是历史遗留。这种"注释和常量打架"的地方，永远信常量。

现在看最关键的字段：`channel_values` 和 `channel_versions`。

`channel_values` 的注释是：

```python
channel_values: dict[str, Any]
"""The values of the channels at the time of the checkpoint.
Mapping from channel name to deserialized channel snapshot value."""
```

"deserialized" —— 这个词说明它是**读出来的样子**，不是**存进去的样子**。真正的存储结构见 2.3。

`versions_seen` 是最容易被忽略但最重要的一个：

```python
versions_seen: dict[str, ChannelVersions]
"""Map from node ID to map from channel name to version seen.
This keeps track of the versions of the channels that each node has seen.
Used to determine which nodes to execute next."""
```

最后一句是重点：**它决定下一步跑谁**。Pregel 的超步调度逻辑是"如果某个节点订阅的 channel 出现了它没见过的版本，就该跑它"。所以 `versions_seen` 是一本**"每个节点已经消费到哪个版本"的账本**。断点续跑能正确，靠的就是这本账——不是靠重放历史。

---

### 2.3 `put()`：值被 pop 出去，按版本存进 blobs

这是本章第一个"啊"的地方。`memory/__init__.py:451-467`：

```python
c = checkpoint.copy()
thread_id = config["configurable"]["thread_id"]
checkpoint_ns = config["configurable"]["checkpoint_ns"]
values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]   ← 值被拿走了
for k, v in new_versions.items():
    self.blobs[(thread_id, checkpoint_ns, k, v)] = (
        self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
    )
self.storage[thread_id][checkpoint_ns].update(
    {
        checkpoint["id"]: (
            self.serde.dumps_typed(c),                                    # 不含值的 checkpoint
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
            config["configurable"].get("checkpoint_id"),                  # parent
        )
    }
)
```

三件事，逐条拆：

**第一，`c.pop("channel_values")`。** 存进 `storage` 的那个 checkpoint **不包含任何 channel 的值**。它只保留 `channel_versions`（名字→版本号的指针）。值全部进 `blobs`。

**第二，blobs 的 key 是 `(thread, ns, channel, version)` —— 四元组，带版本号。** 这不是随便设计的，它带来一个很实际的后果：

> 如果某个 channel 这一步**没变**（版本号没涨），那它就不在 `new_versions` 里，也就**不会写 blob**。上一步的 blob 还在原位，这个 checkpoint 通过 `channel_versions` 里的旧版本号**指回**那个旧 blob。

也就是说 —— **相同内容在磁盘上只存一份**。10 轮对话里，`messages` 每一轮都变（新 blob），但别的字段（比如一个从不改的 `user_id`）只有 1 个 blob。这就是"内容寻址 + 版本指针"。

**第三，`else ("empty", b"")`。** 版本涨了但值不在 `values` 里时，写一个 `"empty"` 哨兵。读的时候 `_load_blobs` 会跳过它（`memory/__init__.py:138-139`）：

```python
if vv[0] == "empty":
    continue
```

后果：这个 channel **不会出现在 `channel_values` 里**，恢复时就是"空 channel"，`get()` 会抛 `EmptyChannelError`。所以**"版本涨了但没有值"是有语义的**，代表"这个 channel 在这一步被清空了"，而不是 bug。

读的时候怎么拼回来？`get_tuple`（`memory/__init__.py:259-267`）：

```python
checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint)
return CheckpointTuple(
    config=config,
    checkpoint={
        **checkpoint_,
        "channel_values": self._load_blobs(
            thread_id, checkpoint_ns, checkpoint_["channel_versions"]
        ),
    },
    ...
```

`_load_blobs` 拿着 `channel_versions`（名字→版本）去 blobs 里逐个取。所以 `channel_values` 是**读的时候现拼的**，从来没整体存在过。

> **一句话记住**：`storage` 存"哪个版本"，`blobs` 存"那个版本长什么样"，`channel_values` 是读的时候拼出来的视图。

---

### 2.4 `get_tuple()`：靠 `max()` 取最新 —— 这就是 uuid6 的用处

你 `invoke` 时只给了 `thread_id`，没给 `checkpoint_id`。框架怎么知道"最新的是哪个"？

`memory/__init__.py:284-287`：

```python
else:
    if checkpoints := self.storage[thread_id][checkpoint_ns]:
        checkpoint_id = max(checkpoints.keys())      # ← 就这一行
```

`max()` 对字符串取字典序最大。**这能等价于"时间最新"的前提是：id 的字符串序 == 时间序。** `uuid4` 是随机的，做不到；`uuid6` 是**按时间排序的 UUID 变体**（RFC 9562），前 48 位是时间戳，所以字符串序就是时间序。

这就是为什么 `empty_checkpoint()` 里写的是：

```python
id=str(uuid6(clock_seq=-2)),
```

`clock_seq=-2` 是给"输入"检查点留的序号段，让 input checkpoint 排在同一步的 loop checkpoint 之前。`create_checkpoint` 里用 `uuid6(clock_seq=step)`，把 step 编进低位。

**为什么这个设计值得单独讲**：它让"取最新"退化成一次字典 key 的 `max()`，不需要额外索引、不需要 `ORDER BY ts`、不需要保证时钟单调。在 Postgres 里就是 `ORDER BY checkpoint_id DESC LIMIT 1`，走主键索引即可。**用一个 id 格式，换掉了一整套排序逻辑。**

代价也很真实：**你不能自己指定 checkpoint_id 的语义**（比如"用时间戳当 id"），因为框架依赖"字符串序=插入序"这个不变量。你要自定义 saver，`get_next_version` 可以改，但 `checkpoint.id` 的单调性不能破坏。

---

### 2.5 "第二次 invoke 只传一条新消息就够" —— 是哪一行实现的

ch05 你看到的现象是：

```python
app.invoke({"messages": [HumanMessage("第1句")]}, cfg)   # 返回 2 条
app.invoke({"messages": [HumanMessage("第2句")]}, cfg)   # 返回 4 条（不是 3 条）
```

第二次你**只传了一条**，但结果是 4 条。为什么？

因为**你传的 `{"messages": [...]}` 不是"新状态"，是"这一步的写入"**。它是 Pregel 的一个 superstep 的输入，会被 reducer 合并进已有状态，而不是替换。

链路是这样的（`pregel/main.py:1657-1696`，`bulk_update_state` 里的 `perform_superstep`，`stream()` 走的是同一套）：

```python
saved = checkpointer.get_tuple(config)                       # L1661 读上一个检查点
checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()   # L1673-1675
step = saved.metadata.get("step", -1) if saved else -1        # L1679
channels, managed = channels_from_checkpoint(                 # L1691 从检查点重建 channels
    self.channels, checkpoint, saver=checkpointer,
    config=saved.config if saved is not None else None,
)
```

`channels_from_checkpoint` 会为每个 channel 调 `channel.from_checkpoint(checkpoint["channel_values"].get(key, MISSING))`，把**上一步的值装回 channel**。然后你传的新消息作为一个 write，被 `apply_writes` 送进 `channel.update([...])`，`BinaryOperatorAggregate.update` 调 `operator(old, new)` —— 也就是 `add_messages(旧的4条, 新1条)`。

所以是"**旧值已在 channel 里，你的输入是增量**"。

> 反过来：如果你**想**用新消息替换全部历史，`add_messages` 支持 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 先清空。这是 ch05 讲过的，本章不重复。

---

### 2.6 核心问题：reducer 会重放吗？—— 看 `from_checkpoint`

ch05 ⑦ 留的问题，答案在这两行里。`langgraph/channels/binop.py:117-125`：

```python
def from_checkpoint(self, checkpoint: Value) -> Self:
    empty = self.__class__(self.typ, self.operator)
    empty.key = self.key
    # A thread written while this field was a DeltaChannel stores snapshots.
    if isinstance(checkpoint, _DeltaSnapshot):
        checkpoint = checkpoint.value
    if checkpoint is not MISSING:
        empty.value = checkpoint          # ← 直接赋值
    return empty
```

对比同文件里的 `update()`（L127-148）：

```python
def update(self, values: Sequence[Value]) -> bool:
    if not values:
        return False
    if self.value is MISSING:
        self.value = values[0]
        values = values[1:]
    ...
    for value in values:
        ...
        if not seen_overwrite:
            self.value = self.operator(self.value, value)     # ← 只有这里调 operator
    return True
```

**`operator` 只在 `update()` 里被调用，`from_checkpoint()` 里一次都没有。** 恢复时是把"已经合并好的最终值"整个塞回 `empty.value`。

所以：

> **reducer 不会重放。** checkpoint 存的是**合并后的结果**，不是"待合并的增量列表"。恢复 = 赋值，不是重算。

这个结论有三个直接推论，都很实用：

1. **reducer 里放副作用是安全的**（从"不会重复执行"这个角度）。但这不等于"应该放"——见下面第 2 条。
2. **reducer 必须是幂等/可重入的吗？** 在恢复路径上不需要。但在**同一步内**，`update()` 收到的是一个 `Sequence`，多个并行节点写同一 channel 时会被依次 `operator` 合并。所以 reducer 需要满足的是**结合律**（`(a+b)+c == a+(b+c)`），不是幂等。
3. **改 reducer 的实现不会追溯修改历史状态**。你昨天用 `operator.add`，今天改成 `lambda a,b: a+b`（等价），历史 checkpoint 不受影响——因为它们存的是结果。但如果你改了**语义**（比如从"追加"改成"去重追加"），旧 checkpoint 里已经是"追加"的结果，不会自动重算。

> 补一句边界：`from_checkpoint` 里那个 `_DeltaSnapshot` 分支是给 **DeltaChannel**（beta）用的。普通 `add_messages` 不会走到。看到 `isinstance(checkpoint, _DeltaSnapshot)` 不要慌，那是增量压缩通道的哨兵，本章不展开。

---

### 2.7 `pending_writes` 与 `WRITES_IDX_MAP`：断点续跑的账本

`CheckpointTuple` 有个字段 `pending_writes: list[PendingWrite] | None`，其中 `PendingWrite = tuple[str, str, Any]`（task_id, channel, value）。

它的含义：**已经写进 saver、但还没被"应用成新 checkpoint"的写入**。

什么场景会非空？**一个 superstep 跑到一半崩了**。比如三个并行节点，两个跑完了写了 writes，第三个挂了。这时 writes 已经 `put_writes` 落盘，但 `put`（生成新 checkpoint）还没发生。下次同 `thread_id` 恢复，`get_tuple` 把 `pending_writes` 一起带回来，框架**先应用这些 writes 再继续**——不会重跑已完成的两个节点。

这就是"断点续跑"和"从头重跑"的区别。也是为什么 LangGraph 敢说"durable execution"。

`WRITES_IDX_MAP`（`base/__init__.py`）给特殊写入分配**负数下标**：

```python
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}
```

原因写在注释里："to avoid those writes from conflicting with regular writes"。普通写入用 `enumerate` 的 0,1,2...，特殊写入占负数段，互不冲突。

而 `put_writes` 里有个**幂等保护**（`memory/__init__.py:502-505`）：

```python
for idx, (c, v) in enumerate(writes):
    inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
    if inner_key[1] >= 0 and outer_writes_ and inner_key in outer_writes_:
        continue          # 普通写入已存在 → 跳过（不覆盖）
    self.writes[outer_key][inner_key] = (task_id, c, self.serde.dumps_typed(v), task_path)
```

注意 `>= 0` 这个条件：**普通写入**重复了才跳过，**特殊写入**（负数）允许覆盖。为什么？因为 `INTERRUPT` / `RESUME` 这类状态**本来就该被最新值替换**——一次运行被中断多次是正常的，最后一次才算数。

---

### 2.8 `metadata.source`：四种来源，决定了你调试时看什么

```python
source: Literal["input", "loop", "update", "fork"]
```

- `"input"`：由 `invoke/stream/batch` 的输入产生。`step == -1`。
- `"loop"`：Pregel 循环内部产生（每个 superstep 一个）。第一个是 `step == 0`，之后递增。
- `"update"`：你手动 `update_state()` 产生的。
- `"fork"`：从别的 checkpoint 复制出来的（时间旅行分叉）。

配合 `step` 的取值规律，你能在 `get_state_history()` 的输出里一眼看出"这次运行到底走了几步"：

```
step=-1  source=input    ← 你把输入交进去
step=0   source=loop     ← 第一个 superstep（比如 model 节点）
step=1   source=loop     ← 第二个 superstep（比如 tools 节点）
step=2   source=loop     ← 第三个 superstep（model 再答一次）
```

所以一个"model → tools → model"的完整工具调用循环，会产生 **1 个 input + 3 个 loop = 4 个 checkpoint**。这个数字在你做 HITL 和流式的时候天天要用。

---

## ③ 你敲这段

**目标**：不碰 LLM，纯图 + 纯 `InMemorySaver`，把三个字典拆开看。

配套文件：`langchain_learn/06_checkpointer.py`。下面只贴核心 A–D 段（完整版含 E–G）。

```python
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class S(TypedDict):
    messages: Annotated[list, add_messages]


def bot(state: S) -> dict:
    n = len(state["messages"])
    return {"messages": [AIMessage(content=f"我看到 {n} 条，回第 {n} 条")]}


g = StateGraph(S)
g.add_node("bot", bot)
g.add_edge(START, "bot")
g.add_edge("bot", END)

saver = InMemorySaver()
app = g.compile(checkpointer=saver)

cfg = {"configurable": {"thread_id": "t1"}}

r1 = app.invoke({"messages": [HumanMessage("第1句")]}, cfg)
print("第一次:", len(r1["messages"]))          # 预期 2

r2 = app.invoke({"messages": [HumanMessage("第2句")]}, cfg)
print("第二次:", len(r2["messages"]))          # 预期 4（不是 3！）
print("最早那句还在吗:", r2["messages"][0].content)   # 预期 "第1句"
```

**逐行说明"为什么这么写"**：

| 行 | 为什么 |
|---|---|
| `messages: Annotated[list, add_messages]` | 有 reducer 才有"累积"。去掉这个 `Annotated`，第二次 invoke 会变成 2 条（覆盖），跟没 checkpointer 一样 |
| `bot` 返回 `{"messages": [AIMessage(...)]}` | 只返回**增量**，不返回全量。Pregel 会调 `add_messages(旧, 新)` |
| `len(state["messages"])` | 这是**探针**：它读的是"恢复后 channel 里真实有几条"，能直接证明恢复成功 |
| `InMemorySaver()` 在 `compile` **之前**建 | `compile(checkpointer=...)` 时要把 saver 挂到图上。反过来不行 |
| `cfg` 每次 `invoke` 都传同一个 `thread_id` | 这一条就是"记忆"的全部秘密。换掉它 = 新会话 |
| `r2["messages"][0].content` | **铁证**：不是"条数对了"，是"最早那条真的还在" |

**然后 D 段——直接把 saver 拆开**（这一段是本章的灵魂，教材要求你敲完必须打印出来看一眼）：

```python
print("\n=== saver.storage 的层级 ===")
print("第一层 thread_id :", list(saver.storage.keys()))            # ['t1']
print("第二层 checkpoint_ns:", list(saver.storage["t1"].keys()))   # ['']  ← 根图是空串
cps = saver.storage["t1"][""]
print("第三层 checkpoint_id 个数:", len(cps))

latest = max(cps.keys())
ck_bytes, md_bytes, parent = cps[latest]
print("最新 checkpoint_id:", latest)
print("它的 parent      :", parent)

ck = saver.serde.loads_typed(ck_bytes)
print("\n=== checkpoint 的 7 个字段 ===")
for k, v in ck.items():
    print(f"  {k:18} = {str(v)[:80]}")
print("  channel_values 在存储里是:", ck.get("channel_values", "<不存在>"))

print("\n=== 值到底在哪：blobs ===")
for k, v in saver.blobs.items():
    thread, ns, ch, ver = k
    print(f"  ({thread!r}, {ns!r}, {ch!r}, v{ver}) -> type={v[0]}")
```

> 这一段的**预期输出**在下一节。**先写下你猜的**：`ck["channel_values"]` 打印出来会是什么？

---

## ④ 你应该看到

> **诚实标注**：本章所有"应该看到"都是**按源码推导的期望值**。本会话的 `pwsh` 被 Windows 沙箱 ACL 拦住（`grantNamedSecurityInfoW failed (Win32 5)`），**我没有在本机跑过这份脚本**。你跑出来的数字如果和我写的不一样，以你的为准，然后回来看是哪一步的推理错了——那正是认知缺口。ch05 的实测数字是另一条执行通道跑出来的，本章不冒用。

**A/B 段（记忆有无）：**

```
第一次: 2
第二次: 4
最早那句还在吗: 第1句
```

对照组（`compile()` 不传 checkpointer，或每次换 `thread_id`）：

```
第一次: 2
第二次: 2
最早那句还在吗: 第2句
```

**C 段（三个字典）：**

```
=== saver.storage 的层级 ===
第一层 thread_id : ['t1']
第二层 checkpoint_ns: ['']
第三层 checkpoint_id 个数: 4
最新 checkpoint_id: 0192f3c8-...（uuid6，形如 1ed4a2b0-... 开头是时间）
它的 parent      : 0192f3c8-...（比最新小一点的那个）
```

**为什么是 4 个**：两次 `invoke`，每次产生 1 个 `input`（step=-1）+ 1 个 `loop`（step=0）= 2 个。两次共 4 个。

**D 段（checkpoint 的 7 个字段）：**

```
=== checkpoint 的 7 个字段 ===
  v                  = 2
  ts                 = 2026-10-23T06:45:12.345678+00:00
  id                 = 0192f3c8-...
  channel_values     = {}
  channel_versions   = {'messages': 4, 'branch:to:bot': 2}
  versions_seen      = {'bot': {'branch:to:bot': 1, 'messages': 3}}
  updated_channels   = ['messages']
  channel_values 在存储里是: {}
```

**三个必须注意的地方**：

1. **`v = 2`**，不是注释里写的 1。信常量 `LATEST_VERSION = 2`。
2. **`channel_values = {}`** —— 空的！因为 `put()` 里 `c.pop("channel_values")` 把它拿走了。这是本章最反直觉的一行，你现在亲眼看到了。
3. **`channel_versions` 里出现了 `branch:to:bot`** —— 你可能没写过这个 channel。它是 LangGraph 自动加的**控制边**（"下一个该谁跑"），本质也是个 channel，也参与版本管理。所以 `versions_seen['bot']` 里记着"bot 已经消费了 `branch:to:bot` 的 v1"。

**E 段（blobs，值在这里）：**

```
=== 值到底在哪：blobs ===
  ('t1', '', 'messages', 1) -> type=json
  ('t1', '', 'messages', 2) -> type=json
  ('t1', '', 'messages', 3) -> type=json
  ('t1', '', 'messages', 4) -> type=json
  ('t1', '', 'branch:to:bot', 1) -> type=json
  ('t1', '', 'branch:to:bot', 2) -> type=json
```

**`messages` 有 4 个 blob，对应版本 1/2/3/4**。每一次 channel 更新就多一个版本、多一个 blob。旧 blob **不删**——这是时间旅行的物质基础：想回到 v2，只要 `channel_versions['messages'] = 2` 再读对应 blob 就行。

**F 段（`get_state_history`，时间轴）：**

```python
for snap in app.get_state_history(cfg):
    print(f"step={snap.metadata['step']:>2}  source={snap.metadata['source']:<7} "
          f"msgs={len(snap.values.get('messages', []))}  next={snap.next}")
```

预期（**从新到旧**，LangGraph 默认倒序）：

```
step= 0  source=loop    msgs=4  next=()
step=-1  source=input   msgs=3  next=('bot',)
step= 0  source=loop    msgs=2  next=()
step=-1  source=input   msgs=1  next=('bot',)
```

**这张表信息量很大**：

- `input` 检查点的 `msgs` 是 1 和 3（你交进去的那一条已经合并了），`next=('bot',)` 表示"下一个该跑 bot"。
- `loop` 检查点是 `msgs=2` 和 `msgs=4`，`next=()` 表示"跑完了，没有下一步"。
- **所以一次 `invoke` = 2 个检查点**，`input` 在前、`loop` 在后，靠 uuid6 的 `clock_seq` 保证顺序。

---

## ⑤ 三个破坏性实验

### 实验 1：删掉 `thread_id`

```python
app.invoke({"messages": [HumanMessage("hi")]}, {})    # 不传 configurable
```

**预期报错**（`langgraph` 在 `config` 校验处抛）：

```
ValueError: Checkpointer requires one or more of the following 'configurable' keys:
thread_id, checkpoint_ns, checkpoint_id
```

（`pregel/main.py:2685` 附近就是这个字符串。）

**要点**：报错发生在**图开始跑之前**，不是跑完发现存不了。所以"忘了 thread_id"不会静默丢数据，而是当场炸——这是好事。

### 实验 2：换 `thread_id`，但保留 saver

```python
cfg2 = {"configurable": {"thread_id": "t2"}}
r = app.invoke({"messages": [HumanMessage("我是新会话")]}, cfg2)
print(len(r["messages"]))    # 预期 1
```

**预期**：只有 1 条（或者 2 条，取决于 `bot` 是否返回）——总之**没有 t1 的历史**。

**要点**：`saver` 是同一个对象，但 `thread_id` 变了就是彻底隔离。这证明**隔离单位是 `thread_id`，不是 saver 实例**。多用户场景下你不需要"每个用户一个 saver"，只需要"每个用户一个 thread_id"。

### 实验 3：同 `thread_id`，但换一个**新的** `saver`

```python
saver2 = InMemorySaver()
app2 = g.compile(checkpointer=saver2)
r = app2.invoke({"messages": [HumanMessage("第三句")]}, cfg)   # 同一个 t1
print(len(r["messages"]))    # 预期 2（不是 6）
```

**预期**：失忆。虽然 `thread_id` 一样，但 `saver2` 里什么都没有。

**要点**：**状态在 saver 里，不在 app 里，也不在 thread_id 里。** `thread_id` 只是一把钥匙，`saver` 才是保险箱。这条在生产里极其重要——`InMemorySaver` 是进程内的，**重启即全丢**。官方 docstring 直接写了：

> Only use `InMemorySaver` for debugging or testing purposes. For production use cases we recommend installing `langgraph-checkpoint-postgres` and using `PostgresSaver` / `AsyncPostgresSaver`.

---

## ⑥ 自检题（5 道，答案就在下面，不用点开）

### Q1. `Checkpoint` 这个 dict 里，`channel_values` 在存储时是空的。值在哪？为什么这么设计？

**参考答案**

值在 `saver.blobs[(thread_id, checkpoint_ns, channel, version)]` 里，四元组 key 带版本号。

`put()` 第一件事就是 `values = c.pop("channel_values")`，然后 `for k, v in new_versions.items(): blobs[(thread, ns, k, v)] = dumps(values[k])`。

这么设计是为了**按版本去重**：某 channel 这一步没变，版本号不涨，就不在 `new_versions` 里，也就不写新 blob，checkpoint 通过 `channel_versions` 里的旧版本号指回旧 blob。10 轮对话里不变的字段只存 1 份。

读的时候 `_load_blobs(thread, ns, checkpoint["channel_versions"])` 现拼成 `channel_values`。所以 `channel_values` 是**视图**，不是**存储形态**。

### Q2. 为什么 `checkpoint_id` 用 `uuid6` 而不是 `uuid4`？

**参考答案**

因为 `get_tuple` 找"最新"用的是 `max(checkpoints.keys())`（`memory/__init__.py:286`）——纯字符串字典序比较。

`uuid6` 是 RFC 9562 里**按时间排序**的 UUID：前 48 位是 Unix 毫秒时间戳，所以**字符串序 == 时间序**。`uuid4` 完全随机，`max()` 会返回一个随机 id。

换成 uuid6 之后，"取最新"退化成一次字典 `max()`，不需要额外索引、不需要 `ORDER BY ts`、不依赖时钟单调。Postgres 里就是主键索引的 `DESC LIMIT 1`。

代价：你不能自己给 `checkpoint_id` 编语义（比如用时间戳当 id），因为框架依赖"字符串序=插入序"这个不变量。

### Q3. 从快照恢复时，`add_messages` 会重新跑一遍吗？为什么？

**参考答案**

**不会。**

决定性代码在 `channels/binop.py:117-125` 的 `from_checkpoint`：

```python
if checkpoint is not MISSING:
    empty.value = checkpoint        # 直接赋值，不调 operator
```

对比 `update()`（L147）：`self.value = self.operator(self.value, value)` —— `operator` **只在 `update()` 里被调用**。

所以 checkpoint 存的是**合并后的最终值**，恢复 = 赋值。reducer 只在"有新写入"时执行。

推论：reducer 不需要幂等（恢复不重放），但需要满足**结合律**——因为同一步内多个并行节点的写入会被依次 `operator` 合并，顺序由 `writes_sort_key` 决定，你不能指望它稳定。

### Q4. `metadata.step` 的 `-1` 和 `0` 分别代表什么？一次 `invoke` 产生几个 checkpoint？

**参考答案**

- `step == -1`：`source == "input"`，由 `invoke/stream/batch` 的输入产生。
- `step == 0`：`source == "loop"`，第一个 superstep。
- 之后每个 superstep 递增。

一次 `invoke` **至少 2 个**：1 个 `input`（step=-1）+ 1 个 `loop`（step=0）。

如果图里有工具调用循环（model → tools → model），会更多：1 个 input + 3 个 loop = 4 个。这个数字在做 HITL 和流式的时候天天要用——你要定位"第几步出问题"，就是数 step。

### Q5. `pending_writes` 什么时候非空？它解决了什么问题？

**参考答案**

一个 superstep **跑到一半崩了**的时候。

具体场景：三个并行节点，两个跑完并 `put_writes` 落盘了，第三个挂了。这时 writes 已经在 saver 里，但 `put`（生成新 checkpoint）还没发生。

下次同 `thread_id` 恢复时，`get_tuple` 把 `pending_writes` 一起带回来，框架**先应用这些 writes 再继续**——已完成的两个节点不重跑。

这就是 "durable execution" 和 "从头重跑" 的区别。没有它，任何一个节点失败都得整轮重来，那对长任务（比如跑了 20 分钟的 agent）是不可接受的。

顺带记住 `WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}`：普通写入用 0,1,2...，特殊写入占负数段避免冲突；而且 `put_writes` 对普通写入做幂等保护（已存在就跳过），对特殊写入允许覆盖（中断/恢复本来就该以最后一次为准）。

---

## ⑦ 承上启下

这一章你拿到的东西，其实是**三个能力的地基**：

1. **记忆**（ch13 三层记忆）：短期记忆 = `thread_id` + checkpointer。你现在知道它存在哪、怎么读、为什么重启会丢（`InMemorySaver` 是进程内的）。
2. **人在环**（ch14）：`interrupt()` 的本质是"在某个 checkpoint 上抛出一个特殊写入 `INTERRUPT`，然后停下来"。恢复就是"从那个 checkpoint 继续，把人的决定作为 `RESUME` 写入喂回去"。你刚才看到的 `WRITES_IDX_MAP` 里那两个负数，就是给这条路留的。
3. **时间旅行**（ch20）：`get_state_history` 拿到的是 `parent_config` 链。想回到某一步，就是拿那个 `checkpoint_id` 当 `config` 重新 invoke——框架会从那个 checkpoint 建 channels，然后**分叉**（`source="fork"`）。

但还有一个东西我没讲：**`checkpoint_ns` 到底怎么被子图用起来的**。ch15（多智能体四形态）里你会看到 subagents / handoffs 各写一遍，那时 `checkpoint_ns` 会从 `""` 变成 `"subagent<uuid>"`，而且**父图和子图的检查点是两套独立的链**。为什么必须独立——因为同一个子 agent 可能被并行调用三次，共用一条链会互相覆盖。这个坑不提前说，ch15 会卡住。

下一章 **ch07 结构化输出**（`agents/structured_output.py`）看起来跟持久化无关，其实有个隐蔽联系：`structured_response` 是 `AgentState` 里一个**只能出不能进**的字段（ch05 讲过），它同样会被写进 checkpoint——所以你在 `get_state_history` 里能看到它，但你在 `invoke` 时传它进去是无效的。

> **本章认知修正**（照契约写一条）：
> 我原本以为 checkpoint 是"把整个 state 序列化存下来"，实际上它**只存版本指针**，值按 `(channel, version)` 单独存在 `blobs` 里，靠内容不变就不写新 blob 来做去重。恢复时 `channel_values` 是现拼的。

---

## 附录 · 官方写法对照

官方 `short-term-memory` 页面的写法，和本章的差别只在**入口**：官方用 `create_agent(model, tools, checkpointer=saver)`，本章用裸 `StateGraph`。

`create_agent` 内部就是 `graph.compile(checkpointer=checkpointer)` —— 也就是说**它没有做任何额外的事**，本章讲的三层主键、blobs 去重、`from_checkpoint` 直接赋值，在 agent 场景下**一模一样**。

差别只有一个，值得单独记：

| | 裸 `StateGraph` | `create_agent` |
|---|---|---|
| state 字段 | 你自己定 | 固定 `AgentState`（`messages` / `jump_to` / `structured_response`，ch05 讲过） |
| 一次 invoke 的 checkpoint 数 | 2（input + 1 loop） | **取决于工具调用轮数**：无工具 2 个，一次工具调用 4 个 |
| `channel_ns` | 恒 `""` | 用了 middleware 子图 / subagent 时会变 |

想看 agent 场景下的实际样子，跑 `langchain_learn/05_official_short_term_memory.py`（ch05 附录那份，A–G 零 token），它已经覆盖了 `create_agent` × checkpointer 的三态对照。本章的 `06_checkpointer.py` 补的是**更底层**的视角：`InMemorySaver` 的三个字典、`blobs` 的版本去重、`from_checkpoint` 不重放 reducer。

两份配合看，你会得到一个完整的图：

```
create_agent(checkpointer=...)
        │
        ▼
graph.compile(checkpointer=saver)          ← 官方和本章在这里汇合
        │
        ▼
Pregel: get_tuple → channels_from_checkpoint → 跑 → put/put_writes
        │
        ├── storage[thread][ns][cp_id] = (ckpt_bytes, meta_bytes, parent)
        ├── blobs[(thread, ns, channel, ver)] = (type, bytes)     ← 值在这
        └── writes[(thread, ns, cp_id)][(task_id, idx)] = ...     ← 断点账本
```
