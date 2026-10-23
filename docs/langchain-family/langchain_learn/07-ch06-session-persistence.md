# 第 6 章 · 会话持久化：invoke 返回之后，state 去哪了

> 本章目标：**能解释"一次 invoke 到底写了几张快照、写在哪三张表里、怎么翻回去"，并且知道重放为什么会重复副作用**。预计 80 分钟。
>
> 环境锚点：`langchain 1.4.6` / `langchain-core 1.6.12` / `langgraph 1.2.16`。本章所有输出都是在本机跑出来的，不是推的。
>
> 官方锚点：<https://docs.langchain.com/oss/python/langgraph/checkpointers>（`/oss/python/langchain/short-term-memory` 只讲到"怎么用"，机制在这页）

---

## ① 本章你会得到什么

1. 能说出**一次 invoke 产生几张 checkpoint**，以及 `step=-1` 那张是干什么的
2. 能画出 checkpointer 的**三层主键**（`thread_id → checkpoint_ns → checkpoint_id`）和**三张表**（`storage` / `blobs` / `writes`）
3. 能解释"每张 checkpoint 都存全量 state"为什么不会爆内存——靠什么去重
4. 能回答第 5 章留下的问题：**恢复时 reducer 会被重放吗**（不会），以及它真正被调用的时机
5. 会用 `get_state` / `get_state_history` / `update_state` / **replay** 四件套，并知道 replay 的代价
6. 能在 `durability` 三档里选一个，并说清选错会丢什么

第 3 条和第 4 条是分水岭。会了这两条，"为什么它记得 / 为什么它忘了 / 为什么它崩在这一步"都能定位到具体哪张表。

---

## ② 源码解剖

本章碰四个文件，都在你的 `.venv` 里：

| 文件 | 里面是什么 |
|---|---|
| `.venv/Lib/site-packages/langgraph/checkpoint/memory/__init__.py` | `InMemorySaver.put`：**值怎么被拆进 blobs** |
| `.venv/Lib/site-packages/langgraph/pregel/_checkpoint.py` | `create_checkpoint`：版本号、`step`、`updated_channels` 怎么来 |
| `.venv/Lib/site-packages/langgraph/channels/binop.py` | `BinaryOperatorAggregate.from_checkpoint`：**恢复为什么不重放** |
| `langgraph/pregel/main.py`（`durability` 分支） | 三档落盘策略在哪一层生效 |

### 2.1 十二行代码，决定了"存什么、怎么去重"

`InMemorySaver.put` 的核心就这一段：

```python
c = checkpoint.copy()
thread_id = config["configurable"]["thread_id"]
checkpoint_ns = config["configurable"]["checkpoint_ns"]
values: dict[str, Any] = c.pop("channel_values")          # ← 主行里把值摘掉

for k, v in new_versions.items():                          # ← 只遍历"本次变了的 channel"
    self.blobs[(thread_id, checkpoint_ns, k, v)] = (
        self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
    )

self.storage[thread_id][checkpoint_ns].update(
    {
        checkpoint["id"]: (
            self.serde.dumps_typed(c),                      # 主行：只剩版本指针
            self.serde.dumps_typed(metadata),               # source / step / parents
            config["configurable"].get("checkpoint_id"),     # parent，串成一条链
        )
    }
)
```

四个动作，四件事：

1. **`pop("channel_values")`** —— 主行**不存值**。你在 `saver.storage[...]` 里解出来的 checkpoint 只有 `v / ts / id / channel_versions / versions_seen / updated_channels` 六个键，`channel_values` 是 `None`。
2. **`new_versions`** —— 只遍历这次真正变了的 channel。**没变的 channel 不写新 blob**，新 checkpoint 靠 `channel_versions` 指回旧 blob。
3. **`blobs[(thread, ns, channel, version)]`** —— 键里带 **version**，所以同一个 channel 的每个历史版本都在，且天然去重。
4. **`storage[thread][ns][id]`** —— 三层主键，值是扁平的三元组，第三项是 parent id。

于是"每个 checkpoint 存一份完整 state"这句话要修正成：**每个 checkpoint 存一份完整 state 的"版本引用表"，值按版本共享。** 只有真变了的 channel 才落新数据。

> 这也是官方 "Optimize checkpoint storage" 那一节讲的东西：默认行为已经是"每个 super-step 写每个 channel 的当前值"，而实现层用 blobs 版本化把它压下来了。真要再省，才需要 `DeltaChannel`（1.2+ 的 beta，把整值换成增量）。

实测（`06_checkpointer.py` E 段：`t1` 两轮 6 张、`t2` 一轮 3 张，两个 thread 的 blobs 合计 18 条）：

```text
blobs 共 18 条：
  ('t1', '', 'messages', v…02) -> type=msgpack, 181 bytes
  ('t1', '', 'messages', v…03) -> type=msgpack, 423 bytes
  ('t1', '', 'messages', v…05) -> type=msgpack, 603 bytes
  ('t1', '', 'messages', v…06) -> type=msgpack, 845 bytes
  ('t1', '', '__start__', v…02) -> type=empty, 0 bytes
  ('t1', '', 'branch:to:bot', v…02) -> type=null, 0 bytes
```

`messages` 只有 4 个版本，而 checkpoint 有 6 张——**版本数 ≠ 快照数**，这就是去重。

### 2.2 `step` 与 super-step：谁在调用 `create_checkpoint`

`create_checkpoint(checkpoint, channels, step, ...)` 干的事是：拿"上一张 checkpoint + 当前所有 channel"生成新的一张，版本号取自每个 channel 自己的计数器。

`step` 的计数方式决定了你看到的数字：

| step | 什么时候写 | `next` |
|---|---|---|
| `-1` | 输入写入口（`source="input"`） | `('__start__',)` |
| `0` | 第一个 super-step 开始前（`source="loop"`） | 该步要跑的节点 |
| `1..n` | 每个 super-step 结束后 | 下一步节点 / `()` |

所以**一次 invoke 的 checkpoint 数 = 1 + (super-step 数) + 1**——更准确地说：入口一张，每个 super-step 边界一张。

实测（`06_official_checkpointer.py` B 段，模型 + 一次工具调用）：

```text
    一次 invoke（中间调了一次工具）→ 5 张（新 → 旧）
    step  source  next             内容
       3  loop    ()               Huma:北京天气？ AIMe: Tool:北京：晴 2 AIMe:北京晴，26
       2  loop    ('model',)       Huma:北京天气？ AIMe: Tool:北京：晴 2
       1  loop    ('tools',)       Huma:北京天气？ AIMe:
       0  loop    ('model',)       Huma:北京天气？
      -1  input   ('__start__',)
```

`next` 是本章最好用的一个字段：**它告诉你"如果现在从这里续跑，下一步会执行谁"**。重放要挑 `next` 非空的那张，原因就在这（见 ⑤ 实验）。

### 2.3 `writes` 表：为什么"崩在半个 super-step"也能救

`storage` 存的是 super-step 边界的完整快照；`writes` 存的是**节点级别**的中间结果：

```python
self.writes[(thread_id, checkpoint_ns, checkpoint_id)][(task_id, idx)] = (channel, value)
```

一个 super-step 里可能有多个节点并行跑。每个节点一跑完，它的输出立刻写进 `writes`（这是 pending write）。所以：

- 同一 super-step 里 A 成功、B 抛错 → 恢复时 **A 不用重算**，直接读 `writes`；
- 但 `writes` **不是** `StateSnapshot`，时间旅行只能从"完整 checkpoint"（super-step 边界）恢复，不能从半个 super-step 恢复。

这条区分很重要：**"能续跑"和"能回滚"用的不是同一份数据。**

### 2.4 `from_checkpoint`：恢复为什么不是重放

第 5 章留了个问题：*checkpointer 存的是快照，reducer 是"合并"语义——从快照恢复时，reducer 会被再执行一次吗？*

答案在 `BinaryOperatorAggregate`（`add_messages` 背后那个 channel 类型）里：

```python
def from_checkpoint(self, checkpoint: Value) -> Self:
    empty = self.__class__(self.typ, self.operator)
    empty.key = self.key
    if isinstance(checkpoint, _DeltaSnapshot):
        checkpoint = checkpoint.value
    if checkpoint is not MISSING:
        empty.value = checkpoint        # ← 直接赋值，operator 一次都没调
    return empty
```

**恢复 = 把快照里的值直接赋给 channel，不调用 reducer。** 而写入才会调：

```python
def update(self, values: Sequence[Value]) -> bool:
    if self.value is MISSING:
        self.value = values[0]
        values = values[1:]
    for value in values:
        self.value = self.operator(self.value, value)   # ← 只有这里会调 reducer
```

实测（`06_checkpointer.py` G 段，会喊的 reducer）：

```text
第一次 invoke：
      >>> reducer 被调用 #1: left=0 条, right=1 条
      >>> reducer 被调用 #2: left=1 条, right=1 条
第二次 invoke（重点看 left 是几条）：
      >>> reducer 被调用 #3: left=2 条, right=1 条
      >>> reducer 被调用 #4: left=3 条, right=1 条
```

第二轮第一次调用时 `left=2`——如果恢复要把历史重放一遍，这里应该是 `left=0`、`left=1` 两连击。没有，说明**恢复是读值，不是重算**。

三条能直接用的结论：

1. **reducer 不必是幂等的**（不会被重放），但**必须满足结合律**——因为合并顺序可能因为分叉、重放而变，`(a+b)+c` 和 `a+(b+c)` 得一样。
2. **reducer 里仍然不能做副作用**：一次 invoke 里它会被调用**多次**（输入一次、每个写 `messages` 的节点各一次）。"不重放"≠"只调一次"。
3. 这也解释了"第二轮只传新消息"的成本：旧消息从 `channel_values` 直接读回，reducer 只处理新增的那条，**O(新增) 而不是 O(全量)**。

### 2.5 `checkpoint_id` 是 UUIDv6：字符串序 = 时间序

```text
最新 checkpoint_id = max(keys()) = 1f1ceb1a-92e5-6272-8004-4e038e5deda4
```

本机实测可以直接 `max(keys())` 取最新，因为 id 是 UUIDv6（前 8 位是时间戳）。换成 UUIDv4 就不成立——这也是为什么"按 id 排序"在 checkpoint 表里可用、在别的地方不要乱用。

在 Postgres 里这条更值钱：`ORDER BY checkpoint_id DESC LIMIT 1` 就是"取最新快照"，不需要额外时间字段。

### 2.6 `durability`：三档不是"存不存"，是"什么时候存"

`main.py` 在每次执行方法上接这个参数：`durability = config.get(CONF, {}).get(CONFIG_KEY_DURABILITY, "async")`——**默认是 `async`，不是 `exit`**。三档的差别用"落盘几张"最直观：

```text
  durability=sync   落盘  3 张  [step-1/input] [step0/loop] [step1/loop]
  durability=async  落盘  3 张  [step-1/input] [step0/loop] [step1/loop]
  durability=exit   落盘  1 张  [step1/loop]
```

`exit` 只在**跑完 / 出错 / 中断**时落一次，所以中间态没有。**HITL、断点续跑、崩溃恢复，都不许用 `exit`。**

---

## ③ 你敲这段

文件：`langchain_learn/06_checkpointer.py`，**A–H 段零 token、零网络**（连 LLM 都不需要，纯图 + 纯 saver），只有 H 段加 `--real` 才联网。

骨架（先照着敲标题和打印，细节自己填）：

```python
# A 记忆有无对照：compile(checkpointer=saver) vs compile()
# B thread_id 隔离：同一 saver 换钥匙 / 换 saver 同钥匙
# C 拆 saver.storage 的三层主键
# D checkpoint 的字段（注意 channel_values 是空的）
# E 拆 saver.blobs：值按 (channel, version) 存
# F get_state_history 时间轴 + update_state 造分叉
# G 会喊的 reducer：证明恢复不重放（带 assert）
# H --real：create_agent(checkpointer=...) 是同一套
```

### 逐段说清"为什么这么写"

**A 段为什么要一个"不调模型的节点"？** 因为本章要观察的是**存储**，不是模型的回答。`bot` 节点只回一句"我看到 N 条"，把"state 里有几条"变成可打印的数字，同时让整个脚本零 token。这是第 5 章 `RecordingFakeModel` 的极简版——**能用假节点就别用假模型**。

**C 段为什么要 `sorted(cps)` 而不是 `list(cps)`？** 字典顺序是插入顺序，不是时间顺序。`checkpoint_id` 是 UUIDv6，`sorted` 之后才是"从旧到新"。这也是 2.5 那条性质的第一次实际使用。

**D 段为什么要 `saver.serde.loads_typed(...)`？** 因为存储层存的是**序列化后的字节**，`('msgpack', b'\x86...')`。不解开就只能看到一串二进制。顺带你会发现 `metadata` 里没有 `writes`——和官方文档的示例不一致（官方示例是 2024 年的旧版本）。**文档会漂，print 不会。**

**E 段为什么要按 `(thread, ns, channel, version)` 排序打印？** 这是唯一能"看见去重"的姿势：`messages` 的版本数少于 checkpoint 数，说明有的 checkpoint 复用了旧 blob。

**G 段为什么要 `assert`？** 因为"恢复不重放"是本套设计里最容易记反的一条。断言写在脚本里，跑一次就校验一次：`after - before == 2`（第二轮只新增 2 次写入，不是把历史重新合并一遍）。**能被断言的知识，就别只写在笔记里。**

**F 段为什么 `update_state` 要传 `as_node="bot"`？** 因为 `update_state` 创建的新 checkpoint 需要知道"这次修改算谁写的"，才能决定下一步跑谁。不传时按最后执行的节点算；传了就等于"假装这条是 bot 写的"，于是下一步直接 `END`。

---

## ④ 你应该看到

先预测，再对照。下面全部是本机输出（只摘关键行）。

### 4.1 A/B：有记忆 vs 没记忆，换钥匙 vs 换库

```text
第一次 invoke: 2 条
第二次 invoke: 4 条   ← 有记忆应该是 4
最早那句还在吗: '第1句'   ← 应该还是「第1句」

-- 对照组：同一个图，但 compile() 不传 checkpointer --
第一次: 2 条 / 第二次: 2 条   ← 没有累积
第二次的第一条: '第2句'   ← 历史被覆盖了

thread_id=t2 第一次: 2 条   ← t1 的历史完全看不到
t1 现在有 4 条，t2 有 2 条   ← 两条独立的链

-- 再换 saver 实例（thread_id 不变）--
新 saver + 老 thread_id: 2 条   ← 照样失忆
```

最后一行是本章最该记住的一句：**状态在 saver 里，`thread_id` 只是一把钥匙，不是保险箱。**

### 4.2 C：三层主键与 parent 链

```text
第一层 thread_id     : ['t1', 't2']
第二层 checkpoint_ns : ['']   ← 根图是空字符串
第三层 checkpoint_id : 6 个
    1f1ceb1a-92d7-…  parent=None
    1f1ceb1a-92e0-…  parent=1f1ceb1a-92d7-…
    1f1ceb1a-92e0-…  parent=1f1ceb1a-92e0-…
    ...
```

`parent` 串成一条链，`get_state().parent_config` 就是这条链的上一环。**时间旅行 = 沿着这条链往回走，不是翻日志。**

### 4.3 D：主行里没有值

```text
  v                  = 4
  ts                 = 2026-10-23T07:16:26.811454+00:00
  id                 = 1f1ceb1a-92e5-6272-8004-4e038e5deda4
  channel_versions   = {'__start__': '…0005…', 'messages': '…0005…', ...}
  versions_seen      = {'__input__': {}, '__start__': {...}, ...}
  updated_channels   = ['messages']

  channel_values 在存储里 = None   ← 空的！
```

`updated_channels` 也值得看一眼：它就是 `new_versions` 的来源——**这张快照相对上一张，到底动了哪些 channel**。

### 4.4 F：历史是一棵能分叉的树

```text
从新到旧：
  step= 4  source=loop    msgs=4  next=()        id=1f1ceb1a-92e5…
  step= 3  source=loop    msgs=3  next=('bot',)  id=1f1ceb1a-92e2…
  step= 2  source=input   msgs=2  next=('__start__',)  id=1f1ceb1a-92e2…
  step= 1  source=loop    msgs=2  next=()        id=1f1ceb1a-92e0…
  step= 0  source=loop    msgs=1  next=('bot',)  id=1f1ceb1a-92e0…
  step=-1  source=input   msgs=0  next=('__start__',)  id=1f1ceb1a-92d7…

-- 拿最近那张 input 检查点（还没合并新消息），改一句话，造一个 fork --
选中: step=2 msgs=2
fork 出来的新 checkpoint: 1f1ceb1b-d1ca…
新分支 msgs=3, 最后一条='我改主意了，走另一条路'
原分支还在: t1 最新 msgs=4
```

注意 `step=1` 那条：`next=()` 却只 `msgs=2`——那是第一轮跑完的瞬间。**分叉之后同一个 step 会出现两张**（附录 A6.6 的 replay、或 F 段的 fork 之后都能看到），那不是 bug，是"同一 step、两条分支"。

### 4.5 G：reducer 的调用时机

```text
第一次 invoke：
      >>> reducer 被调用 #1: left=0 条, right=1 条
      >>> reducer 被调用 #2: left=1 条, right=1 条
第二次 invoke（重点看 left 是几条）：
      >>> reducer 被调用 #3: left=2 条, right=1 条
      >>> reducer 被调用 #4: left=3 条, right=1 条
  ✅ 断言通过：after - before == 2
```

再补一个两节点的实测（把你的 `bot` 拆成 `n1`/`n2`，各写一条消息）：

```text
两个节点：调用序列 = [(0, 1), (1, 1), (2, 1)]
第二轮（恢复 3 条后）：调用序列 = [(3, 1), (4, 1), (5, 1)]  结束 state = 6
```

**写入几次就调几次 reducer**（输入 1 次 + 每节点 1 次）；第二轮从 `left=3` 起步，证明恢复没有重放。

---

## ⑤ 三个破坏性实验

### 实验 A：忘了传 `thread_id`

```python
agent = create_agent(model, checkpointer=InMemorySaver())
agent.invoke({"messages": [...]})        # 没有 config
```

**你会看到**（实测原文）：

```text
ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id
```

**结论**：checkpointer 没有"默认线程"。报错信息明确告诉你三件可传的东西——`thread_id` 必填，`checkpoint_ns` 和 `checkpoint_id` 是给"读历史/重放"用的。**看到这条报错，先查 config，别查 prompt。**

### 实验 B：换了 saver，`thread_id` 照抄

```python
app1 = build(InMemorySaver())   # 存了两轮
app2 = build(InMemorySaver())   # 新 saver
app2.invoke({"messages": [HumanMessage("第三句")]}, {"configurable": {"thread_id": "t1"}})
```

**你会看到**：`2 条`——好像第一次进这个 thread。

**结论**：`thread_id` 只是**主键**，不是**存储**。生产上换库、换环境、换连接串，历史就断了。这条解释了"本地能记住、测试环境记不住"的一类玄学问题：不是代码分支，是 saver 换了。

### 实验 C：往 state 里塞不可序列化的对象

```python
def put_lock(state):
    import threading
    return {"payload": threading.Lock()}      # 任何 saver 都存不了
```

**你会看到**（实测原文）：

```text
TypeError: Type is not msgpack serializable: _thread.lock
```

**结论**：默认 `JsonPlusSerializer` 只认 msgpack 能表达的类型。要存 dataframe / 自定义类，得 `JsonPlusSerializer(pickle_fallback=True)`——但那等于**允许反序列化时执行任意代码**，只在你完全控制写入方时用。

### 附加观察：`durability="exit"` 的代价

同一张图、同一句输入，只改 `durability`：

```text
sync   3 张   [step-1/input] [step0/loop] [step1/loop]
async  3 张   [step-1/input] [step0/loop] [step1/loop]
exit   1 张   [step1/loop]
```

`exit` 跑完就没中间态：**中断了、崩了，都没有可以续跑的位置**。这不是 bug，是它省下来的 IO 换来的。

---

## ⑥ 自检题

答不上来就别进下一章。答案就在题目下面。

**1. 一次 `agent.invoke({"messages": [...]})`（中间调了一次工具）会产生几张 checkpoint？分别是什么？**

**参考答案**

5 张：`step=-1`（输入写入口，`source="input"`）+ 4 张 `source="loop"`（进 model 前、model 出 tool_calls 后、tools 执行完、model 出最终回答后）。规律是"入口 1 张 + 每个 super-step 边界 1 张"。没有工具的单节点图则是 3 张。

**2. 为什么 `saver.storage` 里解出来的 checkpoint 没有 `channel_values`？值存在哪？**

**参考答案**

因为 `InMemorySaver.put` 第一件事就是 `c.pop("channel_values")`，值被拆进 `blobs[(thread_id, checkpoint_ns, channel, version)]`，主行只留 `channel_versions` / `versions_seen` / `updated_channels` 这些版本指针。所以"每张快照存全量 state"实际是"每张快照存引用表，值按版本共享"。

**3. 从快照恢复时，`add_messages` 会被重放一遍吗？会，不会，为什么？**

**参考答案**

不会。`BinaryOperatorAggregate.from_checkpoint` 直接 `empty.value = checkpoint`，operator 一次都不调；只有 `update()`（真实写入）才调。所以 reducer 不必幂等，但要满足结合律；也所以它**仍然不能有副作用**——一次 invoke 里它会被调用多次（输入 1 次 + 每个写该字段的节点 1 次）。

**4. `update_state` 和 replay 都会"回到过去"，区别是什么？各自会重复什么副作用？**

**参考答案**

`update_state` 是**改值造新分支**：不重跑任何节点，只写一张 `source="update"` 的新 checkpoint，之后的续跑用改后的值。replay 是**从某个 checkpoint 重跑后续节点**：`invoke(None, cfg + checkpoint_id)`，`next` 之后的节点会真的再执行，**LLM 调用、工具调用、发邮件这类副作用都会再来一遍**。回滚用 `update_state` 或继续拿旧分支的 id 跑；重放只用于"我想换一个输入再走一遍"。

**5. `durability="exit"` 丢了什么？哪类场景绝对不能选它？**

**参考答案**

丢中间态：只在跑完/出错/中断时落一次，`get_state_history` 里只有最终那张。需要 HITL 审批、断点续跑、崩溃恢复的场景不能选；纯批处理、失败了整批重跑也无所谓的场景才可以。

---

## ⑦ 承上启下

现在你手上有一套完整的"记忆机器"：`thread_id` 分抽屉、`checkpoints` 存快照、`blobs` 按版本去重、`reducer` 负责合并、`get_state_history` 让你回到任意一步。

但有个东西一直在"借住"——`structured_response`。第 5 章我们只说它挂了 `OmitFromInput`，所以你**塞不进去**；可它到底是谁写进去的？`response_format=` 传一个 Pydantic 类，agent 的图会多出什么？为什么有时候它返回的是 `dict`、有时候是模型实例、有时候直接 `None`？

更麻烦的是：**它同样会被 checkpointer 存下来。** 于是"结构化输出没拿到"可能有三条完全不同的病因——模型没按 schema 回、解析失败被吞、或者你从错的 checkpoint 读了 state。第 7 章把那层拆开。

> 想先动手的话：拿本章的 `saver` 给第 5 章的 agent 挂上，然后 `response_format=...` 跑两轮，用 `get_state_history` 逐张看 `structured_response` 是哪一张开始有值的。这个观察会让你在下一章省一半时间。

---

## 附录 · 官方写法对照（checkpointers 页）

> 官方锚点：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
> 代码：`langchain_learn/06_official_checkpointer.py`（A–I 段全离线，假模型只数消息条数；`--real` 才真发请求）

正文拆的是"存储层怎么实现"（`put` 的十二行、`from_checkpoint`、`step` 计数）；这一节换成官方视角：**官方页按"线程 → 快照 → 读状态 → 改状态 → 落盘策略 → 存储与库"组织**，它给的是 API 面和边界，不是实现。

| 官方小节 | 它解决什么问题 | 对应代码段 |
|---|---|---|
| Threads | 记忆按什么分堆 | A |
| Checkpoints · Super-steps | 一次运行几张快照 | B |
| Checkpoint namespace | 子图的状态存哪 | C |
| Get state（StateSnapshot） | 快照里有哪些字段 | D |
| Get state history | 怎么按条件找到某一帧 | E |
| Replay | 从旧帧重跑 | F |
| Update state | 改值、分叉、当 HITL 底座 | G |
| Durability modes | 什么时候落盘 | H |
| Optimize storage / Libraries | 值怎么存、库怎么选、有哪些坑 | I |

### A6.1 Threads：三个容易混淆的点

```text
  saver 里的一级键 = ['alice', 'bob']
    alice  checkpoint= 6  最新 state=Huma:alice  AIMe:答1：见1条 Huma:alice  AIMe:答2：见3条
    bob    checkpoint= 6  最新 state=Huma:bob 第0 AIMe:答3：见1条 Huma:bob 第1 AIMe:答4：见3条
  模型每次看到的条数 = [1, 3, 1, 3]（两个 thread 各 1→3，互不串味）
```

- **换 thread_id = 换一段对话**，还在同一个 saver 里（同一个数据库）；
- **换 saver = 换一个库**，thread_id 一样也翻不到旧照片；
- **不传 thread_id = 直接 `ValueError`**，没有默认线程。

### A6.2 Super-steps：`step` 从 -1 开始

官方那张示意图讲的就是"input → 每个 super-step 一张"。本机实测的 `next` 序列（A6.2 表见正文 2.2）可以直接当调试地图用：**看 `next` 就知道下一步是谁**。

### A6.3 Checkpoint namespace：父图翻不到子图

```text
  ns=''                                             checkpoint=3
  ns='child:59f53290-d5f5-8dfa-b8ea-bb9119709233'   checkpoint=3
  父图 get_state_history 只列到 3 张（全是 ns='' 的）
```

格式：`""` = 根图；`"节点名:uuid"` = 该节点调用的子图；嵌套用 `|` 连接。

**结论**：父图 `get_state_history` 永远只列 `ns=""` 的那堆。想跨图共享，用 `state` 或 `store`——**不要试图"去读子图的 checkpoint"，你手上没有那个 ns**。

### A6.4 StateSnapshot：官方文档和本机对不上的两处

```text
  官方文档里 metadata 的示例是 {'source','writes','step'}；
  本机 1.2.16 实测是 ['ls_integration', 'parents', 'source', 'step']。
  官方还写 StateSnapshot 字段含 'interrupts'，本机没有——它在 snap.tasks[i].interrupts 里。
```

七个字段的本机实测（`values / next / config / metadata / created_at / parent_config / tasks`）见正文 4.2、4.3。

**这条不是挑错，是给你一个习惯**：任何"按 metadata 的 key 写判断逻辑"的代码，升级前先 `print(sorted(snap.metadata))`。文档示例停留在旧版本，这很常见。

### A6.5 History：官方那四种检索姿势

```text
  ① 找'模型执行前'那张：next==('model',) → step=3
  ② 按 step 找：step=1 → 有（msgs=2）
  ③ 找手工改过的分支：source=='update' → 目前 0 张
  ④ 找中断点：any(t.interrupts for t in s.tasks)
```

两个坑：

- `get_state_history` 返回的是**生成器**（本机实测 `type=generator`），`len()` 会失败。先 `list()` 一次，否则第二遍就空了；
- 它是**从新到旧**，要"最早那张"用 `[-1]`，别用 `[0]`。

### A6.6 Replay：三条边界

```text
  起点：step=3，next=('model',)，msgs=3
  invoke(None, cfg+checkpoint_id) 之后：模型调用 2 → 3 次
  历史条数 6 → 8；同一 step 现在有 2 张
```

1. **不是撤销**：旧分支还在，新结果另起一支；
2. **会真的再执行**：LLM、工具、写库、发邮件都会重来一遍；
3. **挑对起点**：要挑 `next` 非空的那张（`next=()` 的重放等于什么都没发生——这是最容易踩的空转）。

### A6.7 Update state：值过 reducer，`as_node` 决定下一步

```text
  改前：'答1：见1条'（msgs=2）
  改后：'我改过的回答'（msgs=2）
  metadata.source = 'update'   next = ()
  历史条数 = 4（旧的没被删，新分支挂在后面）
  续跑时模型看到 = ['原问题', '我改过的回答', '接着说']
```

- 值**同样要过 reducer**：有 reducer 的字段是累积语义，不是赋值；
- `as_node=` 决定"这次修改算谁写的"，从而决定下一步跑谁（HITL 改参数重跑就靠这个）；
- 同 id 替换、新 id 追加——第 5 章 B 段的老规矩，在这里表现为"msgs 不变 / 变多"。

### A6.8 Durability：三档的实测落盘数

```text
  durability=sync   落盘  3 张  [step-1/input] [step0/loop] [step1/loop]
  durability=async  落盘  3 张
  durability=exit   落盘  1 张  [step1/loop]
```

`exit` 最快，代价是崩了没得救。**默认不是 `exit`**——想省 IO 得自己传，别猜默认值。

### A6.9 Storage & Libraries：官方 Troubleshooting 的四条

```text
  checkpoint 数 = 6，messages 的 blob 数 = 4
  最新一张主行里的字段 = ['channel_versions', 'id', 'ts', 'updated_channels', 'v', 'versions_seen']
  序列化器 = JsonPlusSerializer（msgpack）
```

四条官方坑，都是线上真会遇到的：

| 坑 | 现场表现 | 处理 |
|---|---|---|
| `thread_id` 超 255 字符 | Postgres 直接报列长度错误 | 用 uuid7 / hash，别拼长串 |
| `MemorySaver` 不跨进程 | A 进程存、B 进程读不到 | 生产换 `PostgresSaver`（记得 `setup()`） |
| checkpoint 无限增长 | 延迟和存储费用慢慢涨 | 定期删旧 thread / 旧 checkpoint，或设保留策略 |
| 父图读不到子图状态 | 明明子图写了，父图没变 | 各管各的 ns，跨图用 `store` / 共享 state |

库的选择，一句话版：

- **教学/测试** → `InMemorySaver`
- **本地开发** → `SqliteSaver`（`pip install langgraph-checkpoint-sqlite`）
- **生产** → `PostgresSaver`（`langgraph-checkpoint-postgres` + `psycopg[binary]`，`setup()` 建表）
- **异步栈** → 对应 `Async*` 版本，别在 `async` 里调同步 saver

### 认知修正

> 我原本以为"持久化"就是把 state 存成 JSON、下次读回来；实际上它是**三层主键 + 三张表 + 一套版本号**：主行只存指针，值按 `(channel, version)` 去重，`writes` 单独存节点级中间结果，恢复靠 `from_checkpoint` 直接赋值而不是重算。**弄清楚这一层，你才能解释"为什么它记得、为什么它丢了、为什么重放会重复扣款"这三件看起来无关的事。**

## 附 · 114 行复刻版

`langchain_learn/06_lite_checkpoint.py` 不依赖 langgraph，只用标准库把上面这套机制写了一遍（标识符全英文，注释中文），四个方法对应四件事：

| 方法 | 一句话 |
|---|---|
| `save` | 存一帧：`step` / `next` / 父档 / 版本表；值没变就复用旧版本 |
| `head` | 取这个 `thread` 最新的一帧 |
| `load` | 照版本表把值捞回来（直接读，不重放 reducer） |
| `history` | 从新到旧列一遍 |

两轮对话的输出，正好复现了真 saver 的形状：

```
c01  step=-1  next=model   版本表={'messages': 1}
c02  step= 0  next=model   版本表={'messages': 1}   ← 值没变，复用 v1
c03  step= 1  next=tools   版本表={'messages': 2}
c04  step= 2  next=model   版本表={'messages': 3}
c05  step= 3  next=None    版本表={'messages': 4}

快照 8 张，messages 只存了 6 份
```

两个结论一眼可见：**帧数 > 内容份数**（帧是时序，内容是值）；**改一帧不会动原帧**（`load` 出来的是深拷贝，`save` 时新版本号另起）。
