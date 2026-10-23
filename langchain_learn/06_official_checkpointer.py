"""第 6 章 · 官方对照：checkpointers 页的六层——线程、super-step、快照、历史、重放、落盘策略

对应官方页：https://docs.langchain.com/oss/python/langgraph/checkpointers
对应教材：docs/langchain-family/langchain_learn/07-ch06-session-persistence.md

为什么锚点换成 LangGraph 页：`/oss/python/langchain/short-term-memory` 只讲到
"checkpointer 让 agent 记得上一轮"（第 5 章附录已覆盖），而"存了什么、怎么翻历史、
怎么重放、什么时候落盘"全在 checkpointers 这一页。第 6 章要往下走一层，只能读它。

官方那一页的骨架：
  Why / Core concepts
    Threads            → A  thread_id：存储的主键
    Checkpoints        → B  每个 super-step 一张；C  checkpoint namespace（子图）
  Get and update state
    Get state          → D  StateSnapshot 的七个字段
    Get state history  → E  按 next / step / source 检索
    Replay             → F  从旧 checkpoint 重跑（会再调模型）
    Update state       → G  改值不改历史，顺便分叉
  Durability modes     → H  sync / async / exit 各落几张
  Optimize storage     → I  blobs 按版本去重
  Libraries            → I  InMemory / Sqlite / Postgres + 三个官方坑

A–I 全部离线（假模型只数消息条数），零 token；J 段加 --real 才真发请求。
"""

from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import Field
from typing_extensions import TypedDict

from _official_common import banner, real_enabled, short, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/06_official_checkpointer.py"


class MemoryProbe(BaseChatModel):
    """零 token 假模型：记下每次收到几条消息，回一句"答N：见M条"。"""

    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "memory-probe"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        self.seen.append(list(messages))
        message = self.script.pop(0) if self.script else AIMessage(f"答{len(self.seen)}：见{len(messages)}条")
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴 26℃"


def msg_brief(messages) -> str:
    return " ".join(f"{type(m).__name__[:4]}:{str(m.content)[:6]}" for m in messages)


# --------------------------------------------------------------------------
banner("A", "Threads：thread_id 是 checkpointer 的主键")
# --------------------------------------------------------------------------
# 官方原话：thread_id 是"a unique ID assigned to each checkpoint saved by a checkpointer"。
# 翻译成可验证的话：同一个 saver 里，thread_id 决定你翻到哪一堆照片。
probe = MemoryProbe()
saver = InMemorySaver()
agent = create_agent(probe, checkpointer=saver)

for thread in ("alice", "bob"):
    for turn in range(2):
        agent.invoke(
            {"messages": [{"role": "user", "content": f"{thread} 第{turn}问"}]},
            {"configurable": {"thread_id": thread}},
        )

print(f"  saver 里的一级键 = {list(saver.storage)}")
for thread in saver.storage:
    state = agent.get_state({"configurable": {"thread_id": thread}})
    history = list(agent.get_state_history({"configurable": {"thread_id": thread}}))
    print(f"    {thread:6s} checkpoint={len(history):>2}  最新 state={msg_brief(state.values['messages'])}")
print(f"  模型每次看到的条数 = {[len(x) for x in probe.seen]}（两个 thread 各 1→3，互不串味）")

print("\n  三个容易混淆的点：")
print("    · 换 thread_id = 换一段对话，但还在同一个 saver 里（同一个数据库）")
print("    · 换 saver = 换一个库，thread_id 再一样也翻不到旧照片")
print("    · 不传 thread_id → 直接报错。checkpointer 没有'默认线程'这回事")

# --------------------------------------------------------------------------
banner("B", "Checkpoints：一次 invoke 不是一张，是每个 super-step 一张")
# --------------------------------------------------------------------------
probe = MemoryProbe(
    script=[
        AIMessage("", tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "c1"}]),
        AIMessage("北京晴，26℃。"),
    ]
)
saver = InMemorySaver()
agent = create_agent(probe, tools=[get_weather], checkpointer=saver)
cfg = {"configurable": {"thread_id": "step"}}
agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]}, cfg)

history = list(agent.get_state_history(cfg))
print(f"  一次 invoke（中间调了一次工具）→ {len(history)} 张（新 → 旧）")
print(f"    {'step':>4}  {'source':<7} {'next':<16} 内容")
for snap in history:
    print(f"    {snap.metadata['step']:>4}  {snap.metadata['source']:<7} {str(snap.next):<16} {msg_brief(snap.values['messages'])}")

print("\n  对应关系（官方那张图讲的就是这个）：")
print("    source=input  → 你传进来的消息先落一张，next=('__start__',)")
print("    step=0        → 进 model 之前一张，next=('model',)")
print("    step=1        → model 出完 tool_calls 一张，next=('tools',)")
print("    step=2        → 工具回完一张，next=('model',)")
print("    step=3        → 模型给出最终回答，next=() —— 图跑完了")
print("  所以'长对话＝一次一张'是错的：**一个 super-step 一张，step 从 -1 开始数**。")

# --------------------------------------------------------------------------
banner("C", "Checkpoint namespace：子图有自己的抽屉")
# --------------------------------------------------------------------------


class SubState(TypedDict):
    messages: Annotated[list, add_messages]


sub_builder = StateGraph(SubState)
sub_builder.add_node("inner", lambda state: {"messages": [AIMessage("子图产出", id="sub1")]})
sub_builder.add_edge(START, "inner")
sub_builder.add_edge("inner", END)
subgraph = sub_builder.compile()


class ParentState(TypedDict):
    messages: Annotated[list, add_messages]


parent_builder = StateGraph(ParentState)
parent_builder.add_node("child", subgraph)
parent_builder.add_edge(START, "child")
parent_builder.add_edge("child", END)
ns_saver = InMemorySaver()
parent = parent_builder.compile(checkpointer=ns_saver)
parent.invoke({"messages": [HumanMessage("hi", id="h1")]}, {"configurable": {"thread_id": "ns"}})

for ns, entries in ns_saver.storage["ns"].items():
    print(f"  ns={ns!r:46s} checkpoint={len(entries)}")
print(f"  父图 get_state_history 只列到 {len(list(parent.get_state_history({'configurable': {'thread_id': 'ns'}})))} 张（全是 ns='' 的）")
print("\n  官方给的格式：'' = 根图；'节点名:uuid' = 该节点调用的子图；嵌套用 '|' 连接。")
print("  这条决定了：**父图翻不到子图的快照**。跨图要共享的东西放 state 或 store，")
print("  而不是'我去读子图的 checkpoint'——那是拿不到 ns 的。")

# --------------------------------------------------------------------------
banner("D", "Get state：StateSnapshot 七个字段，官方文档有一处和本机对不上")
# --------------------------------------------------------------------------
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "snap"}}
agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, cfg)
snap = agent.get_state(cfg)

field_meaning = [
    ("values", "该 checkpoint 的所有 channel 值", snap.values),
    ("next", "下一步要跑的节点；() = 结束", snap.next),
    ("config", "thread_id / checkpoint_ns / checkpoint_id", snap.config["configurable"]),
    ("metadata", "source / step / parents", snap.metadata),
    ("created_at", "写入时间", snap.created_at),
    ("parent_config", "上一个 checkpoint 的 config", (snap.parent_config or {}).get("configurable")),
    ("tasks", "本步待执行 / 中断的任务", snap.tasks),
]
for name, meaning, value in field_meaning:
    print(f"  {name:13s} {meaning}")
    print(f"  {'':13s} = {short(value, 96)}")

print("\n  官方文档里 metadata 的示例是 {'source','writes','step'}；")
print(f"  本机 1.2.16 实测是 {sorted(snap.metadata)}。")
print("  官方还写 StateSnapshot 字段含 'interrupts'，本机没有——它在 snap.tasks[i].interrupts 里。")
print("  结论：**按版本 print 一遍再写判断逻辑**，别照抄文档的 key。")

# --------------------------------------------------------------------------
banner("E", "Get state history：官方那四种检索姿势")
# --------------------------------------------------------------------------
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "hist"}}
for text in ("第一问", "第二问"):
    agent.invoke({"messages": [{"role": "user", "content": text}]}, cfg)

history = list(agent.get_state_history(cfg))
print(f"  两轮共 {len(history)} 张，按时间**从新到旧**：")
for snap in history:
    print(f"    step={snap.metadata['step']:>2} source={snap.metadata['source']:<6} next={str(snap.next):<14} {msg_brief(snap.values['messages'])}")

before_model = next(s for s in history if s.next == ("model",))
by_step = next(s for s in history if s.metadata["step"] == 1)
updates = [s for s in history if s.metadata["source"] == "update"]
print(f"\n  ① 找'模型执行前'那张：next==('model',) → step={before_model.metadata['step']}")
print(f"  ② 按 step 找：step=1 → {'有' if by_step else '无'}（msgs={len(by_step.values['messages'])}）")
print(f"  ③ 找手工改过的分支：source=='update' → 目前 {len(updates)} 张（G 段之后会多一张）")
print("  ④ 找中断点：any(t.interrupts for t in s.tasks) —— 本文件没上 HITL，第 14 章再演示")
print("\n  history 是**生成器**，别直接用 len()；先 list() 一次，否则第二遍就空了。")

# --------------------------------------------------------------------------
banner("F", "Replay：从旧 checkpoint 重跑，模型会被再叫一次")
# --------------------------------------------------------------------------
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "replay"}}
agent.invoke({"messages": [{"role": "user", "content": "第一问"}]}, cfg)
agent.invoke({"messages": [{"role": "user", "content": "第二问"}]}, cfg)
history = list(agent.get_state_history(cfg))

start = next(s for s in history if s.next == ("model",))
print(f"  起点：step={start.metadata['step']}，next=('model',)，msgs={len(start.values['messages'])}")
before_calls = len(probe.seen)
replay_cfg = {"configurable": {"thread_id": "replay", "checkpoint_id": start.config["configurable"]["checkpoint_id"]}}
result = agent.invoke(None, replay_cfg)
print(f"  invoke(None, cfg+checkpoint_id) 之后：模型调用 {before_calls} → {len(probe.seen)} 次")
print(f"  拿到的结果 = {msg_brief(result['messages'])}（msgs={len(result['messages'])}）")

after = list(agent.get_state_history(cfg))
print(f"  历史条数 {len(history)} → {len(after)}；同一 step 现在有 "
      f"{len([s for s in after if s.metadata['step'] == start.metadata['step'] + 1])} 张")
print("\n  官方把这叫 Replay，三条边界要记住：")
print("    · 重放**不是撤销**：旧分支还在，新结果另起一支（这就是时间旅行的'分支'语义）")
print("    · checkpoint 之后的东西会**真的再执行**：LLM 调用、写数据库、发邮件都会再来一遍")
print("    · 想从'某节点之前'重放，就挑 next==(那个节点,) 的那张；挑错（next=()）会什么都没发生")

# --------------------------------------------------------------------------
banner("G", "Update state：改值不改历史，写完就是一个新分支")
# --------------------------------------------------------------------------
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "update"}}
agent.invoke({"messages": [{"role": "user", "content": "原问题"}]}, cfg)

before = agent.get_state(cfg)
last = before.values["messages"][-1]
print(f"  改前：{last.content!r}（msgs={len(before.values['messages'])}）")

new_cfg = agent.update_state(cfg, {"messages": [AIMessage("我改过的回答", id=last.id)]})
after = agent.get_state(cfg)
print(f"  改后：{after.values['messages'][-1].content!r}（msgs={len(after.values['messages'])}）")
print(f"  metadata.source = {after.metadata['source']!r}   next = {after.next}")
print(f"  历史条数 = {len(list(agent.get_state_history(cfg)))}（旧的没被删，新分支挂在后面）")

agent.invoke({"messages": [{"role": "user", "content": "接着说"}]}, cfg)
print(f"  续跑时模型看到 = {[str(m.content)[:8] for m in probe.seen[-1]]}")

print("\n  官方强调的两点：")
print("    · update_state 的值**同样要过 reducer**：有 reducer 的字段是累积，不是赋值")
print("    · as_node='model' 决定'这次修改算谁写的'，进而决定下一步跑谁 —— HITL 的底座")
print("    · 传同一个 id 是替换，传新 id 是追加（这条第 5 章 B 段讲过，这里看到它对 checkpoint 的影响）")

# --------------------------------------------------------------------------
banner("H", "Durability modes：不是'存不存'，是'什么时候落盘'")
# --------------------------------------------------------------------------
for mode in ("sync", "async", "exit"):
    probe = MemoryProbe()
    agent = create_agent(probe, checkpointer=InMemorySaver())
    mode_cfg = {"configurable": {"thread_id": f"dur-{mode}"}}
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, mode_cfg, durability=mode)
    snaps = list(agent.get_state_history(mode_cfg))
    print(f"  durability={mode:<6} 落盘 {len(snaps):>2} 张  " +
          " ".join(f"[step{s.metadata['step']}/{s.metadata['source']}]" for s in reversed(snaps)))

print("\n  官方三档，从最不耐久到最耐久：")
print("    exit  ── 只在跑完/出错/中断时落一次。上面只剩最终那张，中间态没有")
print("    async ── 下一步执行时异步写。快，但进程崩了可能丢最后一张")
print("    sync  ── 下一步开始前同步写。最稳，IO 开销最大")
print("  选法：要 HITL / 要断点续跑 → sync 或 async；纯批处理、崩了重跑即可 → exit。")
print("  默认值是 async（pregel/main.py：config 里没配就用 'async'），不是 exit —— 别猜。")

# --------------------------------------------------------------------------
banner("I", "Optimize storage / Libraries：值怎么存、库怎么选")
# --------------------------------------------------------------------------
probe = MemoryProbe()
saver = InMemorySaver()
agent = create_agent(probe, checkpointer=saver)
cfg = {"configurable": {"thread_id": "blob"}}
for turn in range(2):
    agent.invoke({"messages": [{"role": "user", "content": f"第{turn}轮"}]}, cfg)

checkpoints = list(agent.get_state_history(cfg))
message_blobs = [k for k in saver.blobs if k[0] == "blob" and k[2] == "messages"]
print(f"  checkpoint 数 = {len(checkpoints)}，messages 的 blob 数 = {len(message_blobs)}")
print("  ↑ 数量不相等：**没变的 channel 不重写**，新 checkpoint 靠 channel_versions 指回旧 blob")
print(f"  blob 的键 = (thread_id, checkpoint_ns, channel, version)，共 {len(saver.blobs)} 条")
print(f"  最新一张主行里的字段 = {sorted(saver.serde.loads_typed(saver.storage['blob'][''][max(saver.storage['blob'][''])][0]))}")
print("  ↑ 没有 channel_values。值全在 blobs，主行只存版本指针 —— 这是'存全量 state'不爆内存的原因。")
print(f"  序列化器 = {type(saver.serde).__name__}（msgpack）；塞不可序列化对象会直接 TypeError")

print("\n  四种后端，官方怎么选：")
print("    InMemorySaver         进程内，重启即丢 —— 只用于测试/教学（本文件就是）")
print("    SqliteSaver           本地文件，开发期最省事：pip install langgraph-checkpoint-sqlite")
print("    PostgresSaver         生产默认：langgraph-checkpoint-postgres + psycopg[binary]，记得 setup() 建表")
print("    Async* 版本            FastAPI 之类的异步栈里用，别在 async 里调同步 saver")

print("\n  官方 Troubleshooting 里那三个坑（都是线上真会遇到的）：")
print("    · thread_id 超过 255 字符 → Postgres 直接报错。用 uuid7/hash，别拼长字符串")
print("    · MemorySaver 不跨进程 → 你在 A 进程存、B 进程读，永远读不到（不是代码 bug）")
print("    · checkpoint 无限增长 → 每轮都新增几张，长期跑要给保留策略（定期删旧 thread/旧 checkpoint）")
print("    · 父图看不到子图的状态 → 各管各的 ns，跨图请用 store / 共享 state")

# --------------------------------------------------------------------------
banner("J", "真跑一次：create_agent + checkpointer 是否同一套（--real）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
    print("  要验证的只有一句：create_agent 内部就是 graph.compile(checkpointer=saver)，")
    print("  三层主键、blobs 去重、from_checkpoint 不重放，和裸 StateGraph 完全同构。")
else:
    from _shared import get_model

    saver = InMemorySaver()
    agent = create_agent(get_model(), checkpointer=saver)
    cfg = {"configurable": {"thread_id": "real-6"}}
    agent.invoke({"messages": [{"role": "user", "content": "我叫小林，一句话回我。"}]}, cfg)
    result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？一句话。"}]}, cfg)
    print(f"  第二轮回答 = {short(result['messages'][-1].content, 60)}")

    snaps = list(agent.get_state_history(cfg))
    print(f"  两轮共 {len(snaps)} 张 checkpoint（step 从 -1 到 {snaps[0].metadata['step']}）")
    for snap in snaps:
        print(f"    step={snap.metadata['step']:>2} source={snap.metadata['source']:<6} "
              f"msgs={len(snap.values['messages'])} {msg_brief(snap.values['messages'])}")
    print(f"  saver 里的一级键 = {list(saver.storage)}，blobs = {len(saver.blobs)} 条")
