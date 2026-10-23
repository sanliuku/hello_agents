"""第 6 章 · 会话持久化：把 InMemorySaver 拆开看。

运行（项目根目录，脚本方式）：
    python langchain_learn/06_checkpointer.py
    python langchain_learn/06_checkpointer.py --real      # 最后一段才联网

分段：
    A  记忆有无对照：有 checkpointer / 没 checkpointer
    B  thread_id 隔离：换钥匙就失忆
    C  拆 saver.storage 的三层主键
    D  checkpoint 的 7 个字段（注意 channel_values 是空的）
    E  blobs：值按 (channel, version) 存，版本去重
    F  get_state_history 时间轴 + update_state 分叉
    G  铁证：从快照恢复时 reducer 不重放
    H  --real：create_agent(checkpointer=...) 等价验证

A–G 全程零 token、零网络（连 LLM 都不用，纯图 + 纯 saver）。
"""

from __future__ import annotations

import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------- 小工具

def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def real_enabled() -> bool:
    return "--real" in sys.argv


# ---------------------------------------------------------------- 图定义

class S(TypedDict):
    """最小 state：只有一个 messages，挂 add_messages reducer。"""

    messages: Annotated[list, add_messages]


def bot(state) -> dict:
    """假节点：不调模型，只数一下自己看到几条。它是"探针"。

    注意这里**故意不标 state 的类型**：langgraph 会把节点函数的注解当作输入 schema
    去和图的 schema 合并，下面 G 段要换一套 reducer 复用同一个 bot，
    标了 `state: S` 就会报 "Channel 'messages' already exists with a different type"。
    """
    n = len(state["messages"])
    return {"messages": [AIMessage(content=f"我看到 {n} 条，回第 {n} 条")]}


def build(saver=None):
    g = StateGraph(S)
    g.add_node("bot", bot)
    g.add_edge(START, "bot")
    g.add_edge("bot", END)
    return g.compile(checkpointer=saver) if saver else g.compile()


# ---------------------------------------------------------------- A
banner("A · 记忆有无对照：唯一变量是 checkpointer")

saver = InMemorySaver()
app = build(saver)
cfg = {"configurable": {"thread_id": "t1"}}

r1 = app.invoke({"messages": [HumanMessage("第1句")]}, cfg)
print(f"第一次 invoke: {len(r1['messages'])} 条")

r2 = app.invoke({"messages": [HumanMessage("第2句")]}, cfg)
print(f"第二次 invoke: {len(r2['messages'])} 条   ← 有记忆应该是 4")
print(f"最早那句还在吗: {r2['messages'][0].content!r}   ← 应该还是「第1句」")

print("\n-- 对照组：同一个图，但 compile() 不传 checkpointer --")
app_nomem = build()
n1 = app_nomem.invoke({"messages": [HumanMessage("第1句")]})
n2 = app_nomem.invoke({"messages": [HumanMessage("第2句")]})
print(f"第一次: {len(n1['messages'])} 条 / 第二次: {len(n2['messages'])} 条   ← 没有累积")
print(f"第二次的第一条: {n2['messages'][0].content!r}   ← 历史被覆盖了")

# ---------------------------------------------------------------- B
banner("B · thread_id 隔离：同一个 saver，换钥匙就失忆")

cfg_t2 = {"configurable": {"thread_id": "t2"}}
r3 = app.invoke({"messages": [HumanMessage("我是新会话")]}, cfg_t2)
print(f"thread_id=t2 第一次: {len(r3['messages'])} 条   ← t1 的历史完全看不到")
print(f"t1 现在有 {len(app.get_state(cfg).values['messages'])} 条，"
      f"t2 有 {len(app.get_state(cfg_t2).values['messages'])} 条   ← 两条独立的链")

print("\n-- 再换 saver 实例（thread_id 不变）--")
saver2 = InMemorySaver()
app2 = build(saver2)
r4 = app2.invoke({"messages": [HumanMessage("第三句")]}, cfg)   # 还是 t1
print(f"新 saver + 老 thread_id: {len(r4['messages'])} 条   ← 照样失忆")
print("结论：状态在 saver 里，thread_id 只是一把钥匙，不是保险箱。")

# ---------------------------------------------------------------- C
banner("C · 拆 saver.storage 的三层主键")

print(f"第一层 thread_id     : {list(saver.storage.keys())}")
print(f"第二层 checkpoint_ns : {list(saver.storage['t1'].keys())}   ← 根图是空字符串")
cps = saver.storage["t1"][""]
print(f"第三层 checkpoint_id : {len(cps)} 个")
for cid in sorted(cps):
    _ck, _md, parent = cps[cid]
    print(f"    {cid}  parent={parent}")

latest = max(cps.keys())
print(f"\n最新 checkpoint_id = max(keys()) = {latest}")
print("   ↑ 能这么取，是因为 uuid6 的字符串序 == 时间序（uuid4 就不行）")

# ---------------------------------------------------------------- D
banner("D · checkpoint 的 7 个字段（注意 channel_values）")

ck_bytes, md_bytes, _ = cps[latest]
ck = saver.serde.loads_typed(ck_bytes)
md = saver.serde.loads_typed(md_bytes)

for k, v in ck.items():
    print(f"  {k:18} = {str(v)[:90]}")

print(f"\n  channel_values 在存储里 = {ck.get('channel_values')}   ← 空的！")
print("  因为 put() 第一件事就是 c.pop('channel_values')，值全进了 blobs。")

print(f"\n  metadata.source = {md.get('source')!r}   step = {md.get('step')}")
print("  一次 invoke = 1 张 input（step=-1 起步）+ 每个 super-step 一张 loop。")
print(f"  这个单节点图两轮共 {len(cps)} 张：每轮 input + '进节点前' + '节点跑完' 各一张。")

# ---------------------------------------------------------------- E
banner("E · blobs：值按 (channel, version) 存，版本去重")

print(f"blobs 共 {len(saver.blobs)} 条：")
for key in sorted(saver.blobs, key=lambda k: (k[2], str(k[3]))):
    thread, ns, ch, ver = key
    typ, payload = saver.blobs[key]
    print(f"  ({thread!r}, {ns!r}, {ch!r}, v{ver}) -> type={typ}, {len(payload)} bytes")

print("\n  messages 有 4 个版本 = 4 个 blob，旧的不删 → 这就是时间旅行的物质基础")
print("  某个 channel 这一步没变，版本号就不涨、就不写新 blob，靠 channel_versions 指回旧 blob")

# ---------------------------------------------------------------- F
banner("F · get_state_history 时间轴 + update_state 分叉")

print("从新到旧：")
history = list(app.get_state_history(cfg))
for snap in history:
    n = len(snap.values.get("messages", []))
    print(f"  step={snap.metadata['step']:>2}  source={snap.metadata['source']:<7} "
          f"msgs={n}  next={snap.next}  id={snap.config['configurable'].get('checkpoint_id', '')[:13]}…")

print("\n-- 拿最近那张 input 检查点（还没合并新消息），改一句话，造一个 fork --")
target = next(s for s in history if s.metadata["source"] == "input")
print(f"选中: step={target.metadata['step']} msgs={len(target.values['messages'])}")
forked = app.update_state(
    target.config,
    {"messages": [HumanMessage("我改主意了，走另一条路")]},
    as_node="bot",
)
print(f"fork 出来的新 checkpoint: {forked['configurable']['checkpoint_id'][:13]}…")
st = app.get_state(forked)
print(f"新分支 msgs={len(st.values['messages'])}, 最后一条={st.values['messages'][-1].content!r}")
print(f"原分支还在: t1 最新 msgs={len(app.get_state(cfg).values['messages'])}")
print("  ↑ 分叉不改历史。source 会变成 'update'（手动 update_state 产生的）")

# ---------------------------------------------------------------- G
banner("G · 铁证：从快照恢复时 reducer 不重放")

CALLS: list[tuple[int, int]] = []      # 每次 reducer 调用记 (left_len, right_len)


def loud_reducer(left, right):
    """会喊的 reducer：每次被调用都打印，并记下 left/right 的长度。"""
    l_len = len(left) if left else 0
    r_len = len(right) if right else 0
    CALLS.append((l_len, r_len))
    print(f"      >>> reducer 被调用 #{len(CALLS)}: left={l_len} 条, right={r_len} 条")
    return (left or []) + (right or [])


class S2(TypedDict):
    messages: Annotated[list, loud_reducer]


def build2(saver):
    g = StateGraph(S2)
    g.add_node("bot", bot)
    g.add_edge(START, "bot")
    g.add_edge("bot", END)
    return g.compile(checkpointer=saver)


saver3 = InMemorySaver()
app3 = build2(saver3)
cfg3 = {"configurable": {"thread_id": "loud"}}

print("第一次 invoke：")
app3.invoke({"messages": [HumanMessage("A")]}, cfg3)
print(f"  第一次共 {len(CALLS)} 次调用：输入写入 1 次 + 节点写入 1 次 —— **每次写入过一次 reducer**")

before = len(CALLS)
print("\n第二次 invoke（重点看 left 是几条）：")
app3.invoke({"messages": [HumanMessage("B")]}, cfg3)
after = len(CALLS)

print(f"\n第二次只调用了 {after - before} 次 reducer，而且 left 是 2 条 ——")
print("  说明恢复时没有把历史 2 条重新 merge 一遍（那样会看到 left=0 然后 left=1）。")
print("  恢复 = from_checkpoint() 直接赋值 empty.value = checkpoint，operator 一次都不碰。")
assert after - before == 2, f"预期 2 次，实际 {after - before} 次"
print("  ✅ 断言通过：after - before == 2")

# ---------------------------------------------------------------- H
banner("H · --real：create_agent(checkpointer=...) 是不是同一套")

if not real_enabled():
    print("跳过（加 --real 才跑）。这一段会真发请求，qwen3 一次 20-40 秒。")
    print("\n但你不用跑也能记住结论：create_agent 内部就是 graph.compile(checkpointer=saver)，")
    print("三层主键、blobs 去重、from_checkpoint 不重放 —— 一模一样，没有任何额外魔法。")
else:
    from langchain.agents import create_agent

    from _shared import get_model

    saver4 = InMemorySaver()
    agent = create_agent(get_model(), tools=[], checkpointer=saver4)
    acfg = {"configurable": {"thread_id": "agent-1"}}

    a1 = agent.invoke({"messages": [HumanMessage("我叫小明，记住这个名字")]}, acfg)
    print(f"第一次: {len(a1['messages'])} 条")
    a2 = agent.invoke({"messages": [HumanMessage("我叫什么？")]}, acfg)
    print(f"第二次: {len(a2['messages'])} 条")
    print(f"回答: {a2['messages'][-1].content[:120]}")

    print("\nagent 场景下 storage 的结构（和裸 StateGraph 完全同构）：")
    for cid in sorted(saver4.storage["agent-1"][""]):
        _ck, _md, parent = saver4.storage["agent-1"][""][cid]
        m = saver4.serde.loads_typed(_md)
        print(f"  {cid[:13]}…  source={m.get('source'):<6} step={m.get('step'):>2} parent={str(parent)[:13]}")

    print(f"\nblobs 里 messages 的版本数: "
          f"{len([k for k in saver4.blobs if k[2] == 'messages'])}")

print("\n" + "=" * 68)
print("本章结论：")
print("  1. thread_id 是主键的第一层，checkpoint_ns 给子图，checkpoint_id 是 uuid6")
print("  2. checkpoint 只存版本指针，值在 blobs[(thread, ns, channel, version)]")
print("  3. 内容不变就不写新 blob → 版本去重 → 旧版本留着 → 时间旅行可行")
print("  4. 恢复时 from_checkpoint 直接赋值，reducer 不重放（所以不必幂等，但要满足结合律）")
print("  5. 状态在 saver 里。InMemorySaver 重启即丢，生产换 PostgresSaver")
print("=" * 68)
