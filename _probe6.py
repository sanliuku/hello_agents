"""探针：ch06 会话持久化要用到的真实数据。跑完即删。"""

from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import Field
from typing_extensions import TypedDict


class Probe(BaseChatModel):
    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "probe"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        msg = self.script.pop(0) if self.script else AIMessage(f"答{len(self.seen)}：见{len(messages)}条")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def get_weather(city: str) -> str:
    """查天气。"""
    return f"{city}：晴 26℃"


print("=" * 70)
print("E1/E2  一次 invoke（带一次工具调用）产生几个 checkpoint")
probe = Probe(script=[
    AIMessage("", tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "c1"}]),
    AIMessage("北京晴，26℃。"),
])
saver = InMemorySaver()
agent = create_agent(probe, tools=[get_weather], checkpointer=saver)
cfg = {"configurable": {"thread_id": "t1"}}
out = agent.invoke({"messages": [{"role": "user", "content": "北京天气？"}]}, cfg)

h = list(agent.get_state_history(cfg))
print(f"  checkpoint 总数 = {len(h)}")
for s in h:
    print(f"    step={s.metadata['step']:>3} source={s.metadata['source']:<7} next={s.next} "
          f"messages={len(s.values.get('messages', []))} id={s.config['configurable']['checkpoint_id'][:13]}…")

snap = agent.get_state(cfg)
print("\n  最新 StateSnapshot 字段：")
for f in ("values", "next", "config", "metadata", "created_at", "parent_config", "tasks"):
    v = getattr(snap, f, "<无此字段>")
    print(f"    {f:13s} = {str(v)[:110]}")
print("  config 全量 =", snap.config)

print("\n" + "=" * 70)
print("E3  checkpointer 里到底存了什么")
tup = saver.get_tuple(cfg)
print("  CheckpointTuple 字段 =", [f for f in dir(tup) if not f.startswith('_')][:12])
cp = tup.checkpoint
print("  checkpoint keys =", list(cp.keys()))
print("  channel_values 里的 key =", list(cp['channel_values'].keys()))
print("  messages 条数 =", len(cp['channel_values']['messages']))
print("  channel_versions =", cp['channel_versions'])
print("  versions_seen =", cp['versions_seen'])
print("  id/ts/v =", cp['id'][:16], "…", cp['ts'], cp['v'])

print("\n" + "=" * 70)
print("E4  reducer 在恢复时会不会再跑一遍")
counts = {"n": 0}


def counting_add(left, right):
    counts["n"] += 1
    left = list(left or [])
    for m in right or []:
        if getattr(m, "id", None) in [x.id for x in left]:
            continue
        left.append(m)
    return left


class S(TypedDict):
    messages: Annotated[list, counting_add]


g = StateGraph(S)
g.add_node("noop", lambda s: {"messages": [AIMessage("节点写的", id="a1")]})
g.add_edge(START, "noop")
g.add_edge("noop", END)
graph = g.compile(checkpointer=InMemorySaver())
c = {"configurable": {"thread_id": "r1"}}
print("  第一次 invoke…")
graph.invoke({"messages": [HumanMessage("u1", id="u1")]}, c)
print(f"  第一次后 reducer 调用次数 = {counts['n']}")
print("  第二次 invoke（同 thread，只传新消息）…")
graph.invoke({"messages": [HumanMessage("u2", id="u2")]}, c)
print(f"  第二次后 reducer 调用次数 = {counts['n']}")
print("  现在 update_state 一次（写入也要过 reducer）…")
graph.update_state(c, {"messages": [HumanMessage("u3", id="u3")]})
print(f"  update_state 后 = {counts['n']}")

print("\n" + "=" * 70)
print("E5  时间旅行：删掉的消息在旧 checkpoint 里还在吗")
probe = Probe()
saver = InMemorySaver()
agent = create_agent(probe, checkpointer=saver)
c2 = {"configurable": {"thread_id": "tt"}}
for t in range(3):
    agent.invoke({"messages": [{"role": "user", "content": f"第{t}轮"}]}, c2)
hist = list(agent.get_state_history(c2))
print(f"  checkpoint 数 = {len(hist)}")
for s in hist[:6]:
    print(f"    step={s.metadata['step']:>2} source={s.metadata['source']:<6} messages={len(s.values['messages'])}")

# 取最老的一条
oldest = hist[-1]
print("  最老的 values =", [(type(m).__name__, m.content) for m in oldest.values["messages"]])

print("\n" + "=" * 70)
print("E6  update_state：改掉最后一条 AI 消息")
probe = Probe()
saver = InMemorySaver()
agent = create_agent(probe, checkpointer=saver)
c3 = {"configurable": {"thread_id": "upd"}}
agent.invoke({"messages": [{"role": "user", "content": "原问题"}]}, c3)
before = agent.get_state(c3)
print("  改前最后一条 =", before.values["messages"][-1].content)
last = before.values["messages"][-1]
agent.update_state(c3, {"messages": [AIMessage("我改过的回答", id=last.id)]})
after = agent.get_state(c3)
print("  改后最后一条 =", after.values["messages"][-1].content)
print("  改后的 metadata =", after.metadata)
print("  value id 相同? =", after.values["messages"][-1].id == last.id)
print("  history 条数 =", len(list(agent.get_state_history(c3))))
agent.invoke({"messages": [{"role": "user", "content": "接着问"}]}, c3)
print("  续跑后模型看到的历史 =", [str(m.content)[:12] for m in probe.seen[-1]])

print("\n" + "=" * 70)
print("E7  把全量历史再传一遍会怎样")
probe = Probe()
agent = create_agent(probe, checkpointer=InMemorySaver())
c4 = {"configurable": {"thread_id": "dup"}}
r1 = agent.invoke({"messages": [{"role": "user", "content": "我叫 Bob"}]}, c4)
old = list(r1["messages"])
r2 = agent.invoke({"messages": [*old, HumanMessage("我叫什么？")]}, c4)
print("  第二次传全量 → state 条数 =", len(r2["messages"]), "（本来应该是 4）")
print("  内容 =", [(type(m).__name__[:4], str(m.content)[:10]) for m in r2["messages"]])

print("\n" + "=" * 70)
print("E8  durability 参数")
probe = Probe()
agent = create_agent(probe, checkpointer=InMemorySaver())
c5 = {"configurable": {"thread_id": "dur"}}
for mode in ("sync", "async", "exit"):
    try:
        agent.invoke({"messages": [{"role": "user", "content": f"mode={mode}"}]}, c5, durability=mode)
        print(f"  durability={mode:<6} 接受，history={len(list(agent.get_state_history(c5)))}")
    except TypeError as e:
        print(f"  durability={mode:<6} 不接受：{e}")

print("\n" + "=" * 70)
print("E9  子图的 checkpoint namespace")


class Sub(TypedDict):
    messages: Annotated[list, add_messages]


def sub_node(state):
    return {"messages": [AIMessage("子图产出", id="sub1")]}


sub = StateGraph(Sub)
sub.add_node("inner", sub_node)
sub.add_edge(START, "inner")
sub.add_edge("inner", END)
subgraph = sub.compile()


class Parent(TypedDict):
    messages: Annotated[list, add_messages]


pg = StateGraph(Parent)
pg.add_node("child", subgraph)
pg.add_edge(START, "child")
pg.add_edge("child", END)
pagent = pg.compile(checkpointer=InMemorySaver())
c6 = {"configurable": {"thread_id": "ns"}}
pagent.invoke({"messages": [HumanMessage("hi", id="h1")]}, c6)
for s in pagent.get_state_history(c6):
    print(f"  ns={s.config['configurable']['checkpoint_ns']!r:12s} step={s.metadata['step']:>2} "
          f"source={s.metadata['source']:<6} messages={len(s.values['messages'])}")
