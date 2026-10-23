"""探针 2：durability / 重复传同一条消息 / saver 内部存储 / 子图 ns。"""

from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.messages import AIMessage, HumanMessage
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


print("=" * 70)
print("D1  durability 各写了哪些 checkpoint")
for mode in ("sync", "async", "exit"):
    saver = InMemorySaver()
    agent = create_agent(Probe(), checkpointer=saver)
    cfg = {"configurable": {"thread_id": f"d-{mode}"}}
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, cfg, durability=mode)
    hist = list(agent.get_state_history(cfg))
    print(f"  {mode:<6} checkpoint={len(hist)}  " +
          " ".join(f"[step{s.metadata['step']}/{s.metadata['source']}/next={s.next}]" for s in reversed(hist)))

print("\n" + "=" * 70)
print("D2  同一条无 id 消息传两次")
saver = InMemorySaver()
agent = create_agent(Probe(), checkpointer=saver)
cfg = {"configurable": {"thread_id": "dup2"}}
same = {"role": "user", "content": "我叫 Bob"}
r1 = agent.invoke({"messages": [same]}, cfg)
r2 = agent.invoke({"messages": [same]}, cfg)
print("  第二次后 state =", [(type(m).__name__[:4], str(m.content)[:8]) for m in r2["messages"]])
print("  条数 =", len(r2["messages"]))

print("\n" + "=" * 70)
print("D3  InMemorySaver 内部长什么样")
saver = InMemorySaver()
agent = create_agent(Probe(), checkpointer=saver)
agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "s1"}})
agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "s2"}})
print("  saver 类型 =", type(saver).__name__)
print("  存储结构 key 示例：")
for k in list(saver.storage.keys()):
    print("   ", k, "→ 该 checkpoint 的 messages 条数 =", len(saver.storage[k][0]["channel_values"]["messages"]))
print("  writes 表 key 示例：", list(saver.writes.keys())[:2])

print("\n" + "=" * 70)
print("D4  子图 checkpoint 落在哪个 ns")


class Sub(TypedDict):
    messages: Annotated[list, add_messages]


sg = StateGraph(Sub)
sg.add_node("inner", lambda s: {"messages": [AIMessage("子图产出", id="sub1")]})
sg.add_edge(START, "inner")
sg.add_edge("inner", END)
sub = sg.compile()


class Parent(TypedDict):
    messages: Annotated[list, add_messages]


pg = StateGraph(Parent)
pg.add_node("child", sub)
pg.add_edge(START, "child")
pg.add_edge("child", END)
saver = InMemorySaver()
pagent = pg.compile(checkpointer=saver)
pagent.invoke({"messages": [HumanMessage("hi", id="h1")]}, {"configurable": {"thread_id": "ns1"}})
for k in saver.storage.keys():
    print("   ", k)

print("\n" + "=" * 70)
print("D5  最老 checkpoint 的 values")
saver = InMemorySaver()
agent = create_agent(Probe(), checkpointer=saver)
cfg = {"configurable": {"thread_id": "old"}}
for t in range(2):
    agent.invoke({"messages": [{"role": "user", "content": f"第{t}轮"}]}, cfg)
hist = list(agent.get_state_history(cfg))
for s in hist:
    print(f"    step={s.metadata['step']:>2} source={s.metadata['source']:<6} msgs={len(s.values['messages'])} "
          f"content={[str(m.content)[:6] for m in s.values['messages']]}")

print("\n" + "=" * 70)
print("D6  checkpoint_id 是不是时间有序")
ids = [s.config["configurable"]["checkpoint_id"] for s in hist]
for i in ids:
    print("   ", i)
print("   前 8 位递增?", [i[:8] for i in ids])
