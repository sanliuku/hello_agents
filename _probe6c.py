"""探针 3：replay / 子图 ns / 不可序列化对象。"""

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
    seen: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "probe"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(f"答{len(self.seen)}"))])


print("=" * 70)
print("R1  replay：从旧 checkpoint 再跑一次")
probe = Probe()
saver = InMemorySaver()
agent = create_agent(probe, checkpointer=saver)
cfg = {"configurable": {"thread_id": "rp"}}
agent.invoke({"messages": [{"role": "user", "content": "第一问"}]}, cfg)
agent.invoke({"messages": [{"role": "user", "content": "第二问"}]}, cfg)
hist = list(agent.get_state_history(cfg))
print("  历史（新→旧）：")
for s in hist:
    print(f"    step={s.metadata['step']:>2} source={s.metadata['source']:<6} msgs={len(s.values['messages'])} id={s.config['configurable']['checkpoint_id'][:13]}")
target = [s for s in hist if s.metadata["step"] == 1][0]
print(f"  拿 step=1 那个 checkpoint 当起点重跑（当时 msgs={len(target.values['messages'])}）")
old_cfg = {"configurable": {"thread_id": "rp", "checkpoint_id": target.config["configurable"]["checkpoint_id"]}}
again = agent.invoke(None, old_cfg)
print("  重跑后模型被调用次数 =", len(probe.seen), "（原先是 2）")
print("  重跑结果 msgs =", len(again["messages"]))
print("  重跑后历史条数 =", len(list(agent.get_state_history(cfg))))

print("\n" + "=" * 70)
print("R2  子图 checkpoint 的 ns")


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
saver2 = InMemorySaver()
pagent = pg.compile(checkpointer=saver2)
pagent.invoke({"messages": [HumanMessage("hi", id="h1")]}, {"configurable": {"thread_id": "ns1"}})
print("  storage 里的 (thread, ns) 组合：", list(saver2.storage["ns1"].keys()))
for ns in saver2.storage["ns1"]:
    print(f"    ns={ns!r:52s} checkpoints={len(saver2.storage['ns1'][ns])}")
print("  父图 get_state_history 只列 ns='' 的：", len(list(pagent.get_state_history({"configurable": {"thread_id": "ns1"}}))))

print("\n" + "=" * 70)
print("R3  往 state 里塞不可序列化的对象")


class Bad(TypedDict):
    messages: Annotated[list, add_messages]
    payload: Any


def put_bad(state):
    import threading

    return {"payload": threading.Lock()}


bg = StateGraph(Bad)
bg.add_node("bad", put_bad)
bg.add_edge(START, "bad")
bg.add_edge("bad", END)
try:
    bg.compile(checkpointer=InMemorySaver()).invoke({"messages": [], "payload": None}, {"configurable": {"thread_id": "bad"}})
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:180]}")

print("\n" + "=" * 70)
print("R4  update_state(as_node=...) 的效果")
probe = Probe()
agent = create_agent(probe, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "asn"}}
agent.invoke({"messages": [{"role": "user", "content": "原问题"}]}, cfg)
agent.update_state(cfg, {"messages": [HumanMessage("手工插一条", id="manual")]}, as_node="model")
snap = agent.get_state(cfg)
print("  update_state(as_node='model') 之后 next =", snap.next, "source =", snap.metadata["source"])
