"""05_state_reducer.py —— 第 5 章：State 与 Reducer

跑法（必须在项目根目录、用脚本方式）：
    python langchain_learn/05_state_reducer.py

分段：
    A  AgentState 到底有几个字段（读注解，不跑模型）
    B  add_messages 的五种情况（纯函数，零 token）
    C  _resolve_schemas：一个 state 变三份 schema
    D  有 reducer / 没 reducer：同一张图，两种行为
    E  agent 级铁证：忘写 add_messages 后，模型第二轮看到几条消息（假模型，零 token）
    F  自己写 reducer：累加与去重
    G  真请求（默认跳过；加 --real 才跑）

零 token 是设计目标：A–F 全部离线，只有 G 段联网。
"""

from __future__ import annotations

import operator
import sys
import uuid
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.factory import _resolve_schemas
from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import (
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from typing_extensions import NotRequired, Required, TypedDict

REAL = "--real" in sys.argv


def head(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)


# --------------------------------------------------------------------------- #
# A  AgentState 的注解
# --------------------------------------------------------------------------- #
def section_a_annotations() -> None:
    head("A  AgentState 的字段与元数据（源码：middleware/types.py:349）")
    from typing import get_type_hints

    hints = get_type_hints(AgentState, include_extras=True)
    for name, tp in hints.items():
        # Required[...] / NotRequired[...] 里再包 Annotated[...]，元数据在最后
        outer = getattr(tp, "__metadata__", None)
        args = getattr(tp, "__args__", ())
        reducer = None
        for a in args:
            md = getattr(a, "__metadata__", None)
            if md:
                reducer = md
        print(f"  {name:22s} 类型={tp!s:.70s}")
        if reducer:
            for r in reducer:
                print(f"  {'':22s} ↳ 元数据：{getattr(r, '__name__', r)}")
        if outer:
            print(f"  {'':22s} ↳ 字段级：{outer}")
    print("\n  读法：messages 的『类型』是 list，但挂了个 reducer（add_messages）；")
    print("  jump_to 挂的是 EphemeralValue + PrivateStateAttr；structured_response 挂 OmitFromInput。")


# --------------------------------------------------------------------------- #
# B  add_messages 五种情况
# --------------------------------------------------------------------------- #
def section_b_add_messages() -> None:
    head("B  add_messages 的五种情况（源码：langgraph/graph/message.py:64）")

    print("\n  B1 两个不同 id → 追加")
    left = [HumanMessage("你好", id="u1")]
    right = [AIMessage("在", id="a1")]
    merged = add_messages(left, right)
    print(f"    {[m.id for m in merged]}  len={len(merged)}")

    print("\n  B2 同一个 id → 原地替换，不是追加")
    merged = add_messages([HumanMessage("你好", id="u1")], [HumanMessage("你好（改过）", id="u1")])
    print(f"    len={len(merged)}  content={merged[0].content!r}")

    print("\n  B3 RemoveMessage(id=...) → 删掉那一条")
    merged = add_messages(
        [HumanMessage("a", id="1"), HumanMessage("b", id="2")],
        [RemoveMessage(id="1")],
    )
    print(f"    剩下 {[m.id for m in merged]}")

    print("\n  B4 删不存在的 id → 直接抛 ValueError（不是静默忽略）")
    try:
        add_messages([HumanMessage("a", id="1")], [RemoveMessage(id="404")])
    except ValueError as e:
        print(f"    ValueError: {e}")

    print("\n  B5 REMOVE_ALL_MESSAGES → 之后只保留它后面的那几条")
    merged = add_messages(
        [HumanMessage("旧1", id="1"), HumanMessage("旧2", id="2")],
        [RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage("清空后的第一条", id="new")],
    )
    print(f"    {[(m.id, m.content) for m in merged]}")
    print(f"    REMOVE_ALL_MESSAGES 的字面值是 {REMOVE_ALL_MESSAGES!r}")

    print("\n  B6 没给 id 的会被就地补 uuid —— 注意 left 也被改了（副作用！）")
    original = [HumanMessage("我原本没有 id")]
    print(f"    调用前 left[0].id = {original[0].id}")
    add_messages(original, [])
    print(f"    调用后 left[0].id = {original[0].id}")


# --------------------------------------------------------------------------- #
# C  一个 state 变三份 schema
# --------------------------------------------------------------------------- #
class TicketState(AgentState[Any]):
    """在 AgentState 基础上加字段——注意是**继承**，不是重新声明。"""

    thread_topic: str
    scratch: NotRequired[Annotated[str, PrivateStateAttr]]
    debug_blob: NotRequired[Annotated[dict, OmitFromOutput]]
    hidden_hint: NotRequired[Annotated[str, OmitFromInput]]


def section_c_three_schemas() -> None:
    head("C  _resolve_schemas：同一个 state，编译出三份 schema")
    state_schema, input_schema, output_schema = _resolve_schemas([TicketState])
    for name, sch in (
        ("StateSchema ", state_schema),
        ("InputSchema ", input_schema),
        ("OutputSchema", output_schema),
    ):
        fields = list(sch.__annotations__.keys())
        print(f"  {name} ({sch.__name__}): {fields}")

    print("\n  对照表：")
    print("    messages / jump_to / structured_response  只在 State，不在 Input/Output")
    print("    thread_topic                              三份都有（普通字段）")
    print("    scratch  (PrivateStateAttr)               只在 State —— 纯内部")
    print("    debug_blob (OmitFromOutput)               不进 Output（可入参）")
    print("    hidden_hint (OmitFromInput)               不进 Input（可输出）")

    print("\n  反例：不继承、直接重写 messages 的类型")
    class BrokenState(TypedDict):
        messages: list[AnyMessage]  # 少了 Annotated[..., add_messages]

    st, inp, outp = _resolve_schemas([BrokenState])
    print(f"    Broken 的字段 = {list(st.__annotations__.keys())}")
    print("    ↑ jump_to / structured_response 都没了：state_schema 是**替换**，不是追加")
    print("    ↑ 而且 messages 上没有任何元数据 → LangGraph 只会当普通 LastValue 字段")


# --------------------------------------------------------------------------- #
# D  有 reducer / 没 reducer
# --------------------------------------------------------------------------- #
class WithReducer(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class NoReducer(TypedDict):
    messages: list[AnyMessage]


def _build(schema: type) -> Any:
    """两个节点，各自往 messages 里塞一条，看最后剩几条。"""
    g = StateGraph(schema)
    g.add_node("first", lambda s: {"messages": [AIMessage("第一条", id="a1")]})
    g.add_node("second", lambda s: {"messages": [AIMessage("第二条", id="a2")]})
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    return g.compile()


def section_d_graph_level() -> None:
    head("D  同一张图，只差一个 Annotated：messages 是『追加』还是『覆盖』")

    out = _build(WithReducer).invoke({"messages": [HumanMessage("问题", id="u1")]})
    print(f"  有 reducer → {[(m.id, m.content) for m in out['messages']]}")

    out = _build(NoReducer).invoke({"messages": [HumanMessage("问题", id="u1")]})
    print(f"  没 reducer → {[(m.id, m.content) for m in out['messages']]}")
    print("\n  没有 reducer 的字段，LangGraph 给的是 LastValue 通道：最后一次写入赢，历史全丢。")
    print("  输入那条 HumanMessage 之所以也没了，是因为 first 节点整体覆盖了 messages。")


# --------------------------------------------------------------------------- #
# E  agent 级铁证（假模型，零 token）
# --------------------------------------------------------------------------- #
class RecordingFakeModel(BaseChatModel):
    """把每次收到的 messages 记下来，按脚本返回固定回复。零网络。"""

    responses: list[AIMessage]
    seen: list[list[AnyMessage]] = []
    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        self.seen.append(list(messages))
        msg = self.responses.pop(0) if self.responses else AIMessage("（脚本用完了）")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def get_weather(city: str) -> str:
    """查询某城市的天气。"""
    return f"{city}：晴，26℃"


def _script() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}],
        ),
        AIMessage("北京晴，26℃。"),
    ]


class BrokenBase(TypedDict):
    messages: list[AnyMessage]


class SubclassRedeclare(AgentState[Any]):
    """最像真人会犯的错：继承 AgentState，又顺手把 messages 重声明了一遍。"""

    messages: list[AnyMessage]  # 忘了 Annotated[..., add_messages]
    topic: str


def section_e_agent_level() -> None:
    head("E  agent 级：四种写法，模型第二轮看到几条消息")

    class SubclassOnly(AgentState[Any]):
        """只加字段、不碰 messages —— 这才是推荐的扩展写法。"""

        topic: str

    cases = (
        ("E1 默认 AgentState", {}),
        ("E2 继承 AgentState 只加字段（推荐）", {"state_schema": SubclassOnly}),
        ("E3 继承但重声明了 messages（丢 reducer）", {"state_schema": SubclassRedeclare}),
        ("E4 裸 TypedDict 完全不继承（丢 reducer + 丢字段）", {"state_schema": BrokenBase}),
    )

    for label, kwargs in cases:
        model = RecordingFakeModel(responses=_script())
        agent = create_agent(model, tools=[get_weather], **kwargs)
        agent.invoke({"messages": [HumanMessage("北京天气怎么样？")]})
        counts = [len(m) for m in model.seen]
        print(f"\n  {label}")
        print(f"    每次模型调用收到的消息条数 = {counts}")
        for i, msgs in enumerate(model.seen, 1):
            print(f"    第 {i} 次：{[type(m).__name__ for m in msgs]}")
            last = msgs[-1]
            print(f"            最后一条内容 = {str(last.content)[:36]!r}")

    print("\n  结论（三条，按代价从小到大排）：")
    print("    1. 想加字段 → 继承 AgentState，只写新字段。子类不重声明 messages 时，")
    print("       注解从父类继承下来，reducer 还在。E2 与 E1 一模一样。")
    print("    2. 一旦你**重声明** messages，就是你说了算：没写 Annotated[..., add_messages]，")
    print("       channel 退化成 LastValue，第二轮模型只看到一条 ToolMessage（E3）。")
    print("    3. 用裸 TypedDict 当 state_schema 更狠：jump_to / structured_response 也一起没了（E4）。")


# --------------------------------------------------------------------------- #
# F  自己写 reducer
# --------------------------------------------------------------------------- #
def merge_unique(left: list[str], right: list[str]) -> list[str]:
    """并集去重，保持先后顺序。"""
    out = list(left or [])
    for x in right or []:
        if x not in out:
            out.append(x)
    return out


class MetricsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tokens: Annotated[int, operator.add]
    visited: Annotated[list[str], merge_unique]


def section_f_custom_reducer() -> None:
    head("F  自己写 reducer：operator.add 累加、自定义函数做并集")
    g = StateGraph(MetricsState)
    g.add_node("a", lambda s: {"tokens": 100, "visited": ["a"], "messages": [AIMessage("a", id="1")]})
    g.add_node("b", lambda s: {"tokens": 50, "visited": ["a", "b"]})
    g.add_node("c", lambda s: {"tokens": 25, "visited": ["c"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    out = g.compile().invoke({"messages": [], "tokens": 0, "visited": []})
    print(f"  tokens  = {out['tokens']}   （100+50+25，operator.add）")
    print(f"  visited = {out['visited']}  （a 只出现一次，自定义并集）")
    print(f"  messages= {len(out['messages'])} 条")
    print("\n  reducer 的契约：接 (当前值, 本步写入值)，返回新值；纯函数，别原地改 left。")


# --------------------------------------------------------------------------- #
# G  真请求（可选）
# --------------------------------------------------------------------------- #
def section_g_real() -> None:
    head("G  真请求：默认 agent 两轮之后 messages 里有什么")
    if not REAL:
        print("  跳过（加 --real 才跑）。它证明的和 E1 完全一样，只是换了真模型。")
        return
    from _shared import get_model

    agent = create_agent(get_model(), tools=[get_weather])
    out = agent.invoke({"messages": [HumanMessage("北京天气怎么样？一句话。")]})
    for i, m in enumerate(out["messages"]):
        extra = ""
        if getattr(m, "tool_calls", None):
            extra = f" tool_calls={[tc['name'] for tc in m.tool_calls]}"
        if isinstance(m, ToolMessage):
            extra = f" tool_call_id={m.tool_call_id}"
        print(f"  [{i}] {type(m).__name__:12s} {str(m.content)[:40]!r}{extra}")


SECTIONS = [
    ("A", section_a_annotations),
    ("B", section_b_add_messages),
    ("C", section_c_three_schemas),
    ("D", section_d_graph_level),
    ("E", section_e_agent_level),
    ("F", section_f_custom_reducer),
    ("G", section_g_real),
]


def main() -> None:
    for name, fn in SECTIONS:
        try:
            fn()
        except Exception as e:  # 单段失败不影响后面的段
            import traceback

            print(f"\n  !! {name} 段出错：{type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
