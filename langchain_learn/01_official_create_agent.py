"""第 1 章 · 官方对照：把官方 agents 页的"五层配置"逐层加上去，看图画了几遍

对应官方页：https://docs.langchain.com/oss/python/langchain/agents
对应教材：docs/langchain-family/langchain_learn/02-ch01-what-invoke-does.md

官方那一页的结构就是本章的目录：
  Core components: Model / Tools / System prompt / Structured output / Agent state
  Invoke / Stream
  Configure the harness: middleware 的各种"官方配方"

A 段：官方两行最小例 → 参数化六行，看清哪几个参数在动
B 段：五层配置逐层叠加，每加一层打印一次图的节点与边（离线，全是编译期）
C 段：官方 harness 配料（retry / PII / HITL）对图的影响——有的加节点，有的不加
D 段：真跑一次，看 `result["messages"][-1].content_blocks`（--real）

默认离线，加 --real 才联网。
"""

from pydantic import BaseModel, Field
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
)
from langchain.tools import tool

from _official_common import banner, real_enabled, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/01_official_create_agent.py"
MODEL = get_model()


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}：晴，26℃"


class Answer(BaseModel):
    """官方 structured output 那节的例子，原样搬过来。"""

    summary: str = Field(description="一句话结论")
    confidence: float = Field(description="0~1 的置信度")


class MyState(AgentState):
    """官方 agent state 那节：继承 AgentState 加自己的字段。"""

    user_id: str
    call_count: int


def graph_of(agent) -> tuple[list[str], list[str]]:
    """把编译后的图压成 (节点名列表, 边列表)，够看结构就行。"""
    graph = agent.get_graph()
    nodes = list(graph.nodes)
    edges = [f"{edge.source}->{edge.target}" for edge in graph.edges]
    return nodes, edges


def report(label: str, agent) -> None:
    nodes, edges = graph_of(agent)
    print(f"\n  {label}")
    print(f"    节点({len(nodes)}): {nodes}")
    print(f"    边({len(edges)}): {edges}")


# --------------------------------------------------------------------------
banner("A", "官方最小例 → 参数化")
# --------------------------------------------------------------------------
print(
    """  官方写两行就完事：
    agent = create_agent(model="openai:gpt-5.5", tools=tools)

  把默认值显式写出来，就是本教材一直在用的六行：
    agent = create_agent(
        model=...,            # 第 3 章：Model 层
        tools=[...],          # 第 4 章：Tool 层
        system_prompt="...",  # 拼成 SystemMessage 塞进 messages
        middleware=[...],     # 第 8/9 章
        response_format=...,  # 第 7 章
        state_schema=...,     # 第 5 章
    )"""
)

# --------------------------------------------------------------------------
banner("B", "五层配置逐层叠加，每层重编一次图")
# --------------------------------------------------------------------------
# 注意：下面全部是**编译期**行为，一个 token 都不花。
# create_agent 返回的是 CompiledStateGraph，图在建的时候就已经定型了。
report("① model + tools=[]（最裸）", create_agent(model=MODEL, tools=[]))
report("② + tools=[get_weather]", create_agent(model=MODEL, tools=[get_weather]))
report(
    "③ + system_prompt",
    create_agent(
        model=MODEL,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Be concise.",
    ),
)
report(
    "④ + response_format=Answer",
    create_agent(model=MODEL, tools=[get_weather], response_format=Answer),
)
report(
    "⑤ + state_schema=MyState",
    create_agent(model=MODEL, tools=[get_weather], state_schema=MyState),
)

print("\n  看出规律：②③⑤ 完全一样；④ 虽然节点数还是 4，边却从 4 条变 6 条——")
print("  model->model 是结构化输出失败后的自环，tools->__end__ 是工具跑完直接收口。")
print("  **节点数不变 ≠ 图不变**：先看节点，再看边。")

# --------------------------------------------------------------------------
banner("C", "官方 harness 配料：哪些会改图，哪些不改")
# --------------------------------------------------------------------------
report(
    "① middleware=[ModelRetryMiddleware(max_retries=3)]",
    create_agent(
        model=MODEL,
        tools=[get_weather],
        middleware=[ModelRetryMiddleware(max_retries=3)],
    ),
)
report(
    '② middleware=[PIIMiddleware("email")]',
    create_agent(
        model=MODEL,
        tools=[get_weather],
        middleware=[PIIMiddleware("email")],
    ),
)
report(
    "③ middleware=[HumanInTheLoopMiddleware(interrupt_on={'get_weather': True})]",
    create_agent(
        model=MODEL,
        tools=[get_weather],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
    ),
)

print("\n  实测结论（和我的直觉相反，值得抄进笔记本）：")
print("    ModelRetry : 不加节点，只加一条 model->model 的自环 —— '失败了再问一次'")
print("    PII        : 加两个节点 before_model / after_model —— 它拦的是进出 model 的数据")
print("    HITL       : 加一个节点 after_model —— 它要在'模型想调工具'之后插一脚")
print("  所以'middleware 一律包在节点里、图不变'是错的：")
print("  只有 wrap 型（retry）不改图，节点型（PII / HITL）会真的长出新节点。")
print("  第 8 章的钩子地图，第一根主线就从这条区别开始。")

# --------------------------------------------------------------------------
banner("D", "真跑一次：官方 overview 结尾那行 print")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
else:
    agent = create_agent(
        model=MODEL,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "杭州天气？"}]})
    print(f"  消息条数：{len(result['messages'])}")
    for index, message in enumerate(result["messages"]):
        print(f"    [{index}] {type(message).__name__}")
    print("\n  result['messages'][-1].content_blocks：")
    for block in result["messages"][-1].content_blocks:
        print(f"    {block}")
