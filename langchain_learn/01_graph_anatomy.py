"""第 1 章：把 create_agent 编出来的图打印出来，并对比'有工具 / 无工具'。

对应教材：docs/langchain-family/langchain_learn/02-ch01-what-invoke-does.md
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from _shared import dump_messages, get_model


@tool
def get_weather(city: str) -> str:
    """查询指定城市当前天气。用户询问天气时必须调用此工具。"""
    return f"{city}：晴，26℃，微风"


model = get_model()

# ---------- A. 有工具的 agent ----------
print("\n" + "#" * 62)
print("# A. tools=[get_weather]")
print("#" * 62)

agent_with_tools = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是天气助手，回答要简洁，用中文。",
    name="weather_agent",  # 会进 compile(name=...) 与 LangSmith metadata
)

print("\n--- 图结构（Mermaid 文本，可粘到 mermaid.live 看图） ---")
print(agent_with_tools.get_graph().draw_mermaid())

print("\n--- 图结构（ASCII，直接在终端看） ---")
print(agent_with_tools.get_graph(xray=True).draw_ascii())

question = {"messages": [{"role": "user", "content": "杭州今天天气怎么样？"}]}
print("\n>>> 提问：", question["messages"][0]["content"])
result = agent_with_tools.invoke(question)
dump_messages(result["messages"])

# ---------- B. 无工具的 agent ----------
print("\n" + "#" * 62)
print("# B. tools=[] —— 文档说：就只剩一个 model 节点，没有循环")
print("#" * 62)

agent_no_tools = create_agent(
    model=model,
    tools=[],
    system_prompt="你是天气助手，回答要简洁，用中文。",
    name="no_tools_agent",
)
print("\n--- 图结构（ASCII） ---")
print(agent_no_tools.get_graph(xray=True).draw_ascii())

result2 = agent_no_tools.invoke(question)
dump_messages(result2["messages"])
