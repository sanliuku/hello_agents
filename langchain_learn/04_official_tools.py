"""第 4 章 · 官方对照：tools 页的四层——怎么定义、怎么拿上下文、怎么返回、怎么兜错

对应官方页：https://docs.langchain.com/oss/python/langchain/tools
对应教材：docs/langchain-family/langchain_learn/05-ch04-tool-layer.md

官方那一页的骨架：
  Create tools        → A @tool 三件套（docstring / 改名 / 改描述）
  Advanced schema     → B 类型提示够不够；不够就上 Pydantic args_schema
  Reserved arguments  → C runtime / config 是保留名，永远不进 schema
  Access context      → D ToolRuntime 的八个部件，一次看清
  Tool execution      → E 四种返回值 + return_direct 短路
  Error handling      → F wrap_tool_call：把异常翻译成 ToolMessage

A–F 全部离线（连工具都是手工 invoke，不花 token），G 段 --real 才让 agent 真调。
"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain.tools.tool_node import ToolCallRequest

from _official_common import banner, real_enabled, short, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/04_official_tools.py"

# --------------------------------------------------------------------------
banner("A", "@tool 三件套：docstring、改名、改描述")
# --------------------------------------------------------------------------


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


@tool("web_search")  # 改名：模型看到的是 web_search，不是 search
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))


for candidate in (search_database, search, calc):
    print(f"  name        : {candidate.name}")
    print(f"  description : {short(candidate.description, 70)}")
    print(f"  schema 参数 : {list(candidate.tool_call_schema.model_json_schema()['properties'])}")
    print()

print("  三件事看清：")
print("    1. docstring 就是 description —— 写不好模型就不会调它（这是最容易忽略的一处）")
print("    2. @tool('名字') 覆盖函数名；@tool('名字', description=...) 再覆盖描述")
print("    3. 带默认值的参数（limit=10）在 schema 里不是 required")

# --------------------------------------------------------------------------
banner("B", "类型提示不够用时：args_schema（官方 Advanced schema definition）")
# --------------------------------------------------------------------------


class WeatherInput(BaseModel):
    """Input for weather queries."""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature unit preference"
    )
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


schema = get_weather.tool_call_schema.model_json_schema()
print("  args_schema 生成出来的 schema：")
for name, spec in schema["properties"].items():
    enum = f" / 枚举={spec['enum']}" if "enum" in spec else ""
    print(f"    {name:18s} {spec.get('type', '?'):8s} 默认={spec.get('default')!r}{enum}")
print(f"\n  必填：{schema.get('required')}")
print("\n  为什么值得多写一个 Pydantic 类：")
print("    · Literal 会变成 enum，模型只能在合法值里选，省掉一轮纠错")
print("    · Field(description=...) 会进 schema，等于给每个参数写了 prompt")
print("    · 参数一多，纯类型提示的可读性会崩，Pydantic 能顺手加校验")

# --------------------------------------------------------------------------
banner("C", "保留参数名：runtime / config 永远不进 schema（官方 Reserved argument names）")
# --------------------------------------------------------------------------


@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    from langchain.messages import HumanMessage

    for message in reversed(runtime.state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return "No user messages found"


@tool
def noisy_report(topic: str) -> str:
    """普通工具，没有任何注入参数。"""
    return f"report about {topic}"


print(f"  noisy_report 的 schema 参数  : {list(noisy_report.tool_call_schema.model_json_schema()['properties'])}")
print(f"  get_last_user_message 的 schema: {list(get_last_user_message.tool_call_schema.model_json_schema()['properties'])}")
print("\n  第二个工具**一个业务参数都没有**，但它是合法工具：")
print("  runtime 是保留名，框架会自己塞进去，模型看不见也填不了。")
print("  这就是官方那句 'Runtime is invisible to the model' 的字面含义。")

print("\n  ToolRuntime 的八个部件（本机版本实测）：")
runtime_fields = [
    ("state", "短期记忆：messages + 自定义字段", "runtime.state['messages']"),
    ("context", "每次运行的不可变配置", "runtime.context.user_id"),
    ("store", "长期记忆，跨会话", "runtime.store.get(...)"),
    ("stream_writer", "工具执行中发进度", "runtime.stream_writer('...')"),
    ("execution_info", "thread_id / run_id / 重试次数", "runtime.execution_info.thread_id"),
    ("server_info", "LangGraph Server 元数据（本地为 None）", "runtime.server_info"),
    ("config", "RunnableConfig：回调、标签", "runtime.config"),
    ("tool_call_id", "本次调用的唯一 ID", "runtime.tool_call_id"),
]
for field, purpose, example in runtime_fields:
    print(f"    {field:16s}{purpose}")
    print(f"    {'':16s}→ {example}")

print("\n  官方把旧的 InjectedState / InjectedStore / InjectedToolCallId 标成待迁移，")
print("  新代码一律用 ToolRuntime —— 一个参数拿全部，别背五个类名。")

# --------------------------------------------------------------------------
banner("D", "四种返回值：直接 invoke 看 ToolMessage 长什么样")
# --------------------------------------------------------------------------


@tool
def return_string(city: str) -> str:
    """返回字符串。"""
    return f"It is currently sunny in {city}."


@tool
def return_object(city: str) -> dict:
    """返回结构化对象。"""
    return {"city": city, "temperature_c": 22, "conditions": "sunny"}


@tool
def return_multimodal() -> list[dict]:
    """返回多模态内容块。"""
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": "https://example.com/page.png"},
    ]


cases = [
    ("str", return_string, {"city": "杭州"}),
    ("dict", return_object, {"city": "杭州"}),
    ("list[dict]", return_multimodal, {}),
]
# 注意：直接 tool.invoke(参数dict) 拿回来的是**原始返回值**；
# 想拿到 ToolMessage（框架真正放进 messages 的那个对象），要传一个 ToolCall：
#   {"name": ..., "args": ..., "id": ..., "type": "tool_call"}
# 官方"手动工具循环"里写的就是 tool.invoke(tool_call)，原因就在这里。
for label, candidate, payload in cases:
    tool_call = {
        "name": candidate.name,
        "args": payload,
        "id": f"call_{label}",
        "type": "tool_call",
    }
    result = candidate.invoke(tool_call)
    print(f"  {label:10s} -> {type(result).__name__}.content 类型 = {type(result.content).__name__}")
    print(f"               content = {short(result.content, 90)}")
    print(f"               tool_call_id = {result.tool_call_id}")

print("\n  str 和 dict 都会进模型上下文（dict 被序列化成 JSON 字符串）；")
print("  多模态返回的 content 是**块列表**，走 content_blocks 读（第 2 章 C 段）。")

print("\n  还有第五种返回值：Command（工具自己改 state）。官方两处硬约束：")
print("    · update 里必须带 ToolMessage，且 tool_call_id=runtime.tool_call_id")
print("    · 忘了带 → ToolNode 直接 ValueError（不是模型报错，是框架报错）")

# --------------------------------------------------------------------------
banner("E", "return_direct：让工具输出直接变成最终答案")
# --------------------------------------------------------------------------


@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    return f"Order {order_id} is shipped and will arrive in 2 days."


print(f"  fetch_order_status.return_direct = {fetch_order_status.return_direct}")
print("\n  它做的事：工具跑完不进 model 节点，直接 __end__。")
print("  官方给的三条边界（很容易记成'只要标了就短路'）：")
print("    · 一批并行调用里，**每个**工具都 return_direct 才短路")
print("    · 只要有一个不是，就带着全部 ToolMessage 回到模型")
print("    · 和 Command 组合时，ToolMessage 仍然要自己补，不会自动生成")

# --------------------------------------------------------------------------
banner("F", "错误兜底：把异常翻译成 ToolMessage（官方 Error handling 那节）")
# --------------------------------------------------------------------------


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Convert tool exceptions into ToolMessages the model can handle."""
    try:
        return handler(request)
    except Exception as error:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({error})",
            tool_call_id=request.tool_call["id"],
        )


print("  handle_tool_errors 被 @wrap_tool_call 装饰后，仍是函数（只是挂了钩子标记），")
print("  可以直接塞进 create_agent(middleware=[handle_tool_errors])。")
print("\n  为什么必须在 middleware 里兜，而不是在工具里 try/except：")
print("    · 工具里兜住 → 模型收到 '出错了' 但不知道错在哪，容易原地重试")
print("    · middleware 里兜 → 统一格式、统一日志，且对**所有**工具生效")
print("    · 工具自己 raise ToolException 也是合法的，但那是'我想让模型知道'的语义")

# --------------------------------------------------------------------------
banner("G", "真跑一次：看 tool_call_id 怎么把请求和结果对上（--real）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
else:
    from langchain.agents import create_agent

    agent = create_agent(
        model=get_model(),
        tools=[get_weather, get_last_user_message],
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "杭州现在多少度？"}]})

    for index, message in enumerate(result["messages"]):
        print(f"    [{index}] {type(message).__name__}")
        calls = getattr(message, "tool_calls", None)
        if calls:
            for call in calls:
                print(f"          tool_call id={call['id']} name={call['name']} args={call['args']}")
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            print(f"          tool_call_id={call_id} content={short(message.content, 50)}")

    print("\n  上面 id 一一对应，这就是第 2 章 B 段那条硬约束的真实样子。")
