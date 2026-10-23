"""第 4 章：Tool 层。A–E 段离线（不花 token），F–G 段才真发请求。

对应教材：docs/langchain-family/langchain_learn/05-ch04-tool-layer.md
运行：在项目根目录 python langchain_learn/04_tool_layer.py
"""

import json
from typing import Annotated

from langchain_core.tools import (
    InjectedToolCallId,
    StructuredTool,  # noqa: F401  仅用于对照类型，不直接实例化
    ToolException,
    tool,
)
from pydantic import BaseModel, Field

# ============ A. @tool 的四种写法，看它到底给你什么 ============
print("=" * 64)
print("A. 四种写法 → 什么类型")
print("=" * 64)


@tool
def t_bare(city: str) -> str:
    """查天气。"""
    return f"{city} 晴"


@tool("renamed_tool")
def t_named(city: str) -> str:
    """查天气。"""
    return f"{city} 晴"


@tool(parse_docstring=True)
def t_doc(city: str) -> str:
    """查天气。

    Args:
        city: 城市名，例如 杭州。
    """
    return f"{city} 晴"


@tool(infer_schema=False)
def t_raw(payload: str) -> str:
    """吃什么给什么。"""
    return payload


for t in (t_bare, t_named, t_doc, t_raw):
    print(f"{t.name:<16} {type(t).__name__:<16} args={list(t.args.keys())}")

print("\n注意最后一行：infer_schema=False 退化成 Tool，args 为空 ——")
print("模型没法知道要传什么，这就是'模型不调我的工具'的第一种死法。")

# 看清 description 从哪来（§2.3 的四级 fallback）
print("\n各工具的 description：")
for t in (t_bare, t_named, t_doc, t_raw):
    print(f"  {t.name:<16} {t.description!r}")

# ============ B. 一个函数 → 三份 schema ============
print("\n" + "=" * 64)
print("B. 三份 schema 的差异")
print("=" * 64)


class WeatherIn(BaseModel):
    """查天气的入参。"""

    city: str = Field(description="城市名")
    days: int = Field(default=1, description="预报天数")


@tool("weather", args_schema=WeatherIn)
def weather(city: str, days: int = 1) -> str:
    """查询指定城市天气。"""
    return f"{city} 未来 {days} 天：晴"


print("1) .args（最瘦，只有 properties）:")
print(json.dumps(weather.args, ensure_ascii=False, indent=2))

print("\n2) tool_call_schema（发给模型的）:")
_tcs = weather.tool_call_schema
print(
    json.dumps(_tcs.model_json_schema(), ensure_ascii=False, indent=2)
    if hasattr(_tcs, "model_json_schema")
    else json.dumps(_tcs, ensure_ascii=False, indent=2)
)

print("\n3) get_input_schema()（校验用的）:")
print(
    json.dumps(
        weather.get_input_schema().model_json_schema(), ensure_ascii=False, indent=2
    )
)

print("\n4) 直接 .run 单参数 → 裸字符串也能过（_parse_input 的 first-field 包装）:")
print("   ", weather.run("杭州"))

print("\n5) 多传一个没声明的字段（extra='forbid'）：")
try:
    weather.invoke(
        {"type": "tool_call", "name": "weather", "args": {"city": "杭州", "x": 1}, "id": "c0"}
    )
except Exception as e:  # noqa: BLE001
    print("   ", type(e).__name__, ":", str(e).splitlines()[0])

# ============ C. 注入参数：模型看不到，但你能拿到 ============
print("\n" + "=" * 64)
print("C. InjectedToolCallId：schema 里没有它")
print("=" * 64)


@tool
def echo_with_id(text: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """回显文本，并把 tool_call_id 一起带回。"""
    return f"{text} | call_id={tool_call_id}"


print(
    "tool_call_schema 字段:",
    list(echo_with_id.tool_call_schema.model_json_schema()["properties"].keys()),
)
print(
    "get_input_schema 字段:",
    list(echo_with_id.get_input_schema().model_json_schema()["properties"].keys()),
)

print("\n手动 invoke（会炸，因为它需要完整 ToolCall）:")
try:
    echo_with_id.invoke({"text": "hi"})
except Exception as e:  # noqa: BLE001
    print("   ", type(e).__name__, ":", str(e)[:120], "...")

print("\n用完整 ToolCall invoke（成功）:")
_call = {
    "type": "tool_call",
    "name": "echo_with_id",
    "args": {"text": "hi"},
    "id": "call_abc",
}
print("   ", echo_with_id.invoke(_call))

# ============ D. 三种失败，三种命运 ============
print("\n" + "=" * 64)
print("D. ToolException vs ValueError")
print("=" * 64)


@tool
def fail_tool_exc(x: int) -> str:
    """抛 ToolException。"""
    raise ToolException(f"业务失败：x={x} 不合法")


@tool
def fail_value(x: int) -> str:
    """抛 ValueError。"""
    raise ValueError(f"内部错误：x={x}")


@tool
def fail_handled() -> str:
    """已配错误处理。"""
    raise ToolException("真实原因：DB 连接池耗尽（不该给模型看）")


# langchain_core 1.x 的 @tool 已不接受 handle_tool_error，只能在实例上设置
fail_handled.handle_tool_error = "工具暂时不可用，请稍后重试"


for _name, _t, _inp in [
    ("ToolException + 无 handler", fail_tool_exc, {"x": 1}),
    ("ValueError    + 无 handler", fail_value, {"x": 1}),
    ("ToolException + handler=字符串", fail_handled, {}),
]:
    try:
        print(f"[{_name}] → {_t.invoke(_inp)!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[{_name}] → 抛出 {type(e).__name__}: {e}")

print("\n参数类型错误（ValidationError）→ handle_validation_error 未设 → 抛出:")
try:
    fail_tool_exc.invoke({"x": "不是整数"})
except Exception as e:  # noqa: BLE001
    print("   ", type(e).__name__, str(e).splitlines()[0], "...")

# ============ E. content vs artifact ============
print("\n" + "=" * 64)
print("E. response_format='content_and_artifact'")
print("=" * 64)


@tool(response_format="content_and_artifact")
def run_tests(path: str) -> tuple[str, dict]:
    """跑测试，返回摘要 + 完整日志。"""
    full_log = "line1\nline2\nFAILED tests/test_api.py:42"
    return f"{path}：2000 行，FAILED 3 处，首条 tests/test_api.py:42", {
        "stdout": full_log,
        "failed": ["tests/test_api.py:42"],
    }


_msg = run_tests.invoke(
    {
        "type": "tool_call",
        "name": "run_tests",
        "args": {"path": "tests/"},
        "id": "call_t1",
    }
)
print("type     :", type(_msg).__name__)
print("content  :", _msg.content)
print("artifact :", json.dumps(_msg.artifact, ensure_ascii=False))
print("模型只看到 content；artifact 留在对象上给下游用。")

# 对照：response_format='content' 时返回 dict 会被 _stringify
@tool
def return_dict() -> dict:
    """返回一个 dict，看它变成什么。"""
    return {"a": 1, "中文": "值"}


_dmsg = return_dict.invoke(
    {"type": "tool_call", "name": "return_dict", "args": {}, "id": "c1"}
)
print("\n返回 dict 的工具 →", type(_dmsg.content).__name__, repr(_dmsg.content))
print("（是 str 不是 dict；中文没被 ensure_ascii 转义成 \\uXXXX）")

# ============ F. 真实模型：它为什么（不）调工具 ============
print("\n" + "=" * 64)
print("F. 真实模型：description 决定它调不调")
print("=" * 64)

from _shared import dump_messages, get_model  # noqa: E402


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单状态。仅当用户明确给出订单号时使用，返回状态与金额。"""
    return f"订单 {order_id}：已发货，金额 299.00"


@tool
def vague_tool(q: str) -> str:
    """处理数据。"""
    return "ok"


model = get_model()
bound = model.bind_tools([query_order, vague_tool])

print("发给模型的 tools payload（只看 name/description/parameters）：")
for _t in bound.kwargs.get("tools", []):
    _fn = _t.get("function", _t)
    _params = _fn.get("parameters", {}).get("properties", {})
    print("  -", _fn.get("name"), "|", str(_fn.get("description"))[:40])
    print("      params:", list(_params.keys()))

try:
    r1 = bound.invoke("帮我查下订单 A10086 到哪了")
    print("\n问『查订单 A10086』→ tool_calls:")
    print("   ", r1.tool_calls)

    r2 = bound.invoke("帮我处理一下数据")
    print("\n问『处理一下数据』→ tool_calls:")
    print("   ", r2.tool_calls)
    print("（观察：模糊描述的工具容易被乱选，或两个都不选）")

    dump_messages([r1, r2])

    # ============ G. 完整往返：工具真的被执行 ============
    print("\n" + "=" * 64)
    print("G. 完整往返：工具真的被执行")
    print("=" * 64)

    first = bound.invoke("订单 A10086 状态？")
    if first.tool_calls:
        _call = first.tool_calls[0]
        _table = {"query_order": query_order, "vague_tool": vague_tool}
        result = _table[_call["name"]].invoke(_call)
        dump_messages([first, result])
        final = model.invoke([first, result, ("user", "用一句话总结")])
        print("\n最终回答:", final.content)
    else:
        print("模型没调工具，first.content =", first.content)
except Exception as e:  # noqa: BLE001  .env 没配好时不让整章崩掉
    print("\n[跳过 F/G] 模型调用失败：", type(e).__name__, str(e)[:200])
    print("先完成第 0 章的 .env 配置。")

# ============ 收尾自检 ============
print("\n" + "=" * 64)
print("本章要点回看")
print("=" * 64)
print("tool_call_schema 剔除注入参数，get_input_schema 保留它")
print("FILTERED_ARGS =", ("run_manager", "callbacks"))
print("ToolException 可拦截（handle_tool_error），其它异常永远抛出")
print("description 优先级：显式 > docstring > args_schema.__doc__ > dict['description']")
