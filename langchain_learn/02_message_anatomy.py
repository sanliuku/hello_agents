"""第 2 章：消息解剖。A–E 段纯本地（不花 token），F–G 段才真发请求。

对应教材：docs/langchain-family/langchain_learn/03-ch02-message-anatomy.md
运行：在项目根目录 python langchain_learn/02_message_anatomy.py
"""

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langchain_core.messages.tool import default_tool_parser
from langchain_core.tools import tool

from _shared import dump_messages, get_model

# ============ A. 一条 AIMessage 的真实字段 ============
print("=" * 64)
print("A. AIMessage 全字段")
print("=" * 64)
ai = AIMessage(
    content="",  # 注意：可以是空串
    tool_calls=[{"name": "get_weather", "args": {"city": "杭州"}, "id": "call_001"}],
)
print("type       :", ai.type)
print("tool_calls :", ai.tool_calls)
print("model_dump():")
print(json.dumps(ai.model_dump(), ensure_ascii=False, indent=2))
print("\nmessage_to_dict() 的骨架:", list(message_to_dict(ai).keys()))

# ============ B. 坏 JSON → invalid_tool_calls（纯本地） ============
print("\n" + "=" * 64)
print("B. default_tool_parser：坏 JSON 去哪了")
print("=" * 64)
raw = [
    {"id": "call_ok", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
    {"id": "call_bad", "function": {"name": "add", "arguments": '{a: 1, b: 2}'}},  # 非法 JSON
    {"id": "call_nofunction"},  # 没有 function 字段
]
ok_calls, bad_calls = default_tool_parser(raw)
print("tool_calls        :", ok_calls)
print("invalid_tool_calls:", bad_calls)
print("→ 第 1 条成功；第 2 条进 invalid（args 保留原始字符串，error=None）；第 3 条被静默丢弃")

# ============ C. additional_kwargs 里的原始格式会被自动升级 ============
print("\n" + "=" * 64)
print("C. _backwards_compat_tool_calls：只有 additional_kwargs 会怎样")
print("=" * 64)
legacy = AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'},
            }
        ]
    },
)
print("tool_calls（本该是空的，被自动填上）:", legacy.tool_calls)

# ============ D. content 两种形态 vs content_blocks ============
print("\n" + "=" * 64)
print("D. content 的形态与 .text / .content_blocks")
print("=" * 64)
m1 = HumanMessage(content="纯文本")
m2 = HumanMessage(
    content=[
        {"type": "text", "text": "请描述这张图："},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
)
for label, m in (("m1", m1), ("m2", m2)):
    print(f"\n{label}.content        =", m.content)
    print(f"{label}.text           =", repr(m.text))
    print(f"{label}.content_blocks =", json.dumps(m.content_blocks, ensure_ascii=False))
print(
    "→ 裸字符串被归一成 {'type':'text'} 块；provider 私有块先进 non_standard，"
    "再由 translator 尝试解开（解不开就留着）"
)

# ============ E. ToolMessage 的三个特殊字段 ============
print("\n" + "=" * 64)
print("E. ToolMessage：coerce / status / artifact")
print("=" * 64)
tm_num = ToolMessage(content=42, tool_call_id=123)  # 数字 content + 数字 id
print(
    "\ncontent(数字 42) →",
    repr(tm_num.content),
    "| tool_call_id(数字 123) →",
    repr(tm_num.tool_call_id),
)
tm_big = ToolMessage(
    content="stdout 前 200 字……（给模型看的摘要）",
    artifact={"stdout": "x" * 5000, "files": ["a.csv"]},  # 全量，不发给模型
    tool_call_id="call_001",
    status="error",
)
print("status     :", tm_big.status)
print("artifact 键:", list(tm_big.artifact.keys()), "| content 长度:", len(tm_big.content))
print("→ 模型只看 content；artifact 留给你的程序")


# ============ 以下开始真调模型 ============
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前天气。用户询问天气时必须调用此工具。"""
    return f"{city}：晴，26℃，微风"


model = get_model().bind_tools([get_weather])

# ============ F. 手工续写历史：不靠 create_agent ============
print("\n" + "=" * 64)
print("F. 自己拼 4 条消息，让模型续答")
print("=" * 64)

messages = [
    SystemMessage("你是天气助手，用中文简洁回答。"),
    HumanMessage("杭州今天天气怎么样？"),
]

first = model.invoke(messages)
print("第 1 次响应 tool_calls:", first.tool_calls)

call = first.tool_calls[0]
tool_result = get_weather.invoke(call["args"])  # 我们手动执行工具
print("手动执行结果:", tool_result)

messages.append(first)
messages.append(
    ToolMessage(content=tool_result, tool_call_id=call["id"])  # 铰链：id 必须一致
)

second = model.invoke(messages)
dump_messages(messages + [second])

# ============ G. 流式：args 是半截 JSON ============
print("\n" + "=" * 64)
print("G. 流式下的 tool_call_chunks")
print("=" * 64)
acc = None
for chunk in model.stream([HumanMessage("杭州天气怎么样？")]):
    for tc in chunk.tool_call_chunks:
        print("  chunk:", tc)
    acc = chunk if acc is None else acc + chunk
print("\n合并后的 tool_calls:", acc.tool_calls)
