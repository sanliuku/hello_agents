"""第 2 章 · 官方对照：messages 页的六节，逐节给一个最小可打印的例子

对应官方页：https://docs.langchain.com/oss/python/langchain/messages
对应教材：docs/langchain-family/langchain_learn/03-ch02-message-anatomy.md

官方那一页的节序就是这里的 A–F：
  Basic usage       → A 三种输入写法，最后都被归一成消息对象
  Message types     → B 四种消息手搓一遍，对着属性表看
  Message content   → C content_blocks 把各家原生格式抹平
  Tool message      → D artifact：给程序看，不给模型看
  Serialization     → E dumpd / load
  Use with models   → F 真跑一次（--real）

默认离线，加 --real 才联网。
"""

from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.load import dumpd, load
from langchain_core.messages import convert_to_messages

from _official_common import banner, real_enabled, skip_hint, short
from _shared import get_model

SCRIPT = "langchain_learn/02_official_messages.py"

# --------------------------------------------------------------------------
banner("A", "官方 Basic usage 的三种写法，最后都归一成同一种东西")
# --------------------------------------------------------------------------
# 官方页把输入分成三种：纯字符串 / 消息对象列表 / OpenAI 字典列表。
# 模型只认"消息对象"——字符串和字典在进模型之前会被转换。
raw_inputs = {
    "① 纯字符串": ["Write a haiku about spring"],  # 注意套了一层 list，下面解释为什么
    "② 消息对象": [SystemMessage("You are a poetry expert"), HumanMessage("写一句关于春天的诗")],
    "③ 字典格式": [
        {"role": "system", "content": "You are a poetry expert"},
        {"role": "user", "content": "写一句关于春天的诗"},
    ],
}
for label, payload in raw_inputs.items():
    messages = convert_to_messages(payload)
    kinds = [type(m).__name__ for m in messages]
    print(f"  {label:10s} convert_to_messages -> {kinds}")

print("\n  字典里的 role 映射：system→SystemMessage, user→HumanMessage,")
print("  assistant→AIMessage, tool→ToolMessage。记住这张表，报错时能省半小时。")

# 坑：convert_to_messages 收的是"序列"，裸字符串会被当成字符序列。
bare = "hi"
print(f"\n  陷阱：convert_to_messages({bare!r}) -> {len(convert_to_messages(bare))} 条消息！")
print("  它按字符遍历了。而 model.invoke('hi') 是合法的——包装发生在模型层，不是这里。")
print(f"  正确写法：convert_to_messages([{bare!r}]) -> "
      f"{[type(m).__name__ for m in convert_to_messages([bare])]}")

# --------------------------------------------------------------------------
banner("B", "四种消息手搓一遍（官方 Message types 那节）")
# --------------------------------------------------------------------------
ai_with_calls = AIMessage(
    content="",
    tool_calls=[
        {"name": "get_weather", "args": {"city": "杭州"}, "id": "call_123"}
    ],
)
messages = [
    SystemMessage("你是天气助手，只回答天气。"),
    HumanMessage(content="杭州天气怎么样？", name="alice", id="msg_001"),
    ai_with_calls,
    ToolMessage(content="杭州：晴，26℃", tool_call_id="call_123", name="get_weather"),
]

print(f"  {'#':3s}{'类型':16s}{'关键字段'}")
print("  " + "-" * 74)
for index, message in enumerate(messages):
    if isinstance(message, SystemMessage):
        detail = f"content={short(message.content, 24)}"
    elif isinstance(message, HumanMessage):
        detail = f"name={message.name}, id={message.id}, content={short(message.content, 20)}"
    elif isinstance(message, AIMessage):
        detail = f"tool_calls={message.tool_calls}"
    else:
        detail = f"tool_call_id={message.tool_call_id}, content={short(message.content, 18)}"
    print(f"  {index:<3d}{type(message).__name__:16s}{detail}")

print("\n  两条硬约束（第 4 章会让模型替你踩一遍）：")
print("    · ToolMessage.tool_call_id 必须等于 AIMessage.tool_calls[i]['id']")
print("    · 工具调用的返回条数必须和调用条数对齐，少一条图就转不动")

# --------------------------------------------------------------------------
banner("C", "content_blocks：把各家原生格式抹平（官方 Message content 那节）")
# --------------------------------------------------------------------------
# 官方给的两个例子：Anthropic 的 thinking、OpenAI 的 reasoning。
# 两者原生格式完全不同，content_blocks 之后都变成 type='reasoning'。
anthropic_style = AIMessage(
    content=[
        {"type": "thinking", "thinking": "先把问题拆成两步", "signature": "WaUjzkyp"},
        {"type": "text", "text": "答案是 42"},
    ],
    response_metadata={"model_provider": "anthropic"},
)
openai_style = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "先算加法"},
                {"type": "summary_text", "text": "再取整"},
            ],
        },
        {"type": "text", "text": "答案是 42", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"},
)

print("  Anthropic 原生 -> content_blocks：")
for block in anthropic_style.content_blocks:
    print(f"    {block}")
print("\n  OpenAI 原生 -> content_blocks：")
for block in openai_style.content_blocks:
    print(f"    {block}")

print("\n  注意两件事：")
print("    1. 一个是 thinking、一个是 reasoning+summary，抹平后都是 type='reasoning'")
print("    2. content 还是原来那坨原生结构，content_blocks 只是**另一个视图**，不覆盖它")
print(f"    3. 块数会变：anthropic {len(anthropic_style.content)} → "
      f"{len(anthropic_style.content_blocks)} 块；"
      f"openai {len(openai_style.content)} → {len(openai_style.content_blocks)} 块"
      "（summary 里两条被展开成两个 reasoning 块）")

# --------------------------------------------------------------------------
banner("D", "artifact：给程序看，不给模型看（官方 Tool message 那节）")
# --------------------------------------------------------------------------
tool_message = ToolMessage(
    content="It was the best of times, it was the worst of times.",  # 进模型
    tool_call_id="call_456",
    name="search_books",
    artifact={"document_id": "doc_123", "page": 0, "score": 0.93},  # 不进模型
)
print(f"  content  （进模型上下文）: {short(tool_message.content, 60)}")
print(f"  artifact （不进模型）    : {tool_message.artifact}")
print("\n  这就是第 4 章 content vs artifact 那条分界线的第一现场：")
print("  检索分数、原始 JSON、调试信息全塞 artifact，模型上下文一点不脏。")

# --------------------------------------------------------------------------
banner("E", "序列化：dumpd / load（官方 Serialization 那节）")
# --------------------------------------------------------------------------
original = HumanMessage("法国的首都是哪里？")
serialized = dumpd(original)
print(f"  dumpd -> dict，顶层键：{list(serialized)}")
print(f"  dumpd 里的 type：{serialized.get('type')} / id 列表长度：{len(serialized.get('id', []))}")
restored = load(serialized)
print(f"  load 回来：{type(restored).__name__} / content={restored.content}")
print("\n  官方那条警告要当真：load() 会实例化 Python 对象，")
print("  **只对你自己写出去的数据调用它**，别 load 不可信来源的内容。")

# --------------------------------------------------------------------------
banner("F", "真跑一次：AIMessage 的三个属性 + 流式 chunk 拼接（官方 Use with chat models）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
else:
    model = get_model()

    # F1：一次 invoke，拿真实的 AIMessage，看官方属性表里那四个字段
    response = model.invoke("用一句话说明什么是消息。")
    print("  F1 invoke 返回：")
    print(f"    type           : {type(response).__name__}")
    print(f"    text           : {short(response.text, 50)}")
    print(f"    content_blocks : {response.content_blocks}")
    print(f"    usage_metadata : {response.usage_metadata}")
    print(f"    response_metadata 键: {list(response.response_metadata)}")

    # F2：官方 Streaming and chunks 那段——chunk 相加能拼回完整消息
    print("\n  F2 stream 拼接（官方那段 for chunk in model.stream(...)）：")
    full = None
    count = 0
    for chunk in model.stream("数到五就好。"):
        count += 1
        full = chunk if full is None else full + chunk
    print(f"    chunk 个数      : {count}")
    print(f"    拼回来的类型    : {type(full).__name__}")
    print(f"    拼回来的 text   : {short(full.text, 50)}")
