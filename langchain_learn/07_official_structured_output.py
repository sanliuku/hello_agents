"""第 7 章 · 官方对照：structured-output 页的五层——自动选择、两条通道、四种 schema、
兜错策略、以及官方没写而本机会踩的四个坑

对应官方页：https://docs.langchain.com/oss/python/langchain/structured-output
对应教材：docs/langchain-family/langchain_learn/08-ch07-structured-output.md

官方那一页的骨架：
  Response format        → A  response_format 的四张脸，最后都变成 AutoStrategy
  Auto / Provider        → A  provider 判据：profile 说支持，或模型名在白名单
  Provider strategy      → B  请求体：bind_tools([业务工具], response_format=...)
  Tool calling strategy  → B  请求体：bind_tools([业务工具, 假工具], tool_choice="any")
  Custom tool message    → D  tool_message_content：默认文案长什么样
  Error handling         → E  六种 handle_errors 归一化成 (要不要重试, 回什么话)
                           F  两类错误现场：Multiple / Validation
  Schema 支持            → C  Pydantic / dataclass / TypedDict / JSON Schema 各返回什么
  官方没写 / 写错的      → G  裸 dict、随机 name、None 而不是缺 key、tool_choice 死循环

A–H 全部离线（假模型只记 bind 参数、脚本回话），零 token；I 段加 --real 才真发请求。

⚠️ 本文件 import 了 `factory` 里的三个私有函数（_supports_provider_strategy /
_handle_structured_output_error / _build_commands）。它们是为了**看见判定逻辑**，
生产代码不要这么写——生产只用 ToolStrategy / ProviderStrategy / 裸 schema。
"""

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.factory import (
    _build_commands,
    _handle_structured_output_error,
    _supports_provider_strategy,
    ModelResponse,
)
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
    _SchemaSpec,
)
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from _official_common import banner, real_enabled, short, skip_hint

SCRIPT = "langchain_learn/07_official_structured_output.py"


# ==========================================================================
# 四种 schema 形态：官方那张"返回什么"的表，C 段要用它们逐行对质
# ==========================================================================
class Person(BaseModel):
    """一个人。这段 docstring 会变成假工具的描述。"""

    name: str = Field(description="姓名")
    age: int = Field(description="年龄（岁）", ge=0)


@dataclass
class Order:
    """一个订单。"""

    order_id: str
    amount: float


class Movie(TypedDict):
    """一部电影。"""

    title: str
    year: int


RAW_SCHEMA = {
    "title": "Weather",
    "description": "一次天气查询结果",
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
}


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴 26℃"


class ScriptedModel(BaseChatModel):
    """零 token 假模型：记下 bind_tools 收到了什么，按剧本回话。

    这是本章最重要的工具——**两条通道的差别全在 bind_tools 的参数里**，
    不需要真模型也能看清楚。
    """

    model_name: str = "scripted"
    profile: dict | None = None
    script: list[AIMessage] = Field(default_factory=list)
    binds: list[tuple[list[str], dict]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        self.binds.append(([getattr(t, "name", str(t)) for t in tools], dict(kwargs)))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        message = self.script.pop(0) if self.script else AIMessage("好的")
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


# --------------------------------------------------------------------------
banner("A", "Response format：四种取值，最后都变成 AutoStrategy")
# --------------------------------------------------------------------------
# 官方签名只有四个分支：ToolStrategy / ProviderStrategy / type / None。
# 本机实测：前两个原样保留，type 和 dict 都被 AutoStrategy(schema=...) 包一层。
for label, value in (
    ("None", None),
    ("裸 Pydantic 类", Person),
    ("裸 JSON Schema dict", RAW_SCHEMA),
    ("ToolStrategy", ToolStrategy(Person)),
    ("ProviderStrategy", ProviderStrategy(Person)),
):
    if value is None:
        wrapped = None
    elif isinstance(value, (ToolStrategy, ProviderStrategy, AutoStrategy)):
        wrapped = value
    else:
        wrapped = AutoStrategy(schema=value)
    print(f"  {label:22s} → {type(wrapped).__name__}")

print("\n  「自动选择」不是第四个类，是 AutoStrategy 在**运行期**问这句话：")
print("  _supports_provider_strategy(model, tools) 为真 → 换 ProviderStrategy；否则用创建期")
print("  预建的 ToolStrategy。判定顺序见下面这张表（全部实测）：\n")

print(f"  {'profile':38s} {'模型名':16s} {'有工具':6s} 判定")
for profile, name, with_tool in (
    (None, "scripted", False),
    ({"structured_output": True}, "scripted", False),
    ({"structured_output": False}, "scripted", False),
    (None, "gpt-5.5", False),
    ({"structured_output": False}, "gpt-5.5", False),
    ({"structured_output": True}, "gemini-2.5-pro", True),
    ({"structured_output": True}, "gemini-3-pro", True),
):
    probe = ScriptedModel(model_name=name, profile=profile)
    verdict = _supports_provider_strategy(probe, tools=[get_weather] if with_tool else [])
    print(f"  {str(profile):38s} {name:16s} {str(with_tool):6s} {verdict}")

print("\n  三个官方没说的细节：")
print("    · profile 为 None 时还有**模型名白名单**兜底（factory.py 的 FALLBACK_MODELS_...）")
print("      gpt-4.1/4o/5.x、claude-sonnet-4-5/4-6、grok-4 都在里面 —— 所以名字对了也走 Provider")
print("    · profile 显式写 False 也**挡不住**白名单：代码是 `profile and profile.get(...)`，")
print("      假值直接落到名字匹配，gpt-5.5 + False 仍然判 True（实测那一行）")
print("    · gemini 2.x 一旦带工具就判 False —— 它不能同时用工具和原生结构化输出；3 系可以")

# --------------------------------------------------------------------------
banner("B", "两条通道的请求体：差别只在 bind_tools 的参数里")
# --------------------------------------------------------------------------
# 业务工具 + 结构化输出同时存在时，看模型到底收到了什么。
print("  【ToolStrategy】response_format=ToolStrategy(Person)")
spy = ScriptedModel(script=[tool_call("Person", {"name": "张三", "age": 30})])
agent = create_agent(spy, tools=[get_weather], response_format=ToolStrategy(Person))
result = agent.invoke({"messages": [{"role": "user", "content": "我叫张三，30 岁"}]})
names, kwargs = spy.binds[-1]
print(f"    tools      = {names}")
print(f"    tool_choice= {kwargs.get('tool_choice')!r}")
print(f"    structured_response = {result.get('structured_response')!r}")

print("\n  【ProviderStrategy】response_format=ProviderStrategy(Person)")
spy = ScriptedModel(
    profile={"structured_output": True},
    script=[AIMessage('{"name": "李四", "age": 25}')],
)
agent = create_agent(spy, tools=[get_weather], response_format=ProviderStrategy(Person))
result = agent.invoke({"messages": [{"role": "user", "content": "我叫李四，25 岁"}]})
names, kwargs = spy.binds[-1]
print(f"    tools      = {names}")
print(f"    tool_choice= {kwargs.get('tool_choice')!r}")
print(f"    response_format = {short(kwargs.get('response_format'), 150)}")
print(f"    structured_response = {result.get('structured_response')!r}")

print("\n  三句话总结这张对照：")
print("    · ToolStrategy：假工具**进** tools 列表，靠 tool_choice='any' 逼模型必须调一个")
print("    · ProviderStrategy：假工具**不进** tools，schema 走 response_format 参数")
print("    · 两条都用 bind_tools(...)，不是 model.bind(...) —— 第 7 章正文那句要按本机版本改")

# --------------------------------------------------------------------------
banner("C", "四种 schema 各返回什么：官方表格有两处和本机对不上")
# --------------------------------------------------------------------------
# 官方写：Pydantic → 实例；dataclass / TypedDict / JSON Schema → dict。
# 实测走的是同一个 _parse_with_schema：只有 json_schema 提前返回，其余都过 TypeAdapter。
cases: list[tuple[str, Any, dict]] = [
    ("Pydantic", Person, {"name": "王五", "age": 41}),
    ("dataclass", Order, {"order_id": "A-1", "amount": 9.9}),
    ("TypedDict", Movie, {"title": "沙丘", "year": 2021}),
    ("JSON Schema", RAW_SCHEMA, {"city": "北京", "celsius": 26}),
]
print(f"  {'schema':12s} {'schema_kind':12s} {'返回类型':12s} 返回值")
for label, schema, data in cases:
    binding = OutputToolBinding.from_schema_spec(_SchemaSpec(schema))
    parsed = binding.parse(data)
    print(f"  {label:12s} {binding.schema_kind:12s} {type(parsed).__name__:12s} {short(parsed, 40)}")

print("\n  官方说 dataclass 返回 dict —— **本机返回 Order 实例**（TypeAdapter 建了对象）。")
print("  TypedDict 没有可实例化的类，所以只能返回 dict；JSON Schema 更是第一行就 return data。")
print("  结论：要「拿到对象」就用 Pydantic 或 dataclass；要「拿到 dict」才用后两个。")

print("\n  附带一条：JSON Schema 的 name 来自 title，没有 title 就现编一个——")
print(f"    有 title  → {_SchemaSpec(RAW_SCHEMA).name!r}")
print(f"    无 title  → {_SchemaSpec({'type': 'object'}).name!r}（uuid 前 4 位，每次都不一样）")
print("  名字会进 tools[].name 和 response_format.json_schema.name，随机 = 无法复现/无法评测。")

# --------------------------------------------------------------------------
banner("D", "tool_message_content：不填时那句话是框架拼的")
# --------------------------------------------------------------------------
for content in (None, "已记录到会议纪要！"):
    spy = ScriptedModel(script=[tool_call("Person", {"name": "赵六", "age": 18})])
    agent = create_agent(spy, response_format=ToolStrategy(Person, tool_message_content=content))
    result = agent.invoke({"messages": [{"role": "user", "content": "我叫赵六"}]})
    tool_message = next(m for m in result["messages"] if type(m).__name__ == "ToolMessage")
    print(f"  tool_message_content={content!r}")
    print(f"    → ToolMessage(content={tool_message.content!r}, name={tool_message.name!r})")

print("\n  默认文案在 factory.py 里是 f'Returning structured response: {structured_response}'，")
print("  也就是把**解析后的对象**塞进 ToolMessage。想让它像正常人说话（或想省 token），自己填。")

# --------------------------------------------------------------------------
banner("E", "handle_errors 六种取值：归一化成一个二元组")
# --------------------------------------------------------------------------
# 官方只给了六种写法，没给判定顺序。本机归一化函数是
# _handle_structured_output_error(exception, response_format) -> (should_retry, message)
validation_error = StructuredOutputValidationError("Person", ValueError("age 不能为负"), AIMessage(""))
multiple_error = MultipleStructuredOutputsError(["Person", "Order"], AIMessage(""))


def custom_handler(error: Exception) -> str:
    return f"自定义处理：{type(error).__name__}"


print("  拿 StructuredOutputValidationError 去试六种写法：\n")
print(f"  {'handle_errors':34s} {'should_retry':13s} 回给模型的话")
for config in (True, "请重试，age 必须 >= 0", ValueError, (ValueError, KeyError), custom_handler, False):
    strategy = ToolStrategy(Person, handle_errors=config)
    should_retry, message = _handle_structured_output_error(validation_error, strategy)
    label = getattr(config, "__name__", str(config))
    print(f"  {label:34s} {str(should_retry):13s} {short(message, 46)}")

print("\n  三个反直觉（都会在真实项目里咬人）：")
print("    ① handle_errors=ValueError 对**校验失败无效** —— 抛出来的是")
print("       StructuredOutputValidationError，它的 MRO 是 (StructuredOutputError, Exception)，")
print(f"       issubclass(...ValueError) = {issubclass(StructuredOutputValidationError, ValueError)}。")
print("       要接住校验失败，写 StructuredOutputValidationError，或直接 True。")
print("    ② 传 (ValueError, KeyError) 元组时，只要没命中就 (False, '') —— 不重试、原样抛。")
print("    ③ ProviderStrategy 一律 (False, '')：provider 通道没有'把错误塞回去重试'这条路。")
print(f"       实测 ProviderStrategy + handle_errors=True → {_handle_structured_output_error(validation_error, ProviderStrategy(Person))}")
print("       （ProviderStrategy 根本没有 handle_errors 字段，想重试只能靠外层 middleware）")

print("\n  MultipleStructuredOutputsError 走的是同一张表：")
print(f"    handle_errors=True → {short(_handle_structured_output_error(multiple_error, ToolStrategy(Person))[1], 70)}")

# --------------------------------------------------------------------------
banner("F", "两类错误现场：两个假工具都调 vs 字段越界")
# --------------------------------------------------------------------------
print("  【Multiple】ToolStrategy(Person | Order)，剧本让模型第一轮两个都调、第二轮只调 Person：")
spy = ScriptedModel(
    script=[
        AIMessage(
            "",
            tool_calls=[
                {"name": "Person", "args": {"name": "张三", "age": 30}, "id": "c1"},
                {"name": "Order", "args": {"order_id": "A-1", "amount": 1.0}, "id": "c2"},
            ],
        ),
        tool_call("Person", {"name": "张三", "age": 30}, "c3"),
    ]
)
agent = create_agent(spy, response_format=ToolStrategy(Person | Order))
result = agent.invoke({"messages": [{"role": "user", "content": "既是人又是订单"}]})
tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
print(f"    第一轮那 2 条都是错误回执，内容：{short(tool_messages[0].content, 90)}")
print(f"    重试后 structured_response = {result.get('structured_response')!r}")
print(f"    轨迹：{' → '.join(type(m).__name__ for m in result['messages'])}")
print(f"    （共 {len(tool_messages)} 条 ToolMessage：第一轮 2 条错误 + 第二轮 1 条成功）")
print("    注意：重试不是'再问一遍'，是把错误当 ToolMessage 塞回去，让模型在同一条对话里改。")

print("\n  【Validation】Person 的 age 要求 ge=0，但模型给 -5：")
spy = ScriptedModel(script=[tool_call("Person", {"name": "张三", "age": -5})])
agent = create_agent(spy, response_format=ToolStrategy(Person, handle_errors=False))
try:
    agent.invoke({"messages": [{"role": "user", "content": "负数年龄"}]})
except StructuredOutputValidationError as error:
    print(f"    抛 StructuredOutputValidationError，.tool_name={error.tool_name!r}")
    print(f"    .source = {type(error.source).__name__}：{short(error.source, 60)}")
    print(f"    .ai_message 是那条坏消息：{short(error.ai_message, 50)}")
print("    handle_errors=False 不是'不校验'，是'校验失败就抛' —— 校验永远发生。")

# --------------------------------------------------------------------------
banner("G", "官方警告 vs 本机实测：四个会踩的坑")
# --------------------------------------------------------------------------
print("  ① 官方说『JSON Schema 必须用显式策略包裹』，本机 1.4.6 裸传也能跑：")
spy = ScriptedModel(script=[tool_call("Weather", {"city": "北京", "celsius": 26})])
agent = create_agent(spy, response_format=RAW_SCHEMA)
result = agent.invoke({"messages": [{"role": "user", "content": "北京天气"}]})
print(f"     裸 dict → structured_response={result.get('structured_response')!r}，"
      f"bind 到的工具={spy.binds[-1][0]}")
print("     能跑，但走的是 AutoStrategy→ToolStrategy，且拿到的是**未校验**的 dict。")
print("     官方那句话的意图是对的：想要校验和重试，就别裸传。")

print("\n  ② 没有产出结构化输出时，`structured_response` 是 None，不是『没有这个 key』：")
commands = _build_commands(
    ModelResponse(result=[AIMessage("我就不给结构化结果")], structured_response=None),
    has_structured_output=True,
)
update = commands[0].update
print(f"     这一轮写回 state 的键 = {sorted(update)}")
print(f"     update['structured_response'] = {update['structured_response']!r}")
print("     factory._build_commands 会显式写 None，为的是不让你读到**上一轮的旧值**。")
print("     所以用 result.get(...) 安全，但 `if 'structured_response' in result` 不能当判据。")

print("\n  ③ ToolStrategy 的 tool_choice='any' 是硬约束：模型不调工具，图会一直转：")
spy = ScriptedModel(script=[AIMessage("我就不调工具")] * 30)
agent = create_agent(spy, response_format=ToolStrategy(Person))
try:
    agent.invoke({"messages": [{"role": "user", "content": "给个结果"}]}, {"recursion_limit": 6})
except Exception as error:  # noqa: BLE001
    print(f"     {type(error).__name__}: {short(error, 70)}")
print("     模型不肯调工具 → model 节点没有出口 → 撞 recursion_limit 才停。")
print("     真模型很少这样，但**假模型/小模型/被 prompt 教坏的模型**会。测超时先看这里。")

print("\n  ④ 显式写 ProviderStrategy 不会回退：官方说『不支持就回退工具调用』，只适用于传裸 schema。")
spy = ScriptedModel(profile=None, script=[AIMessage('{"name": "张三", "age": 30}')])
agent = create_agent(spy, response_format=ProviderStrategy(Person))
result = agent.invoke({"messages": [{"role": "user", "content": "我叫张三"}]})
print(f"     profile=None 的模型 + 显式 Provider → bind kwargs={list(spy.binds[-1][1])}")
print(f"     structured_response={result.get('structured_response')!r}（没有 tool_choice，也没建假工具）")
print("     想要『不支持就自动退回工具调用』，就传裸 schema 让 AutoStrategy 去判。")

# --------------------------------------------------------------------------
banner("H", "选型口诀：什么时候用哪条通道")
# --------------------------------------------------------------------------
print("  先用裸 schema（AutoStrategy）：让框架按 profile/白名单自己挑，最省事。")
print("  要 strict 严格模式 → ProviderStrategy(schema, strict=True)，但只对支持的提供商生效。")
print("  要 Union / oneOf / 自定义报错话术 / 重试 → ToolStrategy，这是唯一支持这些的通道。")
print("  模型只支持工具调用（本地 qwen、多数开源）→ ToolStrategy，别指望 provider 通道。")
print("  schema 是纯 dict 且想校验 → 换成 Pydantic，否则 handle_errors 是摆设。")
print("  两条通道都拿不到结果时，先查 model.profile，再查 tool_choice/recursion_limit。")

# --------------------------------------------------------------------------
banner("I", "真跑一次：ToolStrategy + 自定义 tool_message_content（--real）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
    print("  要验证的只有一句：本地模型走的是 ToolStrategy，轨迹里能看到")
    print("  AIMessage(tool_calls=[Person]) → ToolMessage(你填的那句话) → structured_response。")
else:
    from _shared import dump_messages, get_model

    spy_agent = create_agent(
        get_model(),
        response_format=ToolStrategy(
            Person,
            tool_message_content="已记下这个人。",
            handle_errors="字段不合法，请按 schema 重填。",
        ),
    )
    try:
        real_result = spy_agent.invoke(
            {"messages": [{"role": "user", "content": "我叫孙七，今年 33 岁。"}]}
        )
    except Exception as error:  # noqa: BLE001
        print(f"  模型调用失败：{type(error).__name__}: {short(error, 120)}")
        print("  先确认 .env 的 LLM_BASE_URL 能连上（本地 ollama 要开着），这一段与 A–H 无关。")
    else:
        dump_messages(real_result["messages"])
        print(f"\n  structured_response = {real_result.get('structured_response')!r}")
        print(f"  类型                = {type(real_result.get('structured_response')).__name__}")

banner("收尾", "一句话记住这一章")
print("  response_format 只做两件事：把 schema 变成**工具**（ToolStrategy），")
print("  或者变成**请求参数**（ProviderStrategy）；AutoStrategy 只是把选择推迟到运行期。")
print("  剩下的全是边界：谁支持、返回什么类型、错了重不重试。")
