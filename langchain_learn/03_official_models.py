"""第 3 章 · 官方对照：models 页的五件事，全都不发请求也能看

对应官方页：https://docs.langchain.com/oss/python/langchain/models
对应教材：docs/langchain-family/langchain_learn/04-ch03-model-layer.md

官方那一页的骨架：
  Basic usage      → A init_chat_model 与直连 Model 类，两条路
  Parameters       → B 六个标准参数，全部落在构造期
  Tool calling     → C bind_tools 之后 kwargs 里多了什么；以及"手动工具循环"长什么样
  Structured output→ D 三种 schema + method 三选一
  Advanced         → E 可配置模型 / 动态选模型 / 异常分类

A–E 段全部离线（零 token），F 段 --real 才联网。
"""

import inspect

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from _official_common import banner, real_enabled, short, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/03_official_models.py"


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


class Movie(BaseModel):
    """A movie with details."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


# --------------------------------------------------------------------------
banner("A", "两种取模型的写法（官方 Basic usage → Initialize a model）")
# --------------------------------------------------------------------------
# 写法一：init_chat_model —— 官方推荐入口，靠字符串选 provider
m1 = init_chat_model(
    model="gpt-5.5",
    model_provider="openai",
    api_key="sk-offline-placeholder",  # 只过构造校验，不发请求
)
# 写法二：直接 new 一个 provider 类 —— 能用到该 provider 的独有参数
m2 = ChatOpenAI(
    model="gpt-5.5",
    api_key="sk-offline-placeholder",
)

print(f"  init_chat_model(...)  -> {type(m1).__module__}.{type(m1).__name__}")
print(f"  ChatOpenAI(...)       -> {type(m2).__module__}.{type(m2).__name__}")
print("\n  两者造出来的是同一个类——init_chat_model 就是个分发表：")
print("  'provider:model' 的 provider 决定 import 哪个包、实例化哪个类。")
print("  什么时候必须用写法二：要传 provider 独有参数（如 use_responses_api）时。")

print("\n  provider 字符串与包名的对应关系（装错包时的第一现场）：")
pairs = [
    ("openai", "langchain-openai", "langchain[openai]"),
    ("anthropic", "langchain-anthropic", "langchain[anthropic]"),
    ("google_genai", "langchain-google-genai", "langchain[google-genai]"),
    ("bedrock_converse", "langchain-aws", "langchain[aws]"),
]
print(f"    {'provider':18s}{'要装的包':26s}extra")
print("    " + "-" * 62)
for provider, package, extra in pairs:
    print(f"    {provider:18s}{package:26s}{extra}")

# --------------------------------------------------------------------------
banner("B", "六个标准参数：全部在构造期生效（官方 Parameters 那张表）")
# --------------------------------------------------------------------------
model = get_model()
print("  本机 get_model() 造出来的实例，把关键属性读出来：")
for attr in ("model_name", "temperature", "max_tokens", "request_timeout", "max_retries"):
    value = getattr(model, attr, "<该版本没有这个属性>")
    print(f"    {attr:18s} = {value!r}")

print("\n  官方表里的六个参数，落到本机是这么写的：")
print("    temperature=0.7      → 随机性，越高越发散（本教材全程 0.0）")
print("    max_tokens=1000      → 上限，写小了会被硬截断")
print("    timeout=30           → 单次请求超时秒数（.env 的 LLM_TIMEOUT）")
print("    max_retries=6        → 官方默认 6，指数退避 + 抖动")
print("\n  连接容错的重试边界（这条最容易记反）：")
print("    会重试：网络错误 / 429 限流 / 5xx 服务端错误")
print("    不重试：401 认证失败 / 404 模型不存在 —— 重试一万次也不会好")

# --------------------------------------------------------------------------
banner("C", "bind_tools 之后 kwargs 里多了什么（官方 Tool calling 那节）")
# --------------------------------------------------------------------------
bound = model.bind_tools([get_weather])
print(f"  model.bind_tools([...]) 返回：{type(bound).__name__}")
bound_tools = bound.kwargs.get("tools", [])
print(f"  kwargs['tools'] 条数：{len(bound_tools)}")
print("  塞进请求的 schema（这就是模型看到的工具）：")
for schema in bound_tools:
    fn = schema.get("function", schema)
    print(f"    name        : {fn.get('name')}")
    print(f"    description : {short(fn.get('description'), 60)}")
    print(f"    parameters  : {fn.get('parameters')}")

print("\n  注意：bind_tools 只把 schema 塞进请求，**不执行**工具。")
print("  官方给的手动循环（没有 agent 时你要自己写）：")
print(
    """    messages = [{"role": "user", "content": "..."}]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call)
        messages.append(result)
    final = model_with_tools.invoke(messages)   # 再问一次，模型才给结论"""
)
print("  这 6 行就是 create_agent 内部那个 tools 节点的全部内容——第 1 章图里的循环。")

# --------------------------------------------------------------------------
banner("D", "结构化输出：三种 schema × 三种 method（官方 Structured output 那节）")
# --------------------------------------------------------------------------
print("  ① Pydantic（有校验，官方首选）")
structured = model.with_structured_output(Movie)
print(f"     with_structured_output(Movie) -> {type(structured).__name__}")
schema = Movie.model_json_schema()
print(f"     必填字段：{schema.get('required')}")
for name, spec in schema["properties"].items():
    print(f"       {name:10s} {spec.get('type'):8s} {spec.get('description', '')}")

print("\n  ② TypedDict（轻量，无运行时校验）/ ③ JSON Schema（控制力最强）")
print("     三者共用同一个出口，换的只是 method：")
for method in ("function_calling", "json_schema", "json_mode"):
    try:
        model.with_structured_output(Movie, method=method)
        print(f"       method={method:18s} 构造成功")
    except Exception as error:  # 某些 method 对 provider 有要求
        print(f"       method={method:18s} 构造失败：{type(error).__name__}")

print("\n  和 create_agent(response_format=...) 的关系：")
print("      with_structured_output 是**模型级**约束（一次调用就定型）")
print("      response_format 是**agent 级**约束（多轮工具调用之后再定型）")
print("      后者返回值在 result['structured_response']，第 7 章展开。")

# --------------------------------------------------------------------------
banner("E", "高级主题里最该先知道的两条")
# --------------------------------------------------------------------------
# 1) 可配置模型：不给 model 名，运行时用 config 指定
configurable = init_chat_model(temperature=0)
print(f"  ① init_chat_model(temperature=0) 不给模型名 -> {type(configurable).__name__}")
print("     这是'同一份 agent 代码，跑不同模型'的官方口子（configurable_fields）。")

# 2) 异常分类：is_retryable 决定你要不要重试
print("\n  ② 模型异常自带 is_retryable，别自己猜哪些该重试：")
import langchain_core.exceptions as lc_exceptions

available = [n for n in dir(lc_exceptions) if n.startswith("Model") or n == "ContextOverflowError"]
print(f"     本版本导出的模型异常：{available}")
for error_name in available:
    error_class = getattr(lc_exceptions, error_name)
    retryable = getattr(error_class, "is_retryable", "<基类没定义，看实例>")
    print(f"     {error_name:26s} is_retryable={retryable}")

# --------------------------------------------------------------------------
banner("F", "真跑一次：invoke 与 stream 的差别（--real）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
else:
    model = get_model()
    response = model.invoke("用一句话解释什么是 temperature。")
    print(f"  invoke  -> {type(response).__name__} / {short(response.text, 60)}")
    print(f"  usage   -> {response.usage_metadata}")

    chunks = list(model.stream("数到三。"))
    print(f"\n  stream  -> {len(chunks)} 个 {type(chunks[0]).__name__}")
    print(f"  拼接后  -> {short(sum(chunks[1:], chunks[0]).text, 60)}")
