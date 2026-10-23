"""第 0 章 · 官方对照：Agent = Model + Harness

对应官方页：https://docs.langchain.com/oss/python/langchain/overview
对应教材：docs/langchain-family/langchain_learn/01-ch00-environment-and-map.md

这章要钉住一句话：**Agent = Model + Harness**。
Model 是那个会说话的东西；Harness 是套在它外面的循环——prompt、tools、middleware。

A 段：官方 overview 的 6 行最小例，原文贴出来（只打印，不执行）
B 段：`"provider:model"` 这串字符串到底怎么被拆开（含 ollama:qwen3:latest 的坑）
C 段：create_agent 的参数分两堆——哪些是 harness，哪些是运行时
D 段：LangChain / LangGraph / Deep Agents 的定位判据 + 本机装没装
E 段：真跑一次（--real）

默认离线，加 --real 才联网。
"""

import importlib.util
import inspect

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from _official_common import banner, real_enabled, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/00_official_agent_equation.py"

# --------------------------------------------------------------------------
banner("A", "官方 overview 的最小例（原文，不执行）")
# --------------------------------------------------------------------------
# 这段一个字都不要改着看。它说明三件事：
#   1. model 收一个字符串，不是对象——字符串比对象晚绑定，配 .env 更方便
#   2. tools 收一个**裸函数**，不需要 @tool 装饰器（官方在 agents 页说了：
#      "pass any Python callable, LangChain tool, or tool dict"）
#   3. 返回值是一个 dict，最后一条消息的 .content_blocks 才是答案
print(
    '''
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
'''
)

# --------------------------------------------------------------------------
banner("B", '"provider:model" 是怎么被拆开的')
# --------------------------------------------------------------------------
# 官方页里每个示例第一行都是 model="provider:model"。
# 它长得像魔法，其实就是按**第一个**冒号切两半。
identifiers = [
    "openai:gpt-5.5",
    "google_genai:gemini-3.6-flash",
    "anthropic:claude-sonnet-4-5",
    "ollama:qwen3:latest",
]
for ident in identifiers:
    provider, _, model_name = ident.partition(":")  # partition 只切第一个冒号
    print(f"  {ident:34s} -> provider={provider:14s} model={model_name}")

print("\n  注意最后一行：ollama 的模型名自己带冒号（qwen3:latest，冒号后是 tag）。")
print("  所以必须用 partition/split(':', 1)，用 split(':') 会把它切成三段。")
print("  这也是本机 .env 里不写前缀、只写 LLM_MODEL_ID 的原因：")
print("  我们走的不是 ollama provider，而是 ollama 的 **OpenAI 兼容口**。")

# 下面这一步是离线的：init_chat_model 只解析、只构造客户端，不发请求。
print("\n  init_chat_model 真解析一次（离线，只看它造出了什么类）：")
model = init_chat_model(
    model="gpt-5.5",
    model_provider="openai",
    api_key="sk-offline-placeholder",  # 只为通过构造校验，不发请求
    base_url="http://localhost:11434/v1",
)
print(f"    init_chat_model(...) -> {type(model).__name__}, model_name={model.model_name}")

# --------------------------------------------------------------------------
banner("C", "create_agent 的参数分两堆")
# --------------------------------------------------------------------------
# 官方说法："configure the basics directly with model=, tools=, system_prompt=.
#           For more advanced capabilities, extend the harness with middleware."
# 换句话说：前三个 + middleware 是 harness；其余是"跑起来以后的事"。
HARNESS = {"model", "tools", "system_prompt", "middleware"}
params = list(inspect.signature(create_agent).parameters)

signature = inspect.signature(create_agent)

def show(name: str) -> None:
    default = signature.parameters[name].default
    shown = "<必填>" if default is inspect.Parameter.empty else repr(default)
    print(f"    {name:18s} {shown}")

print(f"  create_agent 一共 {len(params)} 个参数\n")
print("  【harness 层】决定 agent 怎么想、能做什么")
for name in params:
    if name in HARNESS:
        show(name)
print("\n  【运行时层】决定这次跑起来怎么存、怎么停、怎么切模型")
for name in params:
    if name not in HARNESS:
        show(name)

# --------------------------------------------------------------------------
banner("D", "三框架的定位判据（官方 overview 那张选择表）")
# --------------------------------------------------------------------------
rows = [
    ("Deep Agents", "要开箱即用：上下文压缩、虚拟文件系统、子 agent 都已装好", "先试这个"),
    ("LangChain", "要自己配 harness：model / tools / prompt / middleware 逐项拧", "本教材主战场"),
    ("LangGraph", "要确定性流程和 agent 混编、要自己控状态机", "隔壁那本"),
    ("LangSmith", "不管用哪个框架，都要 trace / 调试 / 评测", "上线前必备"),
]
print(f"  {'框架':14s} {'什么时候选它':46s} 备注")
print("  " + "-" * 78)
for name, when, note in rows:
    print(f"  {name:14s} {when:46s} {note}")

print("\n  本机装了什么（决定你能不能跑官方原样示例）：")
for pkg in ("langchain", "langgraph", "deepagents", "langchain_ollama"):
    spec = importlib.util.find_spec(pkg)
    print(f"    {pkg:18s} {'已装' if spec else '未装'}")

# --------------------------------------------------------------------------
banner("E", "真跑一次：把官方例子的 model 换成本机模型")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
else:
    # 官方写 model="openai:gpt-5.5"；本机换成 get_model() 造出来的实例。
    # 这一处替换就是"官方示例移植到本地"的全部工作量。
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(
        model=get_model(),
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "杭州天气怎么样？"}]}
    )
    print(f"  消息条数：{len(result['messages'])}（含 Human / AI / Tool）")
    print(f"  最后一条类型：{type(result['messages'][-1]).__name__}")
    print("  content_blocks（官方 overview 最后那行 print）：")
    for block in result["messages"][-1].content_blocks:
        print(f"    {block}")
