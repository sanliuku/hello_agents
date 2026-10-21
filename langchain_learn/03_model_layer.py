"""第 3 章：Model 层。A–E 段离线，F 段才真发请求。

对应教材：docs/langchain-family/langchain_learn/04-ch03-model-layer.md
运行：在项目根目录 python langchain_learn/03_model_layer.py
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

# ============ A. 字符串怎么被解析 ============
print("=" * 64)
print("A. _parse_model：字符串 → (模型名, provider)")
print("=" * 64)
from langchain.chat_models.base import (
    _BUILTIN_PROVIDERS,
    _attempt_infer_model_provider,
    _parse_model,
)

print("注册表里的 provider 数量:", len(_BUILTIN_PROVIDERS))
for raw, prov in [
    ("openai:gpt-4o", None),
    ("gpt-4o", None),
    ("claude-sonnet-4-5", None),
    ("deepseek-chat", None),
    ("openai:gpt-4o", "azure_openai"),  # ← 坑 1：前缀不剥
]:
    print(f"  _parse_model({raw!r}, {prov!r}) -> {_parse_model(raw, prov)}")

try:
    _parse_model("my-private-llm", None)
except ValueError as e:
    print("  猜不出来时的报错:", str(e).split("\n")[0])


# ============ B. 固定模型 vs 可配置模型 ============
print("\n" + "=" * 64)
print("B. init_chat_model 是个分流器")
print("=" * 64)
from langchain.chat_models import init_chat_model

fixed = init_chat_model("openai:gpt-4o", api_key="sk-dummy", temperature=0)
print("  给了 model      ->", type(fixed).__name__)

cfg = init_chat_model(temperature=0)
print("  没给 model      ->", type(cfg).__name__)
print("  对它 bind_tools ->", type(cfg.bind_tools([{"name": "x", "parameters": {}}])).__name__)
print("  仍然是排队状态，没有真的绑定")


# ============ C. 一个工具变成三段 JSON ============
print("\n" + "=" * 64)
print("C. convert_to_openai_tool：你写的 → 模型看的")
print("=" * 64)


@tool
def get_weather(city: str, unit: str = "c") -> str:
    """查询指定城市当前天气。

    Args:
        city: 城市名，例如 杭州。
        unit: 温度单位，c 或 f。
    """
    return f"{city} 26{unit}"


print(json.dumps(convert_to_openai_tool(get_weather), ensure_ascii=False, indent=2))
print("\n注意三件事：没有 title（被 _rm_titles 删了）、type 是 function、parameters 里只有 city 是 required")


# ============ D. 自己写一个假模型，把链路跑通 ============
print("\n" + "=" * 64)
print("D. 最小 ChatModel：只实现 _generate 和 _llm_type")
print("=" * 64)

SEEN = []  # 记录每次 _generate 收到的 kwargs，用来证明 bind_tools 的透传


class EchoChatModel(BaseChatModel):
    """不发网络请求，把收到的消息和 kwargs 打回来。"""

    reply: str = "ok"

    @property
    def _llm_type(self) -> str:
        return "echo"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "echo", "reply": self.reply}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        SEEN.append(
            {
                "n_messages": len(messages),
                "types": [m.type for m in messages],
                "kwargs": dict(kwargs),
            }
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)


echo = EchoChatModel()
out = echo.invoke("你好")
print("  直接 invoke        ->", type(out).__name__, repr(out.content))
print("  _generate 收到     ->", SEEN[-1])

bound = echo.bind_tools([get_weather], tool_choice="get_weather")
print("  bind_tools 返回类型 ->", type(bound).__name__)
bound.invoke([HumanMessage(content="杭州天气？")])
print("  绑定后 kwargs 里多了 ->", list(SEEN[-1]["kwargs"].keys()))
print("  tools[0].function.name ->", SEEN[-1]["kwargs"]["tools"][0]["function"]["name"])
print("  tool_choice           ->", SEEN[-1]["kwargs"]["tool_choice"])


# ============ E. 缓存键：加工具 = 换缓存 ============
print("\n" + "=" * 64)
print("E. _get_llm_string：缓存键里有什么")
print("=" * 64)
k1 = echo._get_llm_string(stop=None)
k2 = echo._get_llm_string(stop=None, tools=[convert_to_openai_tool(get_weather)])
print("  无工具键长:", len(k1))
print("  有工具键长:", len(k2))
print("  两者相同吗:", k1 == k2, "  ← 不同就说明加了工具缓存必然失效")


# ============ F. 真请求（需要 .env） ============
print("\n" + "=" * 64)
print("F. 真模型：元数据从哪来")
print("=" * 64)
from _shared import get_model

real = get_model()
msg = real.invoke("用一句话说明什么是 LangChain。")
print("  type          :", type(msg).__name__)
print("  content       :", (msg.content or "")[:60])
print("  usage_metadata:", msg.usage_metadata)
print(
    "  response_meta :",
    {
        k: msg.response_metadata.get(k)
        for k in ("model_name", "model_provider", "finish_reason", "system_fingerprint")
    },
)
print("  msg.id        :", msg.id)

# G. 破坏实验：见教材 ⑤
