"""第 0 章环境自检。在项目根目录运行：python langchain_learn/00_env_check.py

只做一件事：把'环境到底行不行'一次性问清楚，避免后面 10 章反复踩同一个坑。
"""

import importlib
import os
import platform
import sys
import warnings

# 第三方库在 1.x 过渡期的弃用噪音，先压掉，免得盖住我们自己的输出
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

try:
    from dotenv import load_dotenv
except ImportError:
    print("✗ 缺少 python-dotenv。请先执行：pip install python-dotenv")
    raise SystemExit(1)

load_dotenv()

print("=" * 64)
print("① 已安装的 LangChain 系版本")
print("=" * 64)
targets = [
    ("langchain", "__version__"),
    ("langchain_core", "__version__"),
    ("langchain_openai", "__version__"),
    ("langchain_community", "__version__"),
    ("langgraph", "__version__"),
    ("deepagents", "__version__"),
    ("langgraph.checkpoint.sqlite", "__version__"),
]
for mod, attr in targets:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<26} {getattr(m, attr, '?')}")
    except Exception as e:
        print(f"  {mod:<26} (未安装: {type(e).__name__})")

print()
print("=" * 64)
print("② create_agent 的实际来源（关键：确认它不是 v0 的旧实现）")
print("=" * 64)
try:
    from langchain.agents import create_agent

    print("  import 路径 :", create_agent.__module__)
    print("  所在文件    :", sys.modules[create_agent.__module__].__file__)
except Exception as e:
    print("  ✗ 导入失败:", type(e).__name__, e)

print()
print("=" * 64)
print("③ 解释器与平台")
print("=" * 64)
print("  python     :", sys.version.split()[0])
print("  executable :", sys.executable)
print("  prefix     :", sys.prefix)
print("  是否虚拟环境:", sys.prefix != sys.base_prefix)
print("  平台       :", platform.platform())

print()
print("=" * 64)
print("④ .env 关键变量（值做脱敏）")
print("=" * 64)
for key in ("LLM_MODEL_ID", "LLM_BASE_URL", "LLM_TIMEOUT", "LLM_API_KEY"):
    val = os.getenv(key)
    if val is None:
        print(f"  {key:<14} (未设置)")
    elif key == "LLM_API_KEY":
        print(f"  {key:<14} {val[:6]}...{val[-4:]}  (len={len(val)})")
    else:
        print(f"  {key:<14} {val}")

print()
print("=" * 64)
print("⑤ 连通性 + 工具调用能力（这一项过不了，后面全章都别学）")
print("=" * 64)
try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        timeout=60,
    )
    reply = llm.invoke("只回四个字：连接正常")
    print("  基础对话 :", repr(reply.content)[:80])

    @tool
    def add(a: int, b: int) -> int:
        """两个整数相加。"""
        return a + b

    bound = llm.bind_tools([add])
    msg = bound.invoke("用工具算一下 37 加 58 等于多少")
    calls = getattr(msg, "tool_calls", None)
    if calls:
        print("  工具调用 : ✓", calls)
    else:
        print("  工具调用 : ✗ 模型没有返回 tool_calls ← 必须换模型，否则本书大部分章节无法进行")
        print("             当前返回:", repr(msg.content)[:120])
except Exception as e:
    print("  ✗ 失败:", type(e).__name__, e)
    print("    排查顺序：API Key 是否正确 → base_url 是否可达 → 是否需开代理")

print()
print("=" * 64)
print("自检结束。①②⑤ 全绿才可以进第 1 章。")
print("=" * 64)
