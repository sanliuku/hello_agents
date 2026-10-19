"""langchain_learn 共用工具：读 .env 造模型、打印消息轨迹。

运行约定：所有章节脚本都在**项目根目录**用脚本方式运行
    python langchain_learn/xx.py
（脚本所在目录会进 sys.path[0]，所以可以 `from _shared import ...`）
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_model(temperature: float = 0.0) -> ChatOpenAI:
    """按 .env 构造一个 OpenAI 兼容模型。缺 key 时立刻抛错，而不是等到 invoke。"""
    missing = [k for k in ("LLM_API_KEY", "LLM_MODEL_ID") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f".env 缺少 {missing}，请先完成第 0 章")
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=temperature,
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


def dump_messages(messages, max_chars: int = 220) -> None:
    """把整条消息轨迹打印成人能读的样子。后面每章都用它当'显微镜'。"""
    print(f"\n--- 消息轨迹（共 {len(messages)} 条）---")
    for i, m in enumerate(messages):
        print(f"\n[{i}] {type(m).__name__}")
        calls = getattr(m, "tool_calls", None)
        if calls:
            print("    tool_calls  :", calls)
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            print("    tool_call_id:", tcid)
        content = m.content
        text = content if isinstance(content, str) else str(content)
        text = (text or "").replace("\n", " ")
        print(f"    content     : {text[:max_chars]}")
