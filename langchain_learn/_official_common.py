"""官方对照示例的共用小工具。

只做三件事：分节横幅、判断要不要联网、把长 dict 打印短一点。
真正的"造模型"仍然走 `_shared.get_model()`，不重复一遍。
"""

import sys


def banner(tag: str, title: str) -> None:
    """打印一节的开头。tag 用 A/B/C，title 一句话说清这节要看什么。"""
    print("\n" + "=" * 66)
    print(f"[{tag}] {title}")
    print("=" * 66)


def real_enabled() -> bool:
    """本仓约定：默认不联网，命令行加 --real 才发请求。

    为什么这么做：qwen3 一次调用要 20-40 秒，而且每章都联网会让"读懂结构"
    变成"等模型回话"。官方对照示例的重点是**看 API 面**，不是看模型发挥。
    """
    return "--real" in sys.argv


def skip_hint(script: str) -> None:
    """没带 --real 时给一句提示，别让人以为脚本坏了。"""
    print(f"  （跳过联网段。要真跑：python {script} --real）")


def short(value, limit: int = 160) -> str:
    """把任意对象压成一行短字符串，方便打表。"""
    text = str(value).replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")
