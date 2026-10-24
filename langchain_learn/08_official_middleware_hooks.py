"""第 8 章 · 官方对照：Middleware 上——钩子地图 -- 12 个钩子、编图顺序与洋葱结构

对应官方页：https://docs.langchain.com/oss/python/langchain/middleware
对应教材：docs/langchain-family/langchain_learn/09-ch08-middleware-hooks.md

官方骨架（本脚本逐段对照）：
  A. 钩子盘点       —— AgentMiddleware 到底有几个可覆盖方法，默认实现是什么
  B. 覆盖检测       —— factory.py 用 `__class__.x is not Base.x` 判断，不是 hasattr
  C. 洋葱顺序       —— `_chain_model_call_handlers` 的嵌套语义（纯函数模拟，不联模型）
  D. 节点 vs 包装器 —— 为什么 wrap_* 不成节点；handler 不调 = 不执行
  E. 装饰器         —— @before_model 把函数升级成单钩子中间件
  F. 真实编图       —— 只有这段需要 --real（会真的发一次模型请求）

运行：在**项目根目录**用脚本方式跑
    python langchain_learn/08_official_middleware_hooks.py          # A–E，零 token
    python langchain_learn/08_official_middleware_hooks.py --real   # 追加 F 段

说明：
  A–E 全部离线可跑，不构造模型、不发网络请求。F 段才 import 模型。
  本脚本只读 `langchain.agents.middleware` 的公开导出，不 import 任何私有函数；
  C 段是"按源码语义重写的等价实现"，不是直接调用私有接口——教材里已标注。
"""

from __future__ import annotations

import inspect
import sys

from langchain.agents.middleware import (
    AgentMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
)
from _official_common import banner, real_enabled, short, skip_hint

SCRIPT = "langchain_learn/08_official_middleware_hooks.py"

# 6 个语义钩子 × (同步名, 异步名)
HOOK_PAIRS = [
    ("before_agent", "abefore_agent"),
    ("before_model", "abefore_model"),
    ("after_model", "aafter_model"),
    ("wrap_model_call", "awrap_model_call"),
    ("after_agent", "aafter_agent"),
    ("wrap_tool_call", "awrap_tool_call"),
]


# ---------------------------------------------------------------- A. 钩子盘点
def segment_a() -> None:
    banner("A", "钩子盘点：12 个方法 = 6 个钩子 × 同步/异步")

    print("AgentMiddleware 直接定义在类体里的成员（跳过 dunder）：")
    for name, obj in vars(AgentMiddleware).items():
        if name.startswith("_"):
            continue
        kind = "property" if isinstance(obj, property) else type(obj).__name__
        print(f"  - {name:<22} {kind}")

    print("\n12 个钩子的默认实现（关键：是 raise，不是 pass）：")
    for sync_name, async_name in HOOK_PAIRS:
        for hook in (sync_name, async_name):
            fn = getattr(AgentMiddleware, hook, None)
            src = inspect.getsource(fn) if fn else ""
            body_is_raise = "raise NotImplementedError" in src
            print(f"  - {hook:<16} 默认 raise NotImplementedError = {body_is_raise}")

    print("\n观察点：")
    print("  1) 默认实现是 raise，所以 `hasattr(m, hook)` 对任何子类都恒为 True。")
    print("  2) 同步/异步是两个独立函数对象，只写一个就只有那一半算覆盖。")


# ---------------------------------------------------------------- B. 覆盖检测
class _A(AgentMiddleware):
    """只覆盖同步 before_model。"""

    def before_model(self, state, runtime):
        return None


class _B(AgentMiddleware):
    """只覆盖异步 abefore_model。"""

    async def abefore_model(self, state, runtime):
        return None


class _C(AgentMiddleware):
    """两个都覆盖。"""

    def after_model(self, state, runtime):
        return None

    async def aafter_model(self, state, runtime):
        return None


class _Empty(AgentMiddleware):
    """什么都不覆盖。"""


def _covers(m: AgentMiddleware, hook: str) -> bool:
    """复刻 factory.py:1163-1186 的判定，一次一个钩子。"""
    sync_name, async_name = hook, "a" + hook
    return (
        getattr(m.__class__, sync_name) is not getattr(AgentMiddleware, sync_name)
        or getattr(m.__class__, async_name) is not getattr(AgentMiddleware, async_name)
    )


def segment_b() -> None:
    banner("B", "覆盖检测：`__class__.x is not Base.x` 的真值表")

    cases = [(_Empty(), "什么都不覆盖"), (_A(), "只写同步 before_model"),
             (_B(), "只写异步 abefore_model"), (_C(), "after_model 双版本都写")]
    hooks = ["before_model", "after_model"]

    print(f"{'中间件':<28} {'说明':<24} " + "  ".join(f"{h:<14}" for h in hooks))
    for m, note in cases:
        cells = "  ".join(f"{str(_covers(m, h)):<14}" for h in hooks)
        print(f"{type(m).__name__:<28} {note:<24} {cells}")

    print("\n对比 hasattr（错误做法）——全部为 True，没有任何区分度：")
    for m, note in cases:
        print(f"  {type(m).__name__:<10} hasattr(before_model)={hasattr(m, 'before_model')}")

    print("\n陷阱：在 __init__ 里给【实例】塞方法，编译期看不到（看的是类）：")
    inst = _Empty()
    inst.before_model = lambda state, runtime: None  # type: ignore[method-assign]
    print(f"  实例上有 before_model = {hasattr(inst, 'before_model')}")
    print(f"  但覆盖检测结果      = {_covers(inst, 'before_model')}   ← 仍然是 False")
    print("  原因：factory.py 读的是 m.__class__，__init__ 里的赋值只改了实例字典。")


# ------------------------------------------------- C. 洋葱顺序（纯函数模拟）
def _chain_handlers(handlers):
    """等价于 langchain 的 _chain_model_call_handlers：first in list = outermost。

    源码语义（factory.py:1187-1224 + 该函数 docstring）：
        handler_i(request) = handlers[i](request, handler_{i+1})
    最内层直接调用真正的模型函数。
    """

    def chained(request, final):
        def build(i):
            if i == len(handlers):
                return final
            return lambda req: handlers[i](req, build(i + 1))

        return build(0)(request)

    return chained


class _Tracer:
    def __init__(self, name: str, depth: int):
        self.name, self.depth = name, depth

    def __call__(self, request, handler):
        pad = "  " * self.depth
        print(f"{pad}{self.name}.wrap_model_call 进入")
        out = handler(request)
        print(f"{pad}{self.name}.wrap_model_call 退出")
        return out


def segment_c() -> None:
    banner("C", "洋葱顺序：wrap_model_call 是嵌套括号，不是一串管道")

    names = ["A", "B", "C"]
    handlers = [_Tracer(n, i) for i, n in enumerate(names)]
    chained = _chain_handlers(handlers)

    def real_model(request):
        print("  " * len(names) + "⚙️ 真正的模型调用（最内层）")
        return {"ok": True}

    print(f"传入顺序：{names}\n期望：A 最先进入、最后退出（first = outermost）\n")
    chained({"req": "dummy"}, real_model)


# ------------------------------------------- D. 节点 vs 包装器 / 短路语义
def segment_d() -> None:
    banner("D", "节点 vs 包装器：handler 不被调用，模型就根本不会跑")

    class _FakeRequest:
        """仅为演示 override() 的不可变语义，不冒充真 ModelRequest。"""

        def __init__(self, payload: dict):
            self._p = dict(payload)

        def override(self, **kw):
            new = _FakeRequest(self._p)
            new._p.update(kw)
            return new

        def __repr__(self):
            return f"_FakeRequest({self._p})"

    calls = {"model": 0}

    def real_model(req):
        calls["model"] += 1
        return f"resp#{calls['model']}"

    class _CacheMiddleware:
        def __init__(self):
            self.cache = {}

        def wrap_model_call(self, request, handler):
            key = repr(request)
            if key in self.cache:
                print(f"  命中缓存，直接返回，不调 handler：{self.cache[key]}")
                return self.cache[key]
            resp = handler(request.override(mutated_by="cache-mw"))
            self.cache[key] = resp
            return resp

    mw = _CacheMiddleware()
    req = _FakeRequest({"payload": "q1"})
    req2 = req.override(x=1)

    print(f"原文 {req}  →  override 后 {req2}")
    print(f"不可变？原文未被改动：{req}")

    print("\n第一次：")
    print("  →", mw.wrap_model_call(req, real_model))
    print("第二次（同 key）：")
    print("  →", mw.wrap_model_call(req, real_model))
    print(f"\n真实模型被调用次数 = {calls['model']}（期望 1：第二次被短路）")
    print("\n结论：wrap_* 能决定“调不调”，所以它不能被编成节点（节点只能前后加料）。")


# ---------------------------------------------------------------- E. 装饰器
@before_model
def _count_calls(state, runtime):
    print("    [_count_calls] before_model 被调用")
    return None


@after_model(can_jump_to=["end"])
def _guard(state, runtime):
    print("    [_guard] after_model 被调用（声明了 can_jump_to=['end']）")
    return None


@dynamic_prompt
def _style(request):
    return (getattr(request, "system_prompt", "") or "") + " 务必先给结论。"


def segment_e() -> None:
    banner("E", "装饰器：把普通函数升级成单钩子中间件")

    for fn, label in ((_count_calls, "@before_model"), (_guard, "@after_model(can_jump_to=...)"),
                      (_style, "@dynamic_prompt")):
        print(f"{label:<28} → type={type(fn).__name__:<24} name={short(getattr(fn, 'name', None))}")
        print(f"{'':<28}   is AgentMiddleware 子类? {isinstance(fn, AgentMiddleware)}")

    print("\n观察点：装饰器返回的不是函数，而是一个中间件对象；它会被 append 进 middleware 列表。")
    print("  - @before_model 的钩子名默认取【函数名】（用于图节点命名）。")
    print("  - 需要跳转能力时必须显式给 can_jump_to，否则 jump_to 静默失效。")


# ---------------------------------------------------------------- F. 真编图
def segment_f() -> None:
    if not real_enabled():
        skip_hint(SCRIPT)
        return

    banner("F", "真实编图：把一个计数中间件塞进 create_agent，看它被调了几次")
    from langchain.agents import create_agent
    from _shared import get_model, dump_messages

    seen: list[str] = []

    class _CountMiddleware(AgentMiddleware):
        def before_model(self, state, runtime):
            seen.append("before_model")
            return None

        def after_model(self, state, runtime):
            seen.append("after_model")
            return None

    agent = create_agent(
        model=get_model(temperature=0.0),
        tools=[],
        middleware=[_CountMiddleware()],
        system_prompt="只回答一个词。",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "用中文说：钩子"}]})
    dump_messages(result["messages"])
    print(f"\n钩子实际命中顺序：{seen}")
    print("期望：['before_model', 'after_model']，且条目数与模型调用次数一致。")
    print("把 agent.ainvoke 再跑一次，可以看到 12 个钩子里只有你覆盖的那两个进图。")


def main() -> None:
    print(f"Python {sys.version.split()[0]} / langchain 钩子地图实验（A–E 零 token）")
    segment_a()
    segment_b()
    segment_c()
    segment_d()
    segment_e()
    segment_f()


if __name__ == "__main__":
    main()
