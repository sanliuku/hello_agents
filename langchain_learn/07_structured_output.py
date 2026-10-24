"""第 7 章 · 结构化输出：response_format 到底把 schema 变成了什么

用法（在项目根目录运行）：
    python langchain_learn/07_structured_output.py
    python langchain_learn/07_structured_output.py --real

A–E 段零 token 消耗（不调模型，纯解析/构造对象）。
F–G 段发真实请求，必须加 --real。

⚠️ 本章会 import 带下划线的私有 API（_SchemaSpec / OutputToolBinding /
ProviderStrategyBinding / _parse_with_schema）。这是为了"看见内部"，
生产代码不要这么写——生产只用 ToolStrategy / ProviderStrategy / 裸 schema。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

# ---- 私有 API：只为解剖而用，生产禁用 --------------------------------
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    ToolStrategy,
    _SchemaSpec,
    _parse_with_schema,
)

from _shared import dump_messages, get_model


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# =====================================================================
# 四种 schema 形态（后面 A 段要逐个喂给 _SchemaSpec）
# =====================================================================

class Person(BaseModel):
    """一个人。这段 docstring 会变成工具描述，发给模型。"""

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
    "properties": {
        "city": {"type": "string"},
        "celsius": {"type": "number"},
    },
    "required": ["city", "celsius"],
}


# =====================================================================
# A. _SchemaSpec：所有翻译的起点
# =====================================================================
def part_a() -> None:
    section("A. _SchemaSpec：四种 schema 被压成同一张名片")

    cases = [
        ("Pydantic BaseModel", Person),
        ("dataclass", Order),
        ("TypedDict", Movie),
        ("raw dict (JSON Schema)", RAW_SCHEMA),
    ]

    for label, schema in cases:
        spec = _SchemaSpec(schema)
        print(f"\n--- {label} ---")
        print(f"  name        : {spec.name}")
        print(f"  description : {spec.description!r}")
        print(f"  schema_kind : {spec.schema_kind}")
        print(f"  json_schema : {json.dumps(spec.json_schema, ensure_ascii=False)[:160]}")

    print("\n要点：")
    print("  1) 类的 description 来自 __doc__ —— 写 docstring 不是装饰，是发给模型的工具描述")
    print("  2) dict 的 schema_kind 是 'json_schema'，且 json_schema 原样不动")
    print("  3) Pydantic 走 model_json_schema()，dataclass/TypedDict 走 TypeAdapter().json_schema()")

    # 无 title、无 __name__ 的 dict 会怎样？
    no_title = {"type": "object", "properties": {"x": {"type": "string"}}}
    spec = _SchemaSpec(no_title)
    print(f"\n  无 title 的 dict → name = {spec.name!r}（uuid 前 4 位兜底，每次不同）")

    # 传实例会怎样？
    try:
        _SchemaSpec(Person(name="a", age=1))  # type: ignore[arg-type]
    except ValueError as e:
        print(f"\n  传实例 → ValueError: {e}")


# =====================================================================
# B. ToolStrategy：Union 被拆成多个假工具
# =====================================================================
def part_b() -> None:
    section("B. ToolStrategy.schema_specs：单 schema 一条，Union 多条")

    single = ToolStrategy(schema=Person)
    print(f"\n单 schema   Person            → {len(single.schema_specs)} 个 spec")
    for s in single.schema_specs:
        print(f"    - {s.name} ({s.schema_kind})")

    union = ToolStrategy(schema=Person | Order)
    print(f"\nUnion       Person | Order    → {len(union.schema_specs)} 个 spec")
    for s in union.schema_specs:
        print(f"    - {s.name} ({s.schema_kind})")

    nested = ToolStrategy(schema=Person | Order | Movie)
    print(f"\n三元 Union  Person|Order|Movie → {len(nested.schema_specs)} 个 spec")
    for s in nested.schema_specs:
        print(f"    - {s.name} ({s.schema_kind})")

    one_of = ToolStrategy(schema={"oneOf": [Person.model_json_schema(), RAW_SCHEMA]})
    print(f"\nJSON oneOf  {{'oneOf': [...]}}    → {len(one_of.schema_specs)} 个 spec")
    for s in one_of.schema_specs:
        print(f"    - {s.name} ({s.schema_kind})")

    print("\n要点：模型面前摆着 N 个工具，它就可能 N 个都调 → MultipleStructuredOutputsError")
    print("      ProviderStrategy 不支持 UnionType（它的 schema 类型没有 UnionType）")

    # handle_errors 的六种取值
    print("\nhandle_errors 六种取值：True / str / type[Exception] / tuple[Exception,...] / Callable / False")
    for he in (True, "请重试", ValueError, (ValueError, KeyError), False):
        s = ToolStrategy(schema=Person, handle_errors=he)  # type: ignore[arg-type]
        print(f"  handle_errors={he!r:<28} → {s.handle_errors!r}")


# =====================================================================
# C. OutputToolBinding：假工具长什么样
# =====================================================================
def part_c() -> None:
    section("C. OutputToolBinding.from_schema_spec：假工具的出生证明")

    from langchain.agents.structured_output import OutputToolBinding

    spec = _SchemaSpec(Person)
    binding = OutputToolBinding.from_schema_spec(spec)
    tool = binding.tool

    print(f"\n  tool.name        : {tool.name}")
    print(f"  tool.description : {tool.description!r}")
    print(f"  type(args_schema): {type(tool.args_schema).__name__}")
    print(f"  args_schema keys : {sorted(tool.args_schema) if isinstance(tool.args_schema, dict) else 'n/a'}")
    print(f"  has .func        : {hasattr(tool, 'func') and tool.func is not None}")
    print(f"  has .coroutine   : {hasattr(tool, 'coroutine') and tool.coroutine is not None}")

    print("\n  它最终被 convert_to_openai_tool 翻译成模型看到的 tools[0]：")
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        print("  " + json.dumps(convert_to_openai_tool(tool), ensure_ascii=False)[:300])
    except Exception as e:  # noqa: BLE001
        print(f"  （转换失败，可跳过：{e}）")

    # 回程：从 tool_call['args'] 解析
    good = {"name": "张三", "age": 30}
    print(f"\n  parse({good}) → {binding.parse(good)}")
    try:
        binding.parse({"name": "张三", "age": -1})
    except ValueError as e:
        print(f"  parse(age=-1) → ValueError: {str(e)[:120]} ...")

    print("\n要点：args_schema 是原始 JSON Schema dict，不是 Pydantic 类；")
    print("      tool 没有 func/coroutine —— 它永远不会被执行，只是给模型看的签名贴纸。")


# =====================================================================
# D. ProviderStrategy：不建工具，直接改请求体
# =====================================================================
def part_d() -> None:
    section("D. ProviderStrategy.to_model_kwargs()：直接产 response_format")

    for strict in (None, True, False):
        ps = ProviderStrategy(schema=Person, strict=strict)
        kwargs = ps.to_model_kwargs()
        pretty = json.dumps(kwargs, ensure_ascii=False)
        print(f"\n  strict={strict!r}")
        print(f"    {pretty[:220]}")

    print("\n要点：")
    print("  1) 'strict' 只在真值时出现 —— None / False 时这个 key 根本不存在（不是发 false）")
    print("  2) 这个 dict 直接被 model.bind(**kwargs) 吃进去，拼进 HTTP 请求体")
    print("  3) 没有创建任何 StructuredTool —— 这是和 ToolStrategy 最根本的区别")

    # 回程：从 AIMessage 里抠 JSON
    from langchain.agents.structured_output import ProviderStrategyBinding
    from langchain_core.messages import AIMessage

    psb = ProviderStrategyBinding.from_schema_spec(_SchemaSpec(Person))
    msg = AIMessage(content='{"name": "李四", "age": 25}')
    print(f"\n  parse(AIMessage(content='{{...}}')) → {psb.parse(msg)}")

    bad = AIMessage(content="抱歉，我无法给出结构化结果。")
    try:
        psb.parse(bad)
    except ValueError as e:
        print(f"  parse(非 JSON) → ValueError: {str(e)[:100]} ...")

    print("\n  注意：ProviderStrategyBinding 只有 schema + schema_kind 两个字段，没有 tool。")


# =====================================================================
# E. AutoStrategy：一个只有字段的类
# =====================================================================
def part_e() -> None:
    section("E. AutoStrategy：什么方法都没有，它只是个『待决』标记")

    auto = AutoStrategy(schema=Person)
    print(f"\n  auto.schema      : {auto.schema}")
    print(f"  type             : {type(auto)}")
    public = [n for n in dir(auto) if not n.startswith("_")]
    print(f"  公开成员（除 schema）: {[n for n in public if n != 'schema']}")
    print(f"  有 to_model_kwargs 吗: {hasattr(auto, 'to_model_kwargs')}")
    print(f"  有 schema_specs 吗  : {hasattr(auto, 'schema_specs')}")

    print("\n  为什么需要它？")
    print("    create_agent 是编译期建图；而『模型是否支持原生结构化输出』要等运行期")
    print("    读到 model.profile 才知道。创建期只能把状态记成『未定』→ 这就是 AutoStrategy。")
    print("    路径：AutoStrategy →(创建期预支)→ ToolStrategy 建好假工具")
    print("          →(运行期)→ 支持则换 ProviderStrategy，不支持则复用那批工具")


# =====================================================================
# F. 真实调用：response_format=裸 schema（走 AutoStrategy）
# =====================================================================
def part_f(real: bool) -> None:
    section("F. 真实调用：response_format=Person")

    if not real:
        print("\n  （跳过，需加 --real）")
        print("  期望：result['structured_response'] 是 Person 实例")
        print("        result['messages'] 最后一条是 AIMessage，可能带 tool_calls（走 ToolStrategy 时）")
        return

    from langchain.agents import create_agent

    model = get_model()
    agent = create_agent(model=model, response_format=Person)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫王五，今年 42 岁。"}]}
    )

    dump_messages(result["messages"])

    sr = result.get("structured_response")
    print(f"\n  structured_response      : {sr!r}")
    print(f"  type                     : {type(sr)}")
    if isinstance(sr, Person):
        print(f"  sr.name / sr.age         : {sr.name} / {sr.age}")

    print("\n  诚实提醒：如果模型不支持原生结构化输出，这里走的是 ToolStrategy，")
    print(f"  messages 里会有一条 tool_call + 一条 ToolMessage。上面轨迹的 tool_calls 字段就是了。")


# =====================================================================
# G. 破坏实验：handle_errors 三种取值
# =====================================================================
def part_g(real: bool) -> None:
    section("G. 破坏实验：让模型必须失败，看 handle_errors 怎么反应")

    class Impossible(BaseModel):
        """一个几乎不可能满足的 schema：值必须 >= 9999。"""

        score: int = Field(description="分数", ge=9999)

    if not real:
        print("\n  （跳过，需加 --real）")
        print("  三种配置各跑一次：")
        print("    (1) handle_errors=True       → 错误变成 ToolMessage 塞回去，agent 循环再问一次")
        print("    (2) handle_errors=False      → 直接抛 StructuredOutputValidationError")
        print("    (3) handle_errors='请重试'    → 错误消息是你那句中文，同样循环再问")
        print("  以及：用 raw dict schema 时，handle_errors 静默失效（_parse_with_schema 原样返回）")
        return

    from langchain.agents import create_agent
    from langchain.agents.structured_output import StructuredOutputValidationError

    model = get_model()

    def run(cfg, label: str) -> None:
        print(f"\n--- {label} ---")
        agent = create_agent(model=model, response_format=cfg)
        try:
            r = agent.invoke(
                {"messages": [{"role": "user", "content": "随便给我一个分数，越小越好，最好是 1 分。"}]}
            )
            print(f"  structured_response: {r.get('structured_response')!r}")
            print(f"  消息条数            : {len(r['messages'])}（>2 说明循环重试过）")
            dump_messages(r["messages"])
        except StructuredOutputValidationError as e:
            print(f"  抛 StructuredOutputValidationError: {str(e)[:160]}")
            print(f"  .tool_name = {e.tool_name}")
        except Exception as e:  # noqa: BLE001
            print(f"  抛 {type(e).__name__}: {str(e)[:160]}")

    run(ToolStrategy(schema=Impossible, handle_errors=True), "(1) handle_errors=True")
    run(ToolStrategy(schema=Impossible, handle_errors=False), "(2) handle_errors=False")
    run(ToolStrategy(schema=Impossible, handle_errors="请重试，score 必须 >= 9999"), "(3) handle_errors='请重试'")

    # (4) raw dict：handle_errors 静默失效
    print("\n--- (4) raw dict schema + handle_errors=True → 静默失效 ---")
    broken = {
        "title": "Score",
        "type": "object",
        "properties": {"score": {"type": "string"}},
        "required": ["score"],
    }
    agent = create_agent(model=model, response_format=ToolStrategy(schema=broken, handle_errors=True))
    try:
        r = agent.invoke({"messages": [{"role": "user", "content": "score 填 1（数字）"}]})
        sr = r.get("structured_response")
        print(f"  structured_response: {sr!r}")
        print(f"  type(score)        : {type(sr.get('score')) if isinstance(sr, dict) else 'n/a'}")
        print("  如果 score 是 int 而不是 str，就证明：dict schema 压根没校验（_parse_with_schema 第一行 return data）")
    except Exception as e:  # noqa: BLE001
        print(f"  抛 {type(e).__name__}: {str(e)[:160]}")


# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="执行需要真实模型的 F/G 段")
    args = parser.parse_args()

    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f(args.real)
    part_g(args.real)

    section("收尾：三句话总结")
    print("  1) ToolStrategy  = 把 schema 伪装成一个函数，让模型去『调用』它")
    print("  2) ProviderStrategy = 把 schema 塞进请求头 response_format，让模型去『遵守』它")
    print("  3) AutoStrategy  = 一个空标记，运行期看 model.profile 决定落到上面哪个")
    print("\n  回程校验只有一个入口：_parse_with_schema(schema, schema_kind, data)")
    print("  而它对 dict schema 直接 return data —— 不校验。这就是 handle_errors 失效的根因。")


if __name__ == "__main__":
    sys.exit(main())
