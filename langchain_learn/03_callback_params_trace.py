"""03_callback_params_trace.py —— callbacks 与 provider 参数，到底各自在哪被用

目的：用一个**带自定义参数的 handler**，把 callbacks 从三个入口塞进去，
      同时把 temperature 从三个位置塞进去，然后把两边的事件流对照着看。
      最后进 agent，看 config 里的 callbacks 是怎么被一路传下去的。

运行（必须在项目根目录，脚本方式）：
    python langchain_learn/03_callback_params_trace.py

分段：
    A  写一个带自己参数的 handler（不联网）
    B  纯本地把 CallbackManager 走一遍（不联网，零 token）
    C  temperature 写在三个位置，看 invocation_params 变成什么（联网 4 次）
    D  偷看真发出去的 payload（联网 1 次）
    E  把 callbacks 放进 agent 的 config（联网 1~2 次）
    F  汇总：每个东西最终在哪一层被消费

已实跑（2026-10-22，langchain 1.4.6 / core 1.6.12 / openai 1.7.2 / qwen3:latest@ollama）。
       D 段 `_get_request_payload` 在 langchain-openai 1.7.2 上签名可用；
       若你本地版本报 TypeError，就是该方法的签名漂移，不是你的错。
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def env_model_kwargs(**overrides: Any) -> dict[str, Any]:
    """从 .env 拼出 ChatOpenAI 的构造参数（就是你平常用的 LLM_* 那套）。"""
    kwargs: dict[str, Any] = {
        "model": os.environ["LLM_MODEL_ID"],
        "api_key": os.environ["LLM_API_KEY"],
        "temperature": 0.0,
        "timeout": int(os.environ.get("LLM_TIMEOUT") or 60),
    }
    if os.environ.get("LLM_BASE_URL"):
        kwargs["base_url"] = os.environ["LLM_BASE_URL"]
    kwargs.update(overrides)
    return kwargs


def show(records: list[dict[str, Any]], limit: int = 20) -> None:
    if not records:
        print("    （这个 handler 一条事件都没收到）")
        return
    for row in records[:limit]:
        print("    " + ", ".join(f"{k}={v}" for k, v in row.items()))
    if len(records) > limit:
        print(f"    ... 还有 {len(records) - limit} 条")


# --------------------------------------------------------------------------- #
# A  一个带自己参数的 handler
# --------------------------------------------------------------------------- #
class Trace(BaseCallbackHandler):
    """带两个**自己的**参数：label 和 verbose_level。

    这两个参数只影响这个 handler 怎么打印，永远不会进入请求体。
    这正是要证明的事：handler 上的东西是"旁路"的。
    """

    # handler 自己出错时是否让主流程崩。默认就是 False。
    raise_error = False

    def __init__(self, label: str, *, verbose_level: int = 1) -> None:
        self.label = label
        self.verbose_level = verbose_level
        self.records: list[dict[str, Any]] = []
        self._tokens: dict[Any, int] = {}

    # ---- 内部：记一条 ----
    def _rec(self, event: str, **fields: Any) -> None:
        row = {"event": event, "label": self.label, **fields}
        self.records.append(row)
        if self.verbose_level >= 2:
            print("    [实时] " + ", ".join(f"{k}={v}" for k, v in row.items()))

    # ---- chat model：开始 / 流式 token / 结束 / 报错 ----
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        inv = kwargs.get("invocation_params") or {}
        self._rec(
            "chat_model_start",
            root=parent_run_id is None,
            tags=list(tags or []),
            meta_keys=sorted((metadata or {}).keys()),
            model=inv.get("model") or inv.get("model_name"),
            temperature=inv.get("temperature"),
            prompts=len(messages),
            msgs_per_prompt=[len(m) for m in messages],
        )

    def on_llm_new_token(self, token: str, *, chunk: Any = None, run_id: Any, **kwargs: Any) -> None:
        # 只有 .stream() / .astream() 才会走到这里
        self._tokens[run_id] = self._tokens.get(run_id, 0) + len(token or "")

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        self._rec(
            "llm_end",
            streamed_chars=self._tokens.get(run_id, 0),
            usage_keys=sorted(usage.keys()),
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._rec("llm_error", error_type=type(error).__name__, error=str(error)[:100])

    # ---- agent 里会额外冒出这些 ----
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name")
        self._rec("chain_start", name=name, root=parent_run_id is None, tags=list(tags or [])[:5])

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._rec(
            "tool_start",
            name=(serialized or {}).get("name") or kwargs.get("name"),
            input_str=str(input_str)[:60],
            parent_is_node=parent_run_id is not None,
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._rec("tool_end", output=str(output)[:60])


# --------------------------------------------------------------------------- #
# B  纯本地：自己驱动 CallbackManager（零 token，先看清机制）
# --------------------------------------------------------------------------- #
def section_b_local_manager() -> None:
    print("\n=== B 纯本地：手动走一遍 CallbackManager.configure + on_chat_model_start ===")
    h_from_config = Trace("来自上游 config", verbose_level=2)
    h_from_model = Trace("来自模型构造参数", verbose_level=2)

    manager = CallbackManager.configure(
        [h_from_config],            # 1 inheritable_callbacks：上游传下来的
        [h_from_model],             # 2 local_callbacks：对象自己身上带的
        False,                      # 3 verbose
        ["tag-from-config"],        # 4 inheritable_tags
        ["tag-from-object"],        # 5 local_tags
        {"trace_id": "abc"},        # 6 inheritable_metadata
        {"owner": "demo"},          # 7 local_metadata
    )
    print(f"  configure 收 7 个参数，拿到 1 个 manager：{type(manager).__name__}")

    # messages 是 list[list[BaseMessage]]：一次可以喂多组对话
    batches = [
        [HumanMessage("第一组对话")],
        [HumanMessage("第二组A"), AIMessage("第二组B")],
    ]
    run_managers = manager.on_chat_model_start(
        {"name": "ChatOpenAI"},
        batches,
        invocation_params={"model": "demo-model", "temperature": 0.42},
        batch_size=len(batches),
    )
    print(f"  喂 {len(batches)} 组 messages → 拿回 {len(run_managers)} 个 run_manager")
    for i, rm in enumerate(run_managers):
        print(f"    run_manager[{i}].run_id = {rm.run_id}")

    # 结束事件得你自己叫，否则 handler 永远等不到 on_llm_end
    fake = LLMResult(
        generations=[[ChatGeneration(message=AIMessage("假装这是模型回复"))]],
        llm_output={"token_usage": {"total_tokens": 0}},
    )
    for rm in run_managers:
        rm.on_llm_end(fake)

    print("  --- 上游 handler 收到 ---")
    show(h_from_config.records)
    print("  --- 模型本地 handler 收到 ---")
    show(h_from_model.records)
    print("  ↑ 注意两边的 tags / meta_keys：configure 会把继承的和本地的**合并**，不是覆盖")


# --------------------------------------------------------------------------- #
# C  temperature 写在三个位置
# --------------------------------------------------------------------------- #
def section_c_temperature() -> None:
    print("\n=== C temperature 写在哪，invocation_params 里就是什么 ===")

    print("\n  C1 构造层 temperature=0.0，invoke 不再传")
    h1 = Trace("C1", verbose_level=2)
    m1 = ChatOpenAI(**env_model_kwargs(temperature=0.0), callbacks=[h1])
    m1.invoke("用五个字回答：你好")

    print("\n  C2 构造层 0.0，调用层传 temperature=0.9（看谁赢）")
    h2 = Trace("C2", verbose_level=2)
    m2 = ChatOpenAI(**env_model_kwargs(temperature=0.0), callbacks=[h2])
    m2.invoke("用五个字回答：你好", temperature=0.9)

    print("\n  C3 用 .bind(temperature=0.7) 预置（返回的是 RunnableBinding）")
    h3 = Trace("C3", verbose_level=2)
    m3 = ChatOpenAI(**env_model_kwargs(temperature=0.0), callbacks=[h3]).bind(temperature=0.7)
    m3.invoke("用五个字回答：你好")

    print("\n  C4 构造时写 callbacks=[]，但 invoke 时 config 传 handler —— 谁生效？")
    h4 = Trace("C4-来自config", verbose_level=2)
    m4 = ChatOpenAI(**env_model_kwargs(), callbacks=[])
    m4.invoke("用五个字回答：你好", config={"callbacks": [h4], "tags": ["from-c4"], "metadata": {"k": "v"}})
    print(f"    C4 的 handler 收到 {len(h4.records)} 条事件"
          f" —— 空 list 不等于关掉回调，它只是『我没有本地 handler』")


# --------------------------------------------------------------------------- #
# D  偷看真发出去的 payload
# --------------------------------------------------------------------------- #
class SpyChatOpenAI(ChatOpenAI):
    """唯一目的：把组好的 wire payload 打出来。

    这行 print 的位置告诉你：真正决定请求体的是 _default_params + **kwargs，
    跟 CallbackManager 一点关系都没有。
    """

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        shown = {k: v for k, v in payload.items() if k != "messages"}
        print(f"    [wire payload] {shown}")
        print(f"    [payload 里有 callbacks 吗] {'callbacks' in payload}")
        return payload


def section_d_payload() -> None:
    print("\n=== D 真发出去的 payload 长什么样 ===")
    h = Trace("D", verbose_level=2)
    model = SpyChatOpenAI(**env_model_kwargs(temperature=0.0), callbacks=[h])
    model.invoke("用三个字回答：你好", temperature=0.9, max_tokens=32)
    print("  ↑ 对比：handler 看到的 invocation_params 与 payload 里的 temperature 是同一个值；")
    print("    payload 里永远没有 callbacks / tags / metadata —— 它们走的是另一条线。")


# --------------------------------------------------------------------------- #
# E  agent 里放 callbacks
# --------------------------------------------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """查询某城市的天气。"""
    return f"{city}：晴，26℃"


def section_e_agent() -> None:
    print("\n=== E 把 callbacks 放进 agent 的 config ===")
    h = Trace("E-agent", verbose_level=1)
    model = ChatOpenAI(**env_model_kwargs())
    agent = create_agent(model, tools=[get_weather])

    out = agent.invoke(
        {"messages": [{"role": "user", "content": "北京天气怎么样？只用一句话回答。"}]},
        config={
            "callbacks": [h],
            "tags": ["agent-demo"],
            "metadata": {"trace_id": "t-001"},
        },
    )
    print(f"  最终回复：{out['messages'][-1].content}")
    print(f"  handler 共收到 {len(h.records)} 条事件：")
    show(h.records, limit=30)
    print("  看点：")
    print("    1) chat_model_start 出现不止一次 —— agent 是 model→tools→model 的循环，每轮一次")
    print("    2) tool_start / tool_end 的 parent_is_node=True —— 它挂在 tools 节点下面")
    print("    3) chain_start 的 root=False 且 tags 里带着 agent-demo —— 你传的 tags 会被继承下去")


# --------------------------------------------------------------------------- #
# F  汇总
# --------------------------------------------------------------------------- #
def section_f_summary() -> None:
    print("\n=== F 一句话汇总 ===")
    print("  temperature / max_tokens / model  →  **kwargs 或构造字段 → _get_request_payload → 请求体")
    print("  callbacks / tags / metadata       →  CallbackManager.configure → 只在本地播报，永不进请求体")
    print("  同一份 config 里的这两个东西，走的是两条互不相交的路。")


SECTIONS = [
    ("B", section_b_local_manager),
    ("C", section_c_temperature),
    ("D", section_d_payload),
    ("E", section_e_agent),
    ("F", section_f_summary),
]


def main() -> None:
    print(f"model = {os.environ.get('LLM_MODEL_ID')}")
    print(f"base_url = {os.environ.get('LLM_BASE_URL') or '(默认)'}")
    for name, fn in SECTIONS:
        try:
            fn()
        except Exception:  # 单段失败不影响后面的段
            print(f"\n  !! {name} 段出错（下面这行就是原因）：")
            traceback.print_exc()


if __name__ == "__main__":
    main()
