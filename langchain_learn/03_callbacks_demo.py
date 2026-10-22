"""03_callbacks_demo.py —— ch03 补充：CallbackManager 到底怎么用。

跑法（必须在项目根目录、用脚本方式）：
    python langchain_learn/03_callbacks_demo.py

A、A2、B 段纯本地，零 token，不需要 .env；
C 段才会发真实请求（需要 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL）。

本章要看清的三件事：
1. CallbackManager.configure 的 7 个参数谁盖谁；
2. 它返回的是 list[CallbackManagerForLLMRun]，不是单个；
3. handler 里的方法名，就是你在 astream_events 里看到的事件名。

已实跑（2026-10-22，langchain 1.4.6 / core 1.6.12 / qwen3:latest@ollama）。
实测结论见教材 2.6「实跑记录」小节；其中两条最反直觉：
- metadata 是"本地覆盖继承"，tags 是"拼接"——两种合并规则不一样；
- run_id.hex[:8] 属于 UUIDv7 的时间戳段，同一毫秒内两个 run 前缀相同，不能当 id 用。
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from _shared import get_model


# --------------------------------------------------------------- handler
class TraceHandler(BaseCallbackHandler):
    """把 chat model 的四个事件全打出来。

    只覆写你想看的钩子即可，没覆写的走基类的空实现。
    """

    # 基类里默认是 False：handler 自己抛异常会被捕获并记日志，不打断模型调用。
    # 想让它真的炸开，改成 True —— 这正好是本章破坏实验的素材。
    raise_error = False

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        # messages 是 list[list[BaseMessage]]：一次 generate 可能带多条 prompt
        params = kwargs.get("invocation_params") or {}
        shown = {
            k: params.get(k)
            for k in ("model", "model_name", "temperature", "stop", "_type")
            if params.get(k) is not None
        }
        # 截尾部 8 位，不截头部：run_id 是 UUIDv7，前 8 位是时间戳段，
        # 同一毫秒内产生的两个 run 前缀完全相同（A2 段会看到这个坑）。
        print(f"  [start] run_id=…{str(run_id)[-8:]} prompts={len(messages)} first={messages[0][0].content!r}")
        print(f"          真实参数={shown}")
        print(f"          tags={tags} metadata={metadata}")

    def on_llm_new_token(self, token, *, chunk=None, run_id, parent_run_id=None, tags=None, **kwargs):
        # 只有 _stream / _astream 路径会触发；非流式 invoke 一个 token 事件都没有
        print(f"  [token] {token!r}")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs):
        msg = response.generations[0][0].message
        print(f"  [end]   run_id=…{str(run_id)[-8:]} content={msg.content!r}")
        print(f"          usage={getattr(msg, 'usage_metadata', None)}")

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        print(f"  [error] {type(error).__name__}: {error}")


# ---------------------------------------------------------------- A 段
def section_a_manual():
    """不碰模型，手工驱动 CallbackManager —— 看清它返回什么。"""
    print("=" * 66)
    print("A. 手工调 CallbackManager.configure（零 token）")

    cm = CallbackManager.configure(
        inheritable_callbacks=None,      # 父级传下来的；agent 里就是 config["callbacks"]
        local_callbacks=[TraceHandler()],  # 本对象自带的；= ChatOpenAI(callbacks=[...])
        verbose=False,
        inheritable_tags=None,
        local_tags=["demo"],
        inheritable_metadata=None,
        local_metadata={"where": "section_a"},
    )
    print(f"  handlers            = {[type(x).__name__ for x in cm.handlers]}")
    print(f"  inheritable_handlers= {[type(x).__name__ for x in cm.inheritable_handlers]}")
    print(f"  tags                = {cm.tags}")
    print(f"  metadata            = {cm.metadata}")

    # 第二个参数是 list[list[BaseMessage]]，返回 list[CallbackManagerForLLMRun]
    run_managers = cm.on_chat_model_start(
        {"name": "fake-model"},
        [[HumanMessage("你好")]],
        invocation_params={"model": "fake-model", "temperature": 0.3},
        options={},
    )
    print(f"  on_chat_model_start -> {type(run_managers).__name__}, len={len(run_managers)}")

    rm = run_managers[0]
    print(f"  run_manager.run_id  = {rm.run_id}")
    rm.on_llm_new_token("你")
    rm.on_llm_new_token("好")
    rm.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("你好"))]]))


def section_a2_batch_shape():
    """两条 prompt 就两个 run_manager —— 这才是 list 的原因。"""
    print("=" * 66)
    print("A2. 为什么返回 list：batch 时一条 prompt 一个")
    cm = CallbackManager.configure(local_callbacks=[TraceHandler()], verbose=False)
    rms = cm.on_chat_model_start(
        {"name": "fake-model"},
        [[HumanMessage("第一条")], [HumanMessage("第二条")]],
        invocation_params={"model": "fake-model"},
        options={},
    )
    print(f"  2 条 prompt -> {len(rms)} 个 run_manager，run_id 各不相同：")
    print(f"  前缀相同、尾部不同：{[str(r.run_id)[-8:] for r in rms]}")
    print(f"  前 8 位（UUIDv7 时间戳段）反而一样：{[str(r.run_id)[:8] for r in rms]}")


# ---------------------------------------------------------------- B 段
def section_b_merge_rule():
    """configure 的合并规则 + tag/metadata 怎么往子链传。"""
    print("=" * 66)
    print("B. 合并规则：继承的在前，本地的在后（同名不去重，两个都留）")

    cm = CallbackManager.configure(
        inheritable_callbacks=[TraceHandler()],   # 来自 config
        local_callbacks=[TraceHandler()],         # 来自构造函数
        verbose=False,
        inheritable_tags=["from-config"],
        local_tags=["from-self"],
        inheritable_metadata={"layer": "config"},
        local_metadata={"layer": "self"},
    )
    print(f"  handlers 数量 = {len(cm.handlers)}")
    print(f"  tags          = {cm.tags}")
    print(f"  metadata      = {cm.metadata}")

    # 子链再 configure 一次：这一层决定哪些 handler 能继续往下传
    child = cm.configure(
        inheritable_callbacks=cm.inheritable_handlers,
        local_callbacks=None,
        verbose=False,
        inheritable_tags=cm.inheritable_tags,
        local_tags=None,
        inheritable_metadata=cm.inheritable_metadata,
        local_metadata=None,
    )
    print(f"  child handlers= {len(child.handlers)}  child tags={child.tags}")


# ---------------------------------------------------------------- C 段
def section_c_real_calls():
    """真实调用：三个注入位置，外加一条对照。"""
    model = get_model()

    print("=" * 66)
    print("C1. 绑在对象上（with_config），此后的每次调用都带")
    tagged = model.with_config(callbacks=[TraceHandler()], tags=["bound"])
    tagged.invoke("用一句话说明什么是回调")

    print("=" * 66)
    print("C2. 只给这一次 invoke（config 里的 callbacks）")
    model.invoke("再说一句", config={"callbacks": [TraceHandler()], "tags": ["only-this-call"]})

    print("=" * 66)
    print("C3. 流式：on_llm_new_token 只有这条路径会触发")
    for _ in model.stream("数到三", config={"callbacks": [TraceHandler()]}):
        pass

    print("=" * 66)
    print("C4. 对照：同一个 model，不带 callback，应该什么都不打印")
    model.invoke("这次应该静默")


if __name__ == "__main__":
    section_a_manual()
    section_a2_batch_shape()
    section_b_merge_rule()
    section_c_real_calls()
