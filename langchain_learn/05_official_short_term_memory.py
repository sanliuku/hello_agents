"""第 5 章 · 官方对照：short-term-memory 页的五层——记忆存在哪、怎么改、怎么裁、怎么读写

对应官方页：https://docs.langchain.com/oss/python/langchain/short-term-memory
对应教材：docs/langchain-family/langchain_learn/06-ch05-state-and-reducer.md

官方那一页的骨架：
  Overview            → 概念：一个 thread 的历史（无代码）
  Usage               → A  checkpointer + thread_id：记忆到底存在哪
  Customizing memory  → B  state_schema：给 state 加自己的字段
  Common patterns
    Trim messages     → C  @before_model + REMOVE_ALL_MESSAGES 重建列表
    Delete messages   → D  RemoveMessage：删一条 / 全清
    Summarize messages→ E  SummarizationMiddleware：删不动了就换摘要
  Access memory
    Tools             → F  runtime.state 读 / Command(update=...) 写
    Prompt            → G  @dynamic_prompt + ModelRequest
    Before/After model→ 已在 C / D 用到（就是那两个钩子）

与 `05_state_reducer.py` 的分工：那个文件回答"**reducer 为什么必须存在**"（拆源码、
造失忆现场）；这个文件回答"**官方 API 怎么用**"（照着页面逐节跑一遍，看每层各自的边界）。
两个文件都要看，因为官方页只讲"怎么用"，不讲"用错了为什么不报错"。

A–G 全部离线：用一个只会数消息条数的假模型，零 token。
H 段加 --real 才真发请求（qwen3 一次 20-40 秒）。
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    SummarizationMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
)
from langchain.messages import AIMessage, RemoveMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import Field
from typing_extensions import NotRequired, TypedDict

from _official_common import banner, real_enabled, short, skip_hint
from _shared import get_model

SCRIPT = "langchain_learn/05_official_short_term_memory.py"


# --------------------------------------------------------------------------
# 零 token 的假模型：它不思考，只把"模型看到了什么"变成可打印的数字
# --------------------------------------------------------------------------
class MemoryProbe(BaseChatModel):
    """记录每次调用收到的 messages，并按脚本或固定句式回话。

    为什么每章都要这么一个东西：真模型"收到几条消息"你观察不到，而"失忆"的表现
    （答非所问）又会被模型的语言能力掩盖。把条数打成数字，证据才硬。
    """

    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[Any]] = Field(default_factory=list)
    echo: str = "答{n}：见到{c}条"

    @property
    def _llm_type(self) -> str:
        return "memory-probe"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        # create_agent 一定会调它；不实现会落到基类的 NotImplementedError
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        self.seen.append(list(messages))
        message = self.script.pop(0) if self.script else AIMessage(
            self.echo.format(n=len(self.seen), c=len(messages))
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def message_line(messages) -> str:
    """把一串消息压成 'Human:xxx | AI:yyy' 一行，方便看历史被改成了什么。"""
    return " | ".join(f"{type(m).__name__[:-7]}:{short(m.content, 12)}" for m in messages)


# --------------------------------------------------------------------------
banner("A", "Usage：checkpointer + thread_id —— 记忆存在哪")
# --------------------------------------------------------------------------
# 官方这一节的全部代码只有三行：create_agent(checkpointer=InMemorySaver())、
# thread_config = {"configurable": {"thread_id": "1"}}、invoke(..., thread_config)。
# 三行里藏着两件事：checkpointer 决定"存不存"，thread_id 决定"存到哪个抽屉"。
print("  三种写法，同一个问题问两遍，看模型第二次收到几条消息：\n")

# ① 有 checkpointer + 同一个 thread_id
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
same_thread = {"configurable": {"thread_id": "1"}}
for text in ("我叫 Bob。", "我叫什么？"):
    state = agent.invoke({"messages": [{"role": "user", "content": text}]}, same_thread)
print(f"  ① checkpointer + 同一个 thread_id")
print(f"     两次模型看到的条数 = {[len(x) for x in probe.seen]}")
print(f"     第二轮结束 state  = {message_line(state['messages'])}")
print(f"     快照里的 messages = {len(agent.get_state(same_thread).values['messages'])} 条（get_state 按 thread 取）")

# ② 有 checkpointer，但换了 thread_id
probe = MemoryProbe()
agent = create_agent(probe, checkpointer=InMemorySaver())
for index, text in enumerate(("我叫 Bob。", "我叫什么？"), start=1):
    agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": str(index)}},
    )
print(f"\n  ② 同一个 saver，换了 thread_id")
print(f"     两次模型看到的条数 = {[len(x) for x in probe.seen]}")

# ③ 完全没有 checkpointer
probe = MemoryProbe()
agent = create_agent(probe)
for text in ("我叫 Bob。", "我叫什么？"):
    agent.invoke({"messages": [{"role": "user", "content": text}]})
print(f"\n  ③ 没有 checkpointer")
print(f"     两次模型看到的条数 = {[len(x) for x in probe.seen]}")

print("\n  [1, 3] vs [1, 1] 就是这一节的全部：")
print("    · 没有 checkpointer → 每次 invoke 都是全新 state，上一轮的东西没地方可存")
print("    · 有 checkpointer 但换了 thread_id → 存下来了，只是存进了另一个抽屉")
print("    · 所以'记不住'永远先查这两样，别急着怀疑模型或 prompt")
print("\n  两个容易漏的细节：")
print("    · 第二轮只需要传**新消息**，旧消息由 checkpointer 恢复后走 add_messages 合并")
print("    · InMemorySaver 进程一退就没；生产换 PostgresSaver（官方 In production 那节）")
print("      —— 那一节的 with ... as checkpointer / checkpointer.setup() 是建表，本机没有 pg，不演示")

# --------------------------------------------------------------------------
banner("B", "Customizing memory：state_schema 是替换，所以必须继承")
# --------------------------------------------------------------------------


class SupportState(AgentState[Any]):
    """官方写法：继承 AgentState，只写新字段。

    注意**不重声明 messages** —— 父类那句 Annotated[..., add_messages] 会被继承下来，
    reducer 因此还在。这一行写不写 messages，是"记得"和"失忆"的分界线。
    """

    user_id: NotRequired[str]
    plan: NotRequired[str]


probe = MemoryProbe()
agent = create_agent(probe, state_schema=SupportState, checkpointer=InMemorySaver())
support_thread = {"configurable": {"thread_id": "u-42"}}
first = agent.invoke(
    {"messages": [{"role": "user", "content": "我是 u_42，帮我看看套餐"}], "user_id": "u_42"},
    support_thread,
)
second = agent.invoke({"messages": [{"role": "user", "content": "我是谁？"}]}, support_thread)

print(f"  第一轮传了 user_id，state 的 key = {sorted(first.keys())}")
print(f"  第二轮**只传 messages**，user_id 仍 = {second.get('user_id')!r}")
print("    ↑ 自定义字段和 messages 一样会被 checkpointer 存下来，不需要每轮重传")

print(f"\n  自定义 state 的 input  schema = {list(agent.get_input_jsonschema()['properties'])}")
print(f"  自定义 state 的 output schema = {list(agent.get_output_jsonschema()['properties'])}")
print(f"  默认 AgentState 的 output   = {list(create_agent(MemoryProbe()).get_output_jsonschema()['properties'])}")


class NakedState(TypedDict):
    """反例：不继承、裸 TypedDict。能跑，但图上只剩你写的字段。"""

    messages: list[Any]
    user_id: str


naked = create_agent(MemoryProbe(), state_schema=NakedState)
print(f"  裸 TypedDict 的 output     = {list(naked.get_output_jsonschema()['properties'])}")
print("\n  差别只在两个词上：")
print("    · 继承 → messages / jump_to / structured_response 一起带过来，你只做加法")
print("    · 裸写 → 你的类就是全部，没写 structured_response 它就不存在，第 7 章会直接失效")
print("  官方原话是 'extend the agent's state'，但 factory.py 里是 base_state = state_schema or AgentState：")
print("  传了就是**替换**。所以 'extend' 的正确姿势是 class MyState(AgentState[Any])，不是裸 TypedDict。")

# --------------------------------------------------------------------------
banner("C", "Common patterns · Trim：before_model + REMOVE_ALL_MESSAGES")
# --------------------------------------------------------------------------


@before_model
def trim_messages(state: AgentState, runtime) -> dict[str, Any] | None:  # noqa: ANN001
    """官方写法：留下第一条 + 最近几条，其余全丢。

    返回 None 表示"不改"；返回 {"messages": [...]} 会走 add_messages 合并。
    REMOVE_ALL_MESSAGES 是个哨兵：它之前的内容全丢，它之后的内容成为新历史。
    """
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    first_message = messages[0]
    recent = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first_message, *recent]}


probe = MemoryProbe()
agent = create_agent(probe, middleware=[trim_messages], checkpointer=InMemorySaver())
trim_thread = {"configurable": {"thread_id": "trim"}}
for turn in range(4):
    state = agent.invoke({"messages": [{"role": "user", "content": f"第{turn}轮：我叫 Bob"}]}, trim_thread)
    print(f"  turn{turn}  模型看到 {len(probe.seen[-1])} 条  |  state {len(state['messages'])} 条  {message_line(state['messages'])}")

print("\n  读法（对着上面四行看）：")
print("    · turn0/1 还没越过 3 条，直接 return None，什么都没发生")
print("    · turn2 那一次 5 进 5 出（没真正剪），但 state 已经涨到 6")
print("    · turn3 起进入稳态：模型永远看到 5 条，state 永远 6 条，不再涨")
print("    · 第一条**永远留着** —— 官方这个例子的意图就是'开场设定不能丢'")
print("\n  一个官方例子没说的副作用，看 turn3 那行的第二个元素：")
print("    留下来的 AI:答2 是回答'第1轮'的，可'第1轮'这半句已经被切掉了。")
print("    按**条数**切就会切出这种'没有问题的答案'。它不违反消息格式（仍是 H/A 交替），")
print("    但会让模型看到一段没头没尾的历史。要严谨就按**问答对**切，或者切完断言首条是 HumanMessage。")
print("\n  为什么用 REMOVE_ALL_MESSAGES 而不是逐条 RemoveMessage：")
print("    · 逐条要自己算 id 列表，还容易删出'AI 有 tool_calls 但 ToolMessage 被删'的非法历史")
print("    · 哨兵 + 重建列表是一次原子替换，你给什么就是什么")
print("    · 代价：裁剪发生在**每次模型调用前**，所以它救的是上下文，不改持久化的历史")

# --------------------------------------------------------------------------
banner("D", "Common patterns · Delete：RemoveMessage 删一条 / 全清")
# --------------------------------------------------------------------------


@after_model
def delete_oldest_two(state: AgentState, runtime) -> dict[str, Any] | None:  # noqa: ANN001
    """官方写法：每轮删掉最早两条（一问一答算一对）。"""
    messages = state["messages"]
    if len(messages) <= 2:
        return None
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


probe = MemoryProbe()
agent = create_agent(probe, middleware=[delete_oldest_two], checkpointer=InMemorySaver())
forget_thread = {"configurable": {"thread_id": "forget"}}
for turn in range(3):
    state = agent.invoke({"messages": [{"role": "user", "content": f"第{turn}轮"}]}, forget_thread)
    print(f"  turn{turn}  state {len(state['messages'])} 条  {message_line(state['messages'])}")
print("\n  ↑ after_model 里删的是**真历史**：state 稳定在 2 条，第 0 轮被彻底抹掉。")
print("    所以官方那个例子里问 'what's my name?' 会得到 'I don't know your name' —— 不是模型笨，")
print("    是那句话真的不在 state 里了。这和 C 段（只裁上下文、历史还在）是两件事。")


@after_model
def clear_all(state: AgentState, runtime) -> dict[str, Any]:  # noqa: ANN001
    """全清：只留一个哨兵，后面什么都没有。"""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


probe = MemoryProbe()
agent = create_agent(probe, middleware=[clear_all], checkpointer=InMemorySaver())
state = agent.invoke({"messages": [{"role": "user", "content": "随便说点什么"}]}, {"configurable": {"thread_id": "wipe"}})
print(f"\n  全清之后 state 的 messages = {state['messages']}（连这一轮的回复也没了）")
print("  这就是'重置上下文'的官方按钮：REMOVE_ALL_MESSAGES 不是'删某条'，是'从这里重新开始'。")

print("\n  三条边界（官方特意写了一段注意事项）：")
print("    · 删不存在的 id → ValueError，不是静默忽略（所以先确认它还在）")
print("    · 删完要保证历史**合法**：某些 provider 要求首条是 user；AI 带 tool_calls 时，")
print("      对应的 ToolMessage 不能单独删掉，否则下一次请求直接被拒")
print("    · 想改内容而不是删：用**同一个 id** 再写一遍，add_messages 会原地替换（05_state_reducer B2）")

# --------------------------------------------------------------------------
banner("E", "Common patterns · Summarize：SummarizationMiddleware")
# --------------------------------------------------------------------------
# 官方这一段给的是一行配置：SummarizationMiddleware(model=..., trigger=..., keep=...)。
# 本机把 trigger 从 ("tokens", 4000) / ("messages", 20) 调小，才能在四轮里看到它触发。
summarizer = MemoryProbe(echo="（摘要）用户叫 Bob，一直在问天气。")
probe = MemoryProbe()
agent = create_agent(
    probe,
    middleware=[
        SummarizationMiddleware(
            model=summarizer,
            trigger=("messages", 5),
            keep=("messages", 2),
        )
    ],
    checkpointer=InMemorySaver(),
)
summary_thread = {"configurable": {"thread_id": "summary"}}
for turn in range(4):
    state = agent.invoke({"messages": [{"role": "user", "content": f"我叫 Bob，第{turn}轮"}]}, summary_thread)

print(f"  摘要模型被调用了 {len(summarizer.seen)} 次（不是每轮都调：只在触发条件满足时）")
print(f"  最终 state {len(state['messages'])} 条：")
for message in state["messages"]:
    print(f"    {type(message).__name__:14s} {short(message.content, 90)}")

print("\n  三个参数要先搞清：")
print("    · trigger：什么时候摘要。('messages', 5) 数条数，('tokens', 4000) 估 token，('fraction', 0.8) 按窗口比例")
print("    · keep：摘要时保留多少原文。('messages', 2) 保留最近 2 条")
print("    · model：**用哪个模型写摘要**。官方示例特意用小模型（gpt-5.4-mini），因为这是纯体力活")
print("\n  摘要和裁剪的选择，一句话：")
print("    · trim/delete 是**丢**，丢掉的再也回不来；摘要多花一次模型调用，换'细节压缩成结论'")
print("    · 结论性的信息（我叫 Bob）用摘要；过程性噪音（工具回了一大坨 JSON）用 clear tool uses / 直接删")
print("    · 摘要会**改写历史**：那条摘要以 HumanMessage 形式排在开头，之后模型看到的就是'摘要 + 最近几条'")

# --------------------------------------------------------------------------
banner("F", "Access memory · Tools：runtime.state 读，Command 写")
# --------------------------------------------------------------------------


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """读取当前用户的画像。

    runtime 是保留参数名：模型看不见、也填不了，框架自己注入。
    """
    user_id = runtime.state.get("user_id") or "匿名用户"
    plan = runtime.state.get("plan") or "free"
    return f"user_id={user_id}, plan={plan}"


@tool
def set_plan(plan: str, runtime: ToolRuntime) -> Command:
    """修改套餐。

    官方硬约束：工具改 state 必须返回 Command(update=...)，
    而且 update 里**必须**带 ToolMessage，tool_call_id 用 runtime.tool_call_id，
    否则 ToolNode 直接 ValueError —— 是框架报错，不是模型报错。
    """
    return Command(
        update={
            "plan": plan,
            "messages": [ToolMessage(f"套餐已改为 {plan}", tool_call_id=runtime.tool_call_id)],
        }
    )


probe = MemoryProbe(
    script=[
        AIMessage("", tool_calls=[{"name": "get_user_info", "args": {}, "id": "call_read"}]),
        AIMessage("", tool_calls=[{"name": "set_plan", "args": {"plan": "pro"}, "id": "call_write"}]),
        AIMessage("已办好。"),
    ]
)
agent = create_agent(probe, tools=[get_user_info, set_plan], state_schema=SupportState)
state = agent.invoke(
    {"messages": [{"role": "user", "content": "我在什么套餐？帮我升到 pro"}], "user_id": "u_42"},
)

print(f"  工具列表 = {[t.name for t in (get_user_info, set_plan)]}")
print(f"  两个工具的 schema 参数 = {[list(t.tool_call_schema.model_json_schema()['properties']) for t in (get_user_info, set_plan)]}")
print("    ↑ get_user_info 一个业务参数都没有，但 runtime 不进 schema —— 模型看不到它")
print("\n  跑完一轮之后的 messages：")
for index, message in enumerate(state["messages"]):
    if isinstance(message, ToolMessage):
        extra = f" tool_call_id={message.tool_call_id}"
    elif getattr(message, "tool_calls", None):
        extra = f" ← 只有 tool_calls，content 是空的：{[c['name'] for c in message.tool_calls]}"
    else:
        extra = ""
    print(f"    [{index}] {type(message).__name__:13s} {short(message.content, 40)}{extra}")
print(f"\n  最终 state['plan'] = {state['plan']!r}   ← 工具用 Command 写进去的")
print("  注意 [2] 里还是 plan=free：读在前、写在后，工具看到的是**执行那一刻**的 state。")
print("  同一轮里两个工具改同一个字段，谁后跑谁赢 —— 它们不是事务，别指望顺序无关。")

print("\n  读写两条路，别混：")
print("    · 读：runtime.state['x']（同步拿到当前 state 快照）")
print("    · 写：return Command(update={...})，并且自己补 ToolMessage")
print("    · 只想返回一段文本给模型看，就直接 return 字符串 / dict —— 那不会改 state")
print("    · 工具里改 state 是一次**状态更新**，同样会过 reducer，所以 messages 是追加不是覆盖")

# --------------------------------------------------------------------------
banner("G", "Access memory · Prompt：@dynamic_prompt + ModelRequest")
# --------------------------------------------------------------------------


class UserContext(TypedDict):
    """每次运行传入的不可变配置（context），和 state 是两回事。"""

    user_name: str


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """每次调用模型前现算 system prompt。request 里能拿到 state、context、messages。"""
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Address the user as {user_name}."


probe = MemoryProbe()
agent = create_agent(probe, middleware=[dynamic_system_prompt], context_schema=UserContext)
agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    context={"user_name": "张三"},
)
first_seen = probe.seen[0]
print(f"  模型收到的第一条 = {type(first_seen[0]).__name__}")
print(f"  内容 = {first_seen[0].content!r}")
print("\n  三个概念划清（官方这一节最容易糊的地方）：")
print("    · state    —— 会涨、会被 checkpointer 存下来的东西（messages、自定义字段）")
print("    · context  —— 每次 invoke 传一次的不可变配置（用户身份、租户、权限），不进 state、不持久化")
print("    · prompt   —— 由前两者现算出来的字符串，只在这次调用有效")
print("  所以'按用户改称呼'该用 context + @dynamic_prompt，而不是把 user_name 塞进 messages。")

# --------------------------------------------------------------------------
banner("H", "真跑一次：同一个 thread 两轮 + 自定义 state（--real）")
# --------------------------------------------------------------------------
if not real_enabled():
    skip_hint(SCRIPT)
    print("  它会证明三件事：同 thread 第二轮真记得上一轮；自定义字段跨轮还在；")
    print("  trim 之后 state 不再无限增长。结构上 H 段和 A/B/C 完全一样，只是把假模型换成真的。")
else:
    real_agent = create_agent(
        get_model(),
        state_schema=SupportState,
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )
    real_thread = {"configurable": {"thread_id": "real-1"}}

    print("  第 1 轮：我叫小林，在做 LangChain 教材。")
    first_state = real_agent.invoke(
        {"messages": [{"role": "user", "content": "我叫小林，在做 LangChain 教材。一句话回我。"}], "user_id": "u_lin"},
        real_thread,
    )
    print(f"    AI: {short(first_state['messages'][-1].content, 80)}")

    print("\n  第 2 轮（只传新消息）：我叫什么？")
    second_state = real_agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？一句话。"}]},
        real_thread,
    )
    print(f"    AI: {short(second_state['messages'][-1].content, 80)}")

    print(f"\n  state 里 messages = {len(second_state['messages'])} 条，user_id = {second_state.get('user_id')!r}")
    for index, message in enumerate(second_state["messages"]):
        print(f"    [{index}] {type(message).__name__:13s} {short(message.content, 50)}")
    print("\n  上面能答出'小林'，靠的就是 A 段那个 checkpointer + 同一个 thread_id。")
    print("  把 thread_id 换成 'real-2' 再问一次，它就不认识了 —— 这是最省事的验证方式。")
