"""第 6 章 · 最小可运行的持久化（只用标准库，不依赖 langgraph）

跑法：python langchain_learn/06_lite_checkpoint.py

它只做三件事，对应真实 checkpointer 的三件事：
    save   目录存"第几步 / 下一步 / 父档 / 版本表"，内容按 (channel, version) 存
    load   照版本表把内容取回来（直接读，不重算）
    head   取最新一档
两轮对话跑完，你能看到：一次 invoke 5 张档、快照数 > 内容份数、改档会另起分支。
"""

import copy

MISSING = object()


class CheckpointStore:
    """内存版存档：catalog 管时序，values 管内容。"""

    def __init__(self):
        self.catalog = {}   # (thread_id, checkpoint_id) -> {step, next, parent, versions}
        self.values = {}    # (thread_id, channel, version) -> value
        self._seq = 0

    def save(self, thread_id, state, *, step, next_node, parent=None, channels=("messages",)):
        self._seq += 1
        checkpoint_id = f"c{self._seq:02d}"
        versions = dict(self.catalog[(thread_id, parent)]["versions"]) if parent else {}
        for channel in channels:
            current = versions.get(channel, 0)
            previous = self.values.get((thread_id, channel, current), MISSING)
            if previous is not MISSING and previous == state[channel]:
                continue                                    # 值没变 → 复用旧版本
            versions[channel] = current + 1
            self.values[(thread_id, channel, versions[channel])] = copy.deepcopy(state[channel])
        self.catalog[(thread_id, checkpoint_id)] = {
            "step": step, "next": next_node, "parent": parent, "versions": versions,
        }
        return checkpoint_id

    def head(self, thread_id):
        ids = [cid for (tid, cid) in self.catalog if tid == thread_id]
        return ids[-1] if ids else None

    def load(self, thread_id, checkpoint_id=None):
        checkpoint_id = checkpoint_id or self.head(thread_id)
        if not checkpoint_id:
            return {}
        versions = self.catalog[(thread_id, checkpoint_id)]["versions"]
        return {ch: copy.deepcopy(self.values[(thread_id, ch, ver)]) for ch, ver in versions.items()}

    def history(self, thread_id):
        return [(cid, meta) for (tid, cid), meta in reversed(self.catalog.items()) if tid == thread_id]


def fake_model(messages):
    """假模型：第一回去调工具，第二回给答案（答案里带条数，用来证明它看见了历史）。"""
    if not any(msg["role"] == "tool" for msg in messages):
        return {"role": "assistant", "tool": "weather", "args": {"city": "北京"}}
    asked = [msg["text"] for msg in messages if msg["role"] == "user"]
    return {"role": "assistant", "text": f"你一共说了 {len(asked)} 句，第一句是「{asked[0]}」"}


def run_turn(store, thread_id, text):
    state = store.load(thread_id) or {"messages": []}       # 先取档
    state["messages"].append({"role": "user", "text": text})

    # 入口帧：父档是上一轮的收尾档，版本从它接着往上数
    parent = store.save(thread_id, state, step=-1, next_node="model",
                        parent=store.head(thread_id))
    step = 0
    while True:
        last = state["messages"][-1]
        if last["role"] == "assistant" and "text" in last:
            break
        parent = store.save(thread_id, state, step=step, parent=parent,
                            next_node="tools" if "tool" in last else "model")   # 每个边界一张
        if "tool" in last:
            state["messages"].append({"role": "tool", "text": f"{last['args']['city']}：晴 26℃"})
        else:
            state["messages"].append(fake_model(state["messages"]))
        step += 1

    # 收尾帧：next 为空，消息又变了，所以还是要写新版本
    store.save(thread_id, state, step=step, next_node=None, parent=parent)
    return state


def main():
    store = CheckpointStore()
    for text in ("我叫小王", "我叫什么？"):
        state = run_turn(store, "u-001", text)
        print(f"用户：{text}\n助手：{state['messages'][-1]['text']}\n")

    print("目录（新→旧）：")
    for checkpoint_id, meta in store.history("u-001"):
        print(f"  {checkpoint_id}  step={meta['step']:>2}  next={str(meta['next']):<6} "
              f"父={meta['parent']}  版本表={meta['versions']}")

    frames = len(store.catalog)
    payloads = len([key for key in store.values if key[1] == "messages"])
    print(f"\n快照 {frames} 张，messages 只存了 {payloads} 份 —— 没变的版本被复用\n")

    target = [cid for cid, meta in store.history("u-001") if meta["next"] is None][0]
    state = store.load("u-001", target)
    state["messages"][-1]["text"] = "（我改过的回答）"
    forked = store.save("u-001", state, step=99, next_node=None, parent=target)
    print(f"从 {target} 改档 → 新分支 {forked}；"
          f"原分支还在：{store.load('u-001', target)['messages'][-1]['text']}；"
          f"新分支头：{store.load('u-001')['messages'][-1]['text']}")


if __name__ == "__main__":
    main()
