"""上下文管理（本地实现）

对照 hello_agents.context.builder 重写 GSSC（Gather-Select-Structure-Compress）流程，
主要修正两处不合理的规则：

1. 相关性计算：原实现用 `text.lower().split()` 按空格切词，中文整句会被当成一个
   token，导致中文 query 与任何中文内容的重叠恒为 0，所有候选包都被 min_relevance
   过滤掉。这里改为「英文按词 + 中文按二元组（bigram）」的混合分词，并在计算前剔除
   疑问词（如何/什么/怎么…），使覆盖率能反映真实重合程度。
2. 筛选策略：字面重合度只是「词面信号」，记忆/RAG 的结果是嵌入检索回来的，词面
   重合度天然偏低（例如「如何优化 Pandas 的内存占用」与「用户在使用 Pandas 做数据
   分析」几乎无字面交集），用 min_relevance 硬过滤会把有效证据全部误杀。因此改为：
   - 检索类包（task_state / related_memory / knowledge_base 等）给相关性保底，
     不再参与硬过滤，只按分数排序；
   - 对话历史与系统指令同样不参与硬过滤（历史即上下文）；
   - min_relevance 只作用于调用方通过 additional_packets 塞进来的候选包。

对外接口与原版保持一致：ContextBuilder / ContextConfig / ContextPacket / count_tokens。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import re

import tiktoken

from ..tool.memory_tool import MemoryTool
from ..tool.rag_tool import RAGTool


# tiktoken 的编码器初始化较慢，复用同一个实例
try:
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # 离线等异常场景下降级为字符估算
    _ENCODING = None

# 英文单词（长度 >= 2 才算，避免 a/an/of 之类噪声）
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
# 连续中文片段
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
# 中文疑问/语气词：不携带检索信息，参与分词会稀释覆盖率
_CJK_QUESTION_RE = re.compile(
    r"如何|怎么|怎样|怎么样|什么|为什么|为何|哪些|哪个|哪儿|是否|多少|请问|帮我|请|能否|可以|一下|吗|呢|吧"
)

_EN_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "this", "that", "what",
    "which", "how", "why", "when", "where", "from", "into", "about", "does",
}

# 检索类包：由记忆/RAG 侧按语义相似度召回，词面重合低不代表不相关
_RETRIEVAL_PRIOR = {
    "task_state": 0.8,
    "related_memory": 0.6,
    "knowledge_base": 0.6,
    "retrieval": 0.6,
    "tool_result": 0.6,
}
# 不参与 min_relevance 硬过滤的包类型
_NEVER_FILTER_TYPES = {"instructions", "history"} | set(_RETRIEVAL_PRIOR)


def tokenize(text: str) -> Set[str]:
    """相关性计算用的分词（中英混合）

    - 英文/数字：按单词切分
    - 中文：按相邻两字（bigram）切分，单字段落保留单字

    用 bigram 而不是单字，是因为「语义记忆」这类词拆成「语」「义」「记」「忆」
    会被大量无关文本命中，bigram 能保留词序信息、区分度更高。
    """
    if not text:
        return set()

    lowered = _CJK_QUESTION_RE.sub(" ", text.lower())
    tokens: Set[str] = set()

    for word in _WORD_RE.findall(lowered):
        if word not in _EN_STOPWORDS:
            tokens.add(word)

    for segment in _CJK_RE.findall(lowered):
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(segment[i:i + 2] for i in range(len(segment) - 1))

    return tokens


def count_tokens(text: str) -> int:
    """计算文本 token 数（tiktoken，失败时按 1 token ≈ 4 字符估算）"""
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    return len(text) // 4


def compute_relevance(query: str, content: str) -> float:
    """关键词覆盖率：query 的 token 有多少出现在 content 中（0.0-1.0）"""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    overlap = query_tokens & tokenize(content)
    return len(overlap) / len(query_tokens)


@dataclass
class ContextPacket:
    """上下文信息包"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0

    def __post_init__(self):
        """自动计算 token 数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 8000  # 总预算
    reserve_ratio: float = 0.15  # 生成余量（10-20%）
    min_relevance: float = 0.3  # 最小相关性阈值（仅过滤 additional_packets）
    enable_mmr: bool = True  # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7  # MMR 平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = ""  # 系统提示模板
    enable_compression: bool = True  # 启用压缩

    def get_available_tokens(self) -> int:
        """获取可用 token 预算（扣除余量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """上下文构建器 - GSSC 流水线

    用法示例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """

    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Any]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """构建完整上下文"""
        # 1. Gather: 收集候选信息
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )

        # 2. Select: 筛选与排序
        selected_packets = self._select(packets, user_query)

        # 3. Structure: 组织成结构化模板
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )

        # 4. Compress: 压缩与规范化（如果超预算）
        return self._compress(structured_context)

    def _gather(
        self,
        user_query: str,
        conversation_history: List[Any],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: 收集候选信息"""
        packets: List[ContextPacket] = []

        # P0: 系统指令（强约束）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                state_results = self.memory_tool.execute(
                    "search",
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))

                related_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")

        # P2: 从 RAG 中获取事实证据（本地 RAGTool 的检索参数名是 limit）
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")

        # P3: 对话历史（辅助材料）
        if conversation_history:
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))

        packets.extend(additional_packets)

        # 去重：任务状态检索与联动记忆检索、RAG 之间可能返回同一段内容，
        # 不去重会在 [State] 与 [Evidence] 里重复占预算。
        deduped: List[ContextPacket] = []
        seen_content: Set[str] = set()
        for p in packets:
            key = p.content.strip()
            if not key or key in seen_content:
                continue
            seen_content.add(key)
            deduped.append(p)
        return deduped

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: 基于分数与预算的筛选"""
        # 1) 计算相关性：词面覆盖率 + 检索来源保底
        for packet in packets:
            lexical = compute_relevance(user_query, packet.content)
            packet.metadata["lexical_relevance"] = round(lexical, 3)
            packet.relevance_score = max(
                lexical,
                _RETRIEVAL_PRIOR.get(packet.metadata.get("type"), 0.0)
            )

        # 2) 计算新近性（指数衰减）
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1 小时时间尺度
            return math.exp(-delta / tau)

        # 3) 计算复合分：0.7*相关性 + 0.3*新近性
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            score = 0.7 * p.relevance_score + 0.3 * recency_score(p.timestamp)
            scored_packets.append((score, p))

        # 4) 系统指令单独拿出，固定纳入
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [
            p for (_, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
            if p.metadata.get("type") != "instructions"
        ]

        # 5) 依据 min_relevance 过滤
        #    系统指令/对话历史/检索结果不参与过滤：前两者本身就是上下文，
        #    后者已由记忆与 RAG 侧按语义相似度召回，词面重合度低不代表不相关。
        filtered = [
            p for p in remaining
            if p.metadata.get("type") in _NEVER_FILTER_TYPES
            or p.relevance_score >= self.config.min_relevance
        ]

        # 6) 按预算填充
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0

        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count

        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count

        return selected

    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections = []

        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)

        sections.append(f"[Task]\n用户问题：{user_query}")

        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)

        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)

        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)

        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议（如适用）"""
        sections.append(output_section)

        return "\n\n".join(sections)

    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()

        if current_tokens <= available_tokens:
            return context

        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")

        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens

        return "\n".join(compressed_lines)
