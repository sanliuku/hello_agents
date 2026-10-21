"""RAG 工具（本地版）

与 hello_agents.tools.builtin.rag_tool.RAGTool 接口一致，底层替换为：
- 向量库：本地 Faiss（memory.storage.faiss_store）
- 嵌入：统一嵌入提供器（默认 ollama bge-m3）
- LLM：HelloAgentsLLM（用于问答与查询扩展）

使用示例：
```python
rag = RAGTool()
rag.run({"action": "add_text", "text": "..."})
rag.run({"action": "ask", "question": "..."})
```
"""

from typing import Any, Dict, List, Optional
import hashlib
import os
import time

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.core.llm import HelloAgentsLLM

from ..memory.rag.pipeline import create_rag_pipeline


class RAGTool(Tool):
    """RAG 工具：本地 Faiss 向量库 + ollama bge 嵌入"""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "hello_agents_rag_vectors",
        rag_namespace: str = "default",
        persist_dir: Optional[str] = None,
        llm=None,
    ):
        super().__init__(
            name="rag",
            description="RAG工具 - 支持多格式文档检索增强生成，提供智能问答能力",
        )

        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self.persist_dir = persist_dir
        self.llm = llm
        self._pipelines: Dict[str, Dict[str, Any]] = {}

        os.makedirs(knowledge_base_path, exist_ok=True)
        self._init_components()

    def _init_components(self):
        """初始化管道与 LLM（LLM 失败不影响检索）"""
        try:
            self._pipelines[self.rag_namespace] = create_rag_pipeline(
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace,
                persist_dir=self.persist_dir,
                llm=self._get_llm(),
            )
            self.initialized = True
            print(f"✅ RAG工具初始化成功: namespace={self.rag_namespace}, collection={self.collection_name}")
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"❌ RAG工具初始化失败: {e}")

    def _get_llm(self):
        """懒加载 LLM，失败时返回 None（检索仍可用）"""
        if self.llm is not None:
            return self.llm
        try:
            self.llm = HelloAgentsLLM()
        except Exception as e:
            print(f"⚠️ LLM 初始化失败，问答功能不可用: {e}")
            self.llm = None
        return self.llm

    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间的管道，不存在则创建"""
        target_ns = namespace or self.rag_namespace
        if target_ns not in self._pipelines:
            self._pipelines[target_ns] = create_rag_pipeline(
                collection_name=self.collection_name,
                rag_namespace=target_ns,
                persist_dir=self.persist_dir,
                llm=self._get_llm(),
            )
        return self._pipelines[target_ns]

    # ==========================
    # Tool 接口
    # ==========================

    def run(self, parameters: Dict[str, Any]) -> str:
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        action = parameters.get("action")
        kwargs = {k: v for k, v in parameters.items() if k != "action"}
        return self.execute(action, **kwargs)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), stats(统计), clear(清空)",
                required=True,
            ),
            ToolParameter(name="file_path", type="string", description="文档文件路径（PDF/Word/Excel/PPT/图片/音频等）", required=False),
            ToolParameter(name="text", type="string", description="要添加的文本内容", required=False),
            ToolParameter(name="question", type="string", description="用户问题（用于智能问答）", required=False),
            ToolParameter(name="query", type="string", description="搜索查询词（用于基础搜索）", required=False),
            ToolParameter(name="namespace", type="string", description="知识库命名空间（默认：default）", required=False, default="default"),
            ToolParameter(name="limit", type="integer", description="返回结果数量（默认：5）", required=False, default=5),
            ToolParameter(name="include_citations", type="boolean", description="是否包含引用来源（默认：true）", required=False, default=True),
        ]

    def execute(self, action: str, **kwargs) -> str:
        if not self.initialized:
            return f"❌ RAG工具未正确初始化，请检查配置: {getattr(self, 'init_error', '未知错误')}"

        try:
            kwargs = self._preprocess_parameters(action, **kwargs)
        except ValueError as e:
            return f"❌ {e}"

        try:
            if action == "add_document":
                return self._add_document(**kwargs)
            elif action == "add_text":
                return self._add_text(**kwargs)
            elif action == "ask":
                return self._ask(**kwargs)
            elif action == "search":
                return self._search(**kwargs)
            elif action == "stats":
                return self._get_stats(namespace=kwargs.get("namespace"))
            elif action == "clear":
                return self._clear_knowledge_base(**kwargs)
            else:
                available = ["add_document", "add_text", "ask", "search", "stats", "clear"]
                return f"❌ 不支持的操作: {action}\n✅ 可用操作: {', '.join(available)}"
        except Exception as e:
            return f"❌ 执行操作 '{action}' 时发生错误: {str(e)}"

    def _preprocess_parameters(self, action: str, **kwargs) -> Dict[str, Any]:
        """补默认值并做必填校验"""
        defaults = {
            "namespace": self.rag_namespace,
            "limit": 5,
            "include_citations": True,
            "enable_advanced_search": True,
            "max_chars": 1200,
            "min_score": 0.1,
            "chunk_size": 800,
            "chunk_overlap": 100,
        }
        for key, value in defaults.items():
            if kwargs.get(key) is None:
                kwargs[key] = value

        if action == "add_document" and not kwargs.get("file_path"):
            raise ValueError("add_document 操作需要提供 file_path 参数")
        if action == "add_text" and not kwargs.get("text"):
            raise ValueError("add_text 操作需要提供 text 参数")
        if action == "ask" and not (kwargs.get("question") or kwargs.get("query")):
            raise ValueError("ask 操作需要提供 question 或 query 参数")
        if action == "search" and not (kwargs.get("query") or kwargs.get("question")):
            raise ValueError("search 操作需要提供 query 或 question 参数")
        return kwargs

    # ==========================
    # 具体操作
    # ==========================

    def _add_document(self, file_path: str, namespace: Optional[str] = None, chunk_size: int = 800, chunk_overlap: int = 100, **kwargs) -> str:
        try:
            if not file_path or not os.path.exists(file_path):
                return f"❌ 文件不存在: {file_path}"

            pipeline = self._get_pipeline(namespace)
            t0 = time.time()
            chunks_added = pipeline["add_documents"](
                file_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            process_ms = int((time.time() - t0) * 1000)

            if chunks_added == 0:
                return f"⚠️ 未能从文件解析内容: {os.path.basename(file_path)}"

            return (
                f"✅ 文档已添加到知识库: {os.path.basename(file_path)}\n"
                f"📊 分块数量: {chunks_added}\n"
                f"⏱️ 处理时间: {process_ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}"
            )
        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"

    def _add_text(self, text: str, document_id: str = None, namespace: Optional[str] = None, chunk_size: int = 800, chunk_overlap: int = 100, **kwargs) -> str:
        try:
            if not text or not text.strip():
                return "❌ 文本内容不能为空"

            # 用内容哈希做文档ID，保证同一文本重复添加时覆盖而不是堆积
            # 用 md5 而非内置 hash：后者每进程随机加盐，跨进程不稳定
            document_id = document_id or "text_" + hashlib.md5(
                text.encode("utf-8")
            ).hexdigest()[:12]
            tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")

            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(text)

                pipeline = self._get_pipeline(namespace)
                t0 = time.time()
                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                process_ms = int((time.time() - t0) * 1000)

                if chunks_added == 0:
                    return "⚠️ 未能从文本生成有效分块"

                return (
                    f"✅ 文本已添加到知识库: {document_id}\n"
                    f"📊 分块数量: {chunks_added}\n"
                    f"⏱️ 处理时间: {process_ms}ms\n"
                    f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}"
                )
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"

    def _search(self, query: str, limit: int = 5, min_score: float = 0.1, enable_advanced_search: bool = True, include_citations: bool = True, namespace: Optional[str] = None, **kwargs) -> str:
        try:
            if not query or not query.strip():
                return "❌ 搜索查询不能为空"

            pipeline = self._get_pipeline(namespace)
            threshold = min_score if min_score > 0 else None

            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=query,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                    score_threshold=threshold,
                )
            else:
                results = pipeline["search"](
                    query=query,
                    top_k=limit,
                    score_threshold=threshold,
                )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的内容"

            lines = ["搜索结果："]
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                score = result.get("score", 0.0)
                content = (meta.get("content", "") or "")[:200]
                source = meta.get("source_path", "unknown")

                lines.append(f"\n{i}. 文档: **{source}** (相似度: {score:.3f})")
                lines.append(f"   {content}")
                if include_citations and meta.get("heading_path"):
                    lines.append(f"   章节: {meta['heading_path']}")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

    def _ask(self, question: Optional[str] = None, query: Optional[str] = None, limit: int = 5, enable_advanced_search: bool = True, include_citations: bool = True, max_chars: int = 1200, namespace: Optional[str] = None, **kwargs) -> str:
        try:
            user_question = (question or query or "").strip()
            if not user_question:
                return "❌ 请提供要询问的问题"

            llm = self._get_llm()
            if llm is None:
                return "❌ LLM 不可用，无法生成答案（可使用 search 操作只做检索）"

            pipeline = self._get_pipeline(namespace)
            t0 = time.time()
            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=user_question, top_k=limit, enable_mqe=True, enable_hyde=True
                )
            else:
                results = pipeline["search"](query=user_question, top_k=limit)
            search_ms = int((time.time() - t0) * 1000)

            if not results:
                return (
                    f"🤔 抱歉，我在知识库中没有找到与「{user_question}」相关的信息。\n\n"
                    f"💡 建议：\n• 尝试使用更简洁的关键词\n• 检查是否已添加相关文档\n• 使用 stats 操作查看知识库状态"
                )

            context_parts: List[str] = []
            citations: List[Dict[str, Any]] = []
            total_score = 0.0
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                content = (meta.get("content", "") or "").strip()
                score = float(result.get("score", 0.0))
                total_score += score
                if not content:
                    continue
                content = " ".join(content.split())
                if len(content) > 300:
                    content = content[:300] + "..."
                context_parts.append(f"片段 {i}：{content}")
                if include_citations:
                    citations.append({
                        "index": i,
                        "source": os.path.basename(meta.get("source_path", "unknown")),
                        "score": score,
                    })

            context = "\n\n".join(context_parts)
            if len(context) > max_chars:
                context = context[:max_chars] + "...[内容被截断]"

            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(user_question, context)},
            ]
            t1 = time.time()
            answer = llm.invoke(messages)
            llm_ms = int((time.time() - t1) * 1000)

            if not answer or not answer.strip():
                return "❌ LLM未能生成有效答案，请稍后重试"

            lines = ["🤖 **智能问答结果**\n", answer.strip()]
            if include_citations and citations:
                lines.append("\n\n📚 **参考来源**")
                for c in citations:
                    emoji = "🟢" if c["score"] > 0.8 else "🟡" if c["score"] > 0.6 else "🔵"
                    lines.append(f"{emoji} [{c['index']}] {c['source']} (相似度: {c['score']:.3f})")
            avg_score = total_score / len(results) if results else 0.0
            lines.append(f"\n⚡ 检索: {search_ms}ms | 生成: {llm_ms}ms | 平均相似度: {avg_score:.3f}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 智能问答失败: {str(e)}\n💡 请检查知识库状态或稍后重试"

    def _build_system_prompt(self) -> str:
        return (
            "你是一个专业的知识助手，具备以下能力：\n"
            "1. 📖 精准理解：仔细理解用户问题的核心意图\n"
            "2. 🎯 可信回答：严格基于提供的上下文信息回答，不编造内容\n"
            "3. 🔍 信息整合：从多个片段中提取关键信息，形成完整答案\n"
            "4. 💡 清晰表达：用简洁明了的语言回答，适当使用结构化格式\n"
            "5. 🚫 诚实表达：如果上下文不足以回答问题，请坦诚说明\n\n"
            "回答格式要求：\n• 直接回答核心问题\n• 必要时使用要点或步骤\n• 引用关键原文时使用引号\n• 避免重复和冗余"
        )

    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"请基于以下上下文信息回答问题：\n\n"
            f"【问题】{question}\n\n"
            f"【相关上下文】\n{context}\n\n"
            f"【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"
        )

    def _get_stats(self, namespace: Optional[str] = None) -> str:
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]() or {}
            info = [
                "📊 **RAG 知识库统计**",
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}",
                f"📋 集合名称: {self.collection_name}",
                f"📂 存储根路径: {stats.get('persist_dir', self.persist_dir or './memory_data/faiss')}",
                f"📦 存储类型: {stats.get('store_type', 'faiss')}",
                f"📊 当前命名空间分块数: {stats.get('namespace_chunks', 0)}",
                f"📊 集合总分块数: {int(stats.get('points_count') or 0)}",
            ]
            config = stats.get("config") or {}
            if isinstance(config, dict):
                info.append(f"🔢 向量维度: {config.get('vector_size', 'unknown')}")
                info.append(f"📎 距离度量: {config.get('distance', 'unknown')}")
            info.extend([
                "",
                "🟢 **系统状态**",
                f"✅ RAG 管道: {'正常' if self.initialized else '异常'}",
                f"✅ LLM 连接: {'正常' if self.llm is not None else '不可用'}",
            ])
            return "\n".join(info)
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def _clear_knowledge_base(self, confirm: bool = False, namespace: Optional[str] = None, **kwargs) -> str:
        """清空知识库

        注意：Faiss 一个集合内的所有命名空间共用同一索引文件，
        因此这里清空的是整个集合（与 store.clear_collection 的语义一致）。
        """
        try:
            if not confirm:
                return (
                    "⚠️ 危险操作：清空知识库将删除所有数据！\n"
                    "请使用 confirm=true 参数确认执行。"
                )

            target_ns = namespace or self.rag_namespace
            pipeline = self._get_pipeline(target_ns)
            store = pipeline.get("store")
            success = store.clear_collection() if store else False

            if success:
                self._pipelines[target_ns] = create_rag_pipeline(
                    collection_name=self.collection_name,
                    rag_namespace=target_ns,
                    persist_dir=self.persist_dir,
                    llm=self._get_llm(),
                )
                return f"✅ 知识库已成功清空（集合：{self.collection_name}）"
            return "❌ 清空知识库失败"
        except Exception as e:
            return f"❌ 清空知识库失败: {str(e)}"

    # ==========================
    # 便捷方法
    # ==========================

    def add_document(self, file_path: str, namespace: str = None) -> str:
        return self.run({"action": "add_document", "file_path": file_path, "namespace": namespace})

    def add_text(self, text: str, namespace: str = None, document_id: str = None) -> str:
        return self.run({"action": "add_text", "text": text, "namespace": namespace, "document_id": document_id})

    def ask(self, question: str, namespace: str = None, **kwargs) -> str:
        params = {"action": "ask", "question": question, "namespace": namespace}
        params.update(kwargs)
        return self.run(params)

    def search(self, query: str, namespace: str = None, **kwargs) -> str:
        params = {"action": "search", "query": query, "namespace": namespace}
        params.update(kwargs)
        return self.run(params)

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        """供 Agent 调用的上下文获取接口"""
        try:
            if not query:
                return ""
            results = self._get_pipeline(namespace)["search"](query=query, top_k=limit)
            parts = [r.get("metadata", {}).get("content", "") for r in results]
            merged = "\n\n".join([p for p in parts if p])
            if len(merged) > max_chars:
                merged = merged[:max_chars] + "..."
            return merged
        except Exception as e:
            return f"获取上下文失败: {str(e)}"
