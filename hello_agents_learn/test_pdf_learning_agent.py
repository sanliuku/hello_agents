# -*- coding: utf-8 -*-
"""PDFLearningAssistant 测试脚本

覆盖：
1. 加载文档（真实生成一个临时 PDF 走 MarkItDown 解析）
2. 文档问答
3. 学习笔记 / 记忆回顾
4. 学习统计
5. 生成学习报告
6. 清理测试数据

运行：
    python hello_agents_learn/test_pdf_learning_agent.py

测试数据统一存放在 hello_agents_learn/memory_test_data/ 下：
    pdf/      测试 PDF
    faiss/    向量库
    sqlite/   记忆数据库 memory.db
    *.json    学习报告
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

LEARN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(LEARN_DIR)

# 保证从任意工作目录都能 import 到 hello_agents_learn
sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from hello_agents_learn.agents.pdf_learning_agent import PDFLearningAssistant
from hello_agents_learn.memory import MemoryConfig

DATA_DIR = os.path.join(LEARN_DIR, "memory_test_data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")
SQLITE_DIR = os.path.join(DATA_DIR, "sqlite")

TEST_USER_ID = f"test_pdf_agent_{int(time.time())}"

PDF_SECTIONS = [
    ("HelloAgents 框架简介", [
        "HelloAgents 是一个面向学习场景的多智能体框架。",
        "框架由记忆模块、RAG 模块和工具模块三部分组成。",
    ]),
    ("记忆模块", [
        "记忆模块包含工作记忆、情景记忆和语义记忆三种类型。",
        "语义记忆使用 Neo4j 图数据库存储实体与关系。",
        "向量检索使用本地 Faiss 索引，嵌入模型为 ollama bge-m3。",
    ]),
    ("RAG 模块", [
        "RAG 模块负责文档解析、分块、向量化与检索。",
        "文档解析基于 MarkItDown，支持 PDF、Word、Excel 等格式。",
        "检索支持多查询扩展 MQE 与 HyDE 假想文档两种增强方式。",
    ]),
]


def build_test_pdf(pdf_path: str) -> None:
    """用 reportlab 生成一个中文测试 PDF（内置 STSong-Light 字体）"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 72

    c.setFont("STSong-Light", 18)
    c.drawString(72, y, "HelloAgents 学习手册")
    y -= 36

    for title, lines in PDF_SECTIONS:
        c.setFont("STSong-Light", 14)
        c.drawString(72, y, title)
        y -= 22
        c.setFont("STSong-Light", 11)
        for line in lines:
            c.drawString(72, y, line)
            y -= 18
        y -= 10
        if y < 100:
            c.showPage()
            y = height - 72
            c.setFont("STSong-Light", 11)

    c.save()


def main() -> int:
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(SQLITE_DIR, exist_ok=True)

    # 记忆模块的向量库通过环境变量指定落盘目录，保证测试数据都在 memory_test_data 下
    os.environ["FAISS_PERSIST_DIR"] = FAISS_DIR

    pdf_path = os.path.join(PDF_DIR, "hello_agents_manual.pdf")
    report_path = None
    failures = []

    def check(name: str, ok: bool, detail: str = ""):
        print(f"{'✅' if ok else '❌'} {name}" + (f" | {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 60)
    print("PDFLearningAssistant 测试开始")
    print(f"数据目录: {DATA_DIR}")
    print(f"测试用户: {TEST_USER_ID}")
    print("=" * 60)

    # ---------- 0. 准备测试 PDF ----------
    build_test_pdf(pdf_path)
    check("生成测试 PDF", os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0,
          f"{os.path.getsize(pdf_path)} bytes")

    # ---------- 1. 初始化助手 ----------
    assistant = PDFLearningAssistant(
        user_id=TEST_USER_ID,
        memory_config=MemoryConfig(storage_path=SQLITE_DIR),
        persist_dir=FAISS_DIR,
    )
    check("初始化助手", assistant is not None,
          f"session_id={assistant.session_id}")

    # ---------- 2. 加载文档 ----------
    load_result = assistant.load_document(pdf_path)
    print(f"   load_document -> {load_result}")
    check("加载文档", bool(load_result.get("success")),
          load_result.get("message", ""))
    check("当前文档记录", assistant.current_document == os.path.basename(pdf_path),
          str(assistant.current_document))

    # ---------- 3. 文档问答 ----------
    if load_result.get("success"):
        question = "HelloAgents 包含哪些模块？语义记忆用什么存储？"
        answer = assistant.ask(question, use_advanced_search=False)
        print(f"   ask -> {answer[:500]}")
        llm_unavailable = "LLM 不可用" in answer
        check("文档问答返回内容", bool(answer and answer.strip()))
        if llm_unavailable:
            print("   ⚠️ 当前环境 LLM 不可用，已跳过答案质量校验")
        else:
            check("问答引用知识库内容", ("模块" in answer) or ("Neo4j" in answer))
    else:
        check("文档问答", False, "文档加载失败，跳过问答")

    # ---------- 4. 学习笔记与记忆回顾 ----------
    assistant.add_note("HelloAgents 的记忆模块分三类，语义记忆用 Neo4j。", concept="记忆模块")
    check("添加学习笔记", assistant.stats["concepts_learned"] == 1,
          f"concepts_learned={assistant.stats['concepts_learned']}")

    recall_result = assistant.recall("语义记忆 存储", limit=5)
    print(f"   recall -> {recall_result[:300]}")
    check("记忆回顾", bool(recall_result and "失败" not in recall_result))

    # ---------- 5. 学习统计 ----------
    stats = assistant.get_stats()
    print("   get_stats ->")
    for k, v in stats.items():
        print(f"     {k}: {v}")
    check("统计-加载文档数", stats["加载文档"] == 1)
    check("统计-提问次数", stats["提问次数"] == (1 if load_result.get("success") else 0))
    check("统计-当前文档", stats["当前文档"] == os.path.basename(pdf_path))

    # ---------- 6. 生成学习报告 ----------
    # generate_report 按当前工作目录写文件，这里切到数据目录，报告也落在 memory_test_data
    original_cwd = os.getcwd()
    try:
        os.chdir(DATA_DIR)
        report = assistant.generate_report(save_to_file=True)
    finally:
        os.chdir(original_cwd)
    report_path = (
        os.path.join(DATA_DIR, report["report_file"]) if report.get("report_file") else None
    )
    check("生成学习报告", bool(report_path) and os.path.exists(report_path), str(report_path))
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("报告内容完整",
              set(["session_info", "learning_metrics", "memory_summary", "rag_status"]).issubset(saved.keys()))

    # ---------- 7. 清理测试数据 ----------
    # 只删除本次测试文档对应的分块，避免影响集合内其他数据
    try:
        store = assistant.rag_tool._get_pipeline().get("store")
        ids_to_delete = [
            sid for sid, meta in store._meta.items()
            if meta.get("source_path") == pdf_path
        ]
        if ids_to_delete:
            store.delete_vectors(ids_to_delete)
        check("清理文档向量", not any(
            meta.get("source_path") == pdf_path for meta in store._meta.values()
        ), f"删除 {len(ids_to_delete)} 条")
    except Exception as e:
        check("清理文档向量", False, str(e))

    # 注意：语义记忆的 Faiss/Neo4j 存储不是按 user_id 隔离的，
    # 调用 memory_tool 的 clear_all 会连带清空其他数据，所以这里不做清空。
    print("ℹ️ 测试记忆保留在用户 %s 名下（不执行 clear_all，避免误删共享语义记忆）" % TEST_USER_ID)
    print(f"ℹ️ 测试产物保留在: {DATA_DIR}")
    print(f"   测试 PDF: {pdf_path}")
    if report_path:
        print(f"   学习报告: {report_path}")

    print("=" * 60)
    if failures:
        print(f"❌ 测试结束，{len(failures)} 项失败: {', '.join(failures)}")
        return 1
    print("✅ 全部测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
