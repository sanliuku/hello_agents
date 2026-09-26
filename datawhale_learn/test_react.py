from datawhale_learn.tools.duck_search import search
from datawhale_learn.llms.ollama_client import HelloAgentsLLM
from datawhale_learn.agents.react_agent import ReActAgent
from datawhale_learn.tools.tool_base import ToolExecutor

llm = HelloAgentsLLM()
toolExecutor = ToolExecutor()
search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
toolExecutor.registerTool("Search", search_description, search)
agent = ReActAgent(llm, toolExecutor)
agent.run("deepseek harness是什么？")
