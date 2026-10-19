# test_reflection_agent.py
from dotenv import load_dotenv
from hello_agents import CalculatorTool, HelloAgentsLLM, ToolRegistry
from hello_agents import ReActAgent

load_dotenv()
llm = HelloAgentsLLM()

tool_registry = ToolRegistry()
calculator = CalculatorTool()
tool_registry.register_tool(calculator)

general_agent = ReActAgent(name="ReAct助手", llm=llm, tool_registry=tool_registry)

# 测试使用
result = general_agent.run("计算123加456")
print(f"最终结果: {result}")
