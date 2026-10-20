from dotenv import load_dotenv
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
import os
import sys

# 把仓库根目录加入 sys.path，确保可按完整包路径导入本地 memory 包（hello_agents_learn.memory）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "memory_test_data")
os.environ.setdefault("FAISS_PERSIST_DIR", os.path.join(_DATA_DIR, "faiss"))

from hello_agents_learn.tool.memory_tool import MemoryTool
from hello_agents_learn.memory.base import MemoryConfig
from hello_agents_learn.memory.manager import MemoryManager


# 创建LLM实例
load_dotenv()
llm = HelloAgentsLLM()

# 创建记忆工具
model_config = MemoryConfig(storage_path=_DATA_DIR)
memory_tool = MemoryTool(user_id="user123", memory_config=model_config)
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)

# 创建具有记忆能力的Agent
agent = SimpleAgent(name="记忆助手", llm=llm, tool_registry=tool_registry)
 
# 体验记忆功能
print("=== 添加多个记忆 ===")

# 1. 工作记忆 - 临时信息，容量有限
result1 = memory_tool.run({
    "action": "add",
    "content": "用户刚才问了关于Python函数的问题",
    "memory_type": "working",
    "importance": 0.6,
}
)
print(result1)

# 2. 情景记忆 - 具体事件和经历
result2 = memory_tool.run({
    "action": "add",
    "content": "2024年3月15日，用户张三完成了第一个Python项目",
    "memory_type": "episodic",
    "importance": 0.8,
    "event_type": "milestone",
    "location": "在线学习平台",
}
)
print(result2)

# 3. 语义记忆 - 抽象知识和概念
result3 = memory_tool.run({
    "action": "add",
    "content": "Python是一种解释型、面向对象的编程语言",
    "memory_type": "semantic",
    "importance": 0.9,
    "knowledge_type": "factual"
}
)
print(result3)

# 4. 感知记忆 - 多模态信息
# result4 = memory_tool.run({
#     "action": "add",
#     "content": "用户上传了一张Python代码截图，包含函数定义",
#     "memory_type": "perceptual",
#     "importance": 0.7,
#     "modality": "image",
#     "file_path": "./uploads/code_screenshot.png"
# }
# )
# print(result4)
