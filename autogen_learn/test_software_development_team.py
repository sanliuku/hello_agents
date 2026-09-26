from autogen_learn.llms.llm_client import create_openai_model_client
from autogen_learn.teams.code_team import software_development_team
from autogen_agentchat.ui import Console

import asyncio


async def run_software_development_team():
    # ... 初始化客户端和智能体 ...
    llm_client = create_openai_model_client()
    team_chat = software_development_team(llm_client)
    
    # 定义任务描述
    task = """我们需要开发一个美国大米价格显示应用，具体要求如下：
            核心功能：
            - 实时显示美国大米当前价格（美元/吨）
            - 显示24小时价格变化趋势（涨跌幅和涨跌额）
            - 提供价格刷新功能

            技术要求：
            - 使用 Streamlit 框架创建 Web 应用
            - 界面简洁美观，用户友好
            - 添加适当的错误处理和加载状态

            注意：
            - 搜索引擎使用DuckDuckGo
            - 用"http://127.0.0.1:7897"代理，确保能够正常访问互联网

            请团队协作完成这个任务，从需求分析到最终实现。"""
    
    # 异步执行团队协作，并流式输出对话过程
    result = await Console(team_chat.run_stream(task=task))
    return result

# 主程序入口
if __name__ == "__main__":
    result = asyncio.run(run_software_development_team())
