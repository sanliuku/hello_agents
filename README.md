# hello_agents

AI Agent 学习与实践项目，通过动手实现来理解智能体的核心概念（LLM 调用、Planner、ReAct、Reflection、记忆、工具使用、多智能体协作）。

## 目录结构

```
hello_agents/
├── datawhale_learn/   # 跟随《Hello Agents》教程从零实现 Agent 核心组件
├── autogen_learn/     # 基于 Microsoft AutoGen 框架的多智能体实践
├── agentscope_learn/  # （规划中）AgentScope 框架学习
└── docs/              # 学习笔记
```

## 模块说明

### datawhale_learn

从零搭建 Agent 的基础组件，不依赖框架：

- `llms/ollama_client.py` — 兼容 OpenAI 接口的 LLM 客户端，默认流式响应
- `agents/` — 三种经典 Agent 范式：
  - `planner_agent.py` — 计划执行（Plan-and-Execute）
  - `react_agent.py` — 推理 + 行动循环（ReAct）
  - `reflection_agent.py` — 自我反思迭代优化（Reflection）
- `memorys/` — Agent 记忆模块
- `tools/` — 工具基类与 DuckDuckGo 搜索工具

运行测试脚本：

```bash
python -m datawhale_learn.test_react
```

### autogen_learn

基于 AutoGen（autogen-agentchat）搭建一个软件开发团队：

- `roles/` — 产品经理、工程师、代码评审员、用户代理四个角色
- `teams/code_team.py` — RoundRobinGroupChat 轮询协作，以 `TERMINATE` 结束对话
- `test_software_development_team.py` — 端到端演示

```bash
python autogen_learn/test_software_development_team.py
```

## 环境配置

1. 安装依赖（建议使用 `.venv` 虚拟环境）：

```bash
pip install openai python-dotenv autogen-agentchat
```

2. 在项目根目录创建 `.env` 文件，配置任意兼容 OpenAI 接口的 LLM 服务：

```env
LLM_API_KEY=你的密钥
LLM_MODEL_ID=模型名称
LLM_BASE_URL=服务地址
```
