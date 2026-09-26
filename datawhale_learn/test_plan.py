from datawhale_learn.agents.planner_agent import PlanAndSolveAgent
from datawhale_learn.llms.ollama_client import HelloAgentsLLM

llm = HelloAgentsLLM()
agent = PlanAndSolveAgent(llm_client=llm)

agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
