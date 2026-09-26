from datawhale_learn.agents.reflection_agent import ReflectionAgent
from datawhale_learn.llms.ollama_client import HelloAgentsLLM


llm = HelloAgentsLLM()
agent = ReflectionAgent(llm_client=llm)
agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")