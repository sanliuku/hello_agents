from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from roles.product_manager import create_product_manager
from roles.engineer import create_engineer
from roles.code_reviewer import create_code_reviewer
from roles.user_proxy import create_user_proxy

def software_development_team(llm_client):
    product_manager = create_product_manager(llm_client)
    engineer = create_engineer(llm_client)
    code_reviewer = create_code_reviewer(llm_client)
    user_proxy = create_user_proxy()
    team_chat = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            user_proxy
        ],
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=20,
    )
    return team_chat
