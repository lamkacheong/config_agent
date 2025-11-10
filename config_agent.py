from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent


async def create_config_agent():
    # 创建主 Agent
    model = ChatOpenAI(
        model="glm-4.6",
        temperature=0,
    )

    main_agent = create_deep_agent(
        model=model,
    )
    return main_agent
