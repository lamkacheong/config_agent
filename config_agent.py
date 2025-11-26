from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from context import MainContext
from langgraph.runtime import Runtime


async def create_config_agent(runtime: Runtime[MainContext]):
    configurable = runtime.get("configurable", {})
    prompt = configurable.get("system_prompt", "You are a helpful AI assistant.")

    # 创建主 Agent
    model = ChatOpenAI(
        model="glm-4.6",
        temperature=0,
    )

    main_agent = create_deep_agent(
        model=model,
        system_prompt=prompt,
        context_schema=MainContext,
    )
    return main_agent
