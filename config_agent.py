from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

from context import MainContext
from langgraph.runtime import Runtime
from tools.get_todays_date import get_todays_date
from tools.internet_search import internet_search

# 读取当前目录的 .env 文件
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# 工具映射表
TOOL_MAP = {
    "get_todays_date": get_todays_date,
    "internet_search": internet_search,
}


async def create_config_agent(runtime: Runtime[MainContext]):
    configurable = runtime.get("configurable", {})
    prompt = configurable.get("system_prompt", "You are a helpful AI assistant.")
    selected_tools = configurable.get("selected_tools", ["get_todays_date"])

    # 根据配置动态加载工具
    tools = [TOOL_MAP[tool_name] for tool_name in selected_tools if tool_name in TOOL_MAP]

    # 创建主 Agent
    # model = ChatOpenAI(
    #     model="glm-4.6",
    #     temperature=0,
    # )
    model = init_chat_model("claude-sonnet-4-5-20250929")

    main_agent = create_deep_agent(
        model=model,
        system_prompt=prompt,
        context_schema=MainContext,
        tools=tools,
    )
    return main_agent
