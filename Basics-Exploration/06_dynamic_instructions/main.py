import os
import asyncio
from dotenv import load_dotenv
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunContextWrapper,
)
from openai import AsyncOpenAI
from agents.run import Runner
from pydantic import BaseModel


# Disable tracing (optional, for debugging/tracing tools)
set_tracing_disabled(True)

# Load environment variables from .env
load_dotenv()
GEMINI_API = os.getenv("GEMINI_API_KEY")
if not GEMINI_API:
    raise ValueError("Gemini API key not found")

# Set up Gemini client
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Initialize Gemini model wrapper
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


async def main():

    class UserContext(BaseModel):
        name: str
        is_premium: bool

    def dynamic_instructions(context: RunContextWrapper[UserContext], agent):
        user_name = context.context.name
        if context.context.is_premium:
            return f"Hello {user_name}! You are a premium user. Give detailed premium support."
        else:
            return f"Hello {user_name}! Provide standard support."

    # Define your agent
    agent = Agent[UserContext](
        name="Support provide",
        instructions=dynamic_instructions,
        model=gemini_model,
    )

    # Run agent with a prompt
    premium_user = UserContext(name="Alice", is_premium=True)

    free_user = UserContext(name="Bob", is_premium=False)

    # Run with different contexts
    result1 = await Runner.run(
        agent, "What support do I have access to?", context=premium_user
    )
    print("--- Premium User ---")
    print(result1.final_output)

    result2 = await Runner.run(
        agent, "What support do I have access to?", context=free_user
    )
    print("--- Free User ---")
    print(result2.final_output)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
