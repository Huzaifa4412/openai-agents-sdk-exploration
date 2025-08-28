import os, asyncio
from dotenv import load_dotenv
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    RunContextWrapper,
)
from typing import TypedDict
from openai import AsyncOpenAI

# Disable tracing
set_tracing_disabled(True)

# Load env
load_dotenv()
GEMINI_API = os.getenv("GEMINI_API_KEY")
if not GEMINI_API:
    raise ValueError("Gemini API key not found")

# Gemini setup
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client,
)


# Define context type
class UserContext(TypedDict):
    name: str


# Dynamic instructions function
def dynamic_instructions(
    ctx: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"User ka naam hai {ctx.context['name']}. Coding mein help karo."


# Agent creation
agent = Agent[UserContext](
    name="Dynamic Gemini Agent", instructions=dynamic_instructions, model=gemini_model
)


# Main run
async def main():
    context = {"name": "Huzaifa"}
    result = await Runner.run(
        agent, "Python decorators kya hotay hain?", context=context
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
