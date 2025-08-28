import os
import asyncio
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

set_tracing_disabled(True)
load_dotenv()

GEMINI_API = os.getenv("GEMINI_API_KEY")
if not GEMINI_API:
    raise ValueError("Gemini API key not found")

gemini_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client,
)


# 🧠 User ka context define
class UserContext(TypedDict):
    name: str
    level: str
    interest: str


# 🔀 Dynamic instructions function
def dynamic_instructions(
    ctx: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    user = ctx.context
    return (
        f"User ka naam {user['name']} hai. "
        f"Unka level {user['level']} hai aur unko {user['interest']} pasand hai. "
        f"Unhain {user['level']} style mein samjhao aur examples {user['interest']} se related do."
    )


# 🎓 Personalized Tutor Agent
agent = Agent[UserContext](
    name="Tutor Agent", instructions=dynamic_instructions, model=gemini_model
)


# 🔁 Test with real user context
async def main():
    context = {"name": "Huzaifa", "level": "beginner", "interest": "web development"}

    result = await Runner.run(agent, "what is my name", context=context)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
