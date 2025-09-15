import os
import asyncio
from agents.run import Runner
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
from agents import enable_verbose_stdout_logging

# ? For debugging
enable_verbose_stdout_logging()
# from config import GEMINI_API_KEY

# import sys
# from pathlib import Path

# ROOT_FILE = Path(__file__).resolve().parents[1]
# sys.path.append(str(ROOT_FILE))

set_tracing_disabled(True)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found!")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=external_client
)

coding_agent = Agent(
    name="coder",
    instructions="You are a helpful coding assistant. Your task is to write code for me.",
    model=model,
)

maths_agent = Agent(
    name="maths",
    instructions="You are a helpful maths assistant. Your task is to solve maths problems for me.",
    model=model,
)

history_agent = Agent(
    name="history",
    instructions="You are a helpful history assistant. Your task is to answer history questions for me.",
    model=model,
)


async def main():
    triage_agent = Agent(
        name="Triage Agent",
        instructions="You are a helpful assistant. Your task is to triage the user's request and route it to the appropriate agent.",
        handoffs=[coding_agent, maths_agent, history_agent],
        model=model,
    )
    result = await Runner.run(triage_agent, input="What is the my previous question")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
