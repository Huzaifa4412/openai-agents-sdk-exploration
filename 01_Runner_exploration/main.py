import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from config import GEMINI_API_KEY
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from agents.run import Runner
from dotenv import load_dotenv
import asyncio

load_dotenv(ROOT_DIR / ".env")

set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

coder_agent = Agent(
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


def sync_demo():
    print("\n🔹 SYNC DEMO (Sequential Execution)\n")
    result1 = Runner.run_sync(
        coder_agent, "Write a function to sort an array using python."
    )
    print(f"Coder: {result1.final_output}\n")
    result2 = Runner.run_sync(maths_agent, "Solve for x: 2x + 3 = 7")
    print(f"Maths: {result2.final_output}\n")
    result3 = Runner.run_sync(
        history_agent, "Who was the first president of the Pakistan?"
    )
    print(f"History: {result3.final_output}\n")


async def async_demo():
    print("\n🔹 ASYNC DEMO (Parallel Execution)\n")

    task1 = Runner.run(coder_agent, "Write a function to sort an array using python.")
    task2 = Runner.run(maths_agent, "Solve for x: 2x + 3 = 7")
    task3 = Runner.run(history_agent, "Who was the first president of the Pakistan?")

    results = await asyncio.gather(task1, task2, task3)

    print(f"Coder: {results[0].final_output}\n")
    print(f"Maths: {results[1].final_output}\n")
    print(f"History: {results[2].final_output}\n")


def main():
    sync_demo()
    asyncio.run(async_demo())


if __name__ == "__main__":
    main()
