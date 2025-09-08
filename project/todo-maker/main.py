# Import necessary modules
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tools import add_task, list_tasks, complete_task
import os
import asyncio

set_tracing_disabled(True)
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Initialize Gemini client using OpenAI-compatible interface
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model using Gemini
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


# def add_task(task):
#     pass


# Main async function to run the agent
async def main():
    while True:
        print("\n--- TODO APP ---")
        print("\n Press")
        print("1. Add a task")
        print("2. Show all tasks")
        print("3. Complete a task")
        print("4. Delete a task")
        print("5. Update a task")
        print("\n Press 0 to exit or type exit for exit")

        choice = input("\n Enter your choice: ")
        if choice == "0" or choice.lower() == "exit":
            break
        elif choice == "1":
            add_task()


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
