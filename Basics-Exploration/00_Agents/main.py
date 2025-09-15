# Import necessary modules
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv
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


# Main async function to run the agent
async def main():
    # ? Agent with context Management

    agent = Agent(
        name="Assistant",
        instructions="Simple Ai assistant",
        model=gemini_model,
    )

    # Run the agent with a sample prompt
    result = await Runner.run(agent, "What is the capital of France?")
    print(result.final_output)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
