# 🔹 Size Estimate (English)

# 100 tokens ≈ 75 words (average)

# 1 token ≈ 4 characters (English average)

# So:

# 1,000 tokens ≈ 750 words ≈ ~1 page essay

# 4,000 tokens ≈ 3,000 words ≈ ~4–5 pages

# 🔹 Real Examples
# Text	Token Count	Note
# "Hi"	1	single short word
# "Hello world"	2	simple words
# "I'm learning OpenAI Agents SDK"	6	contractions + capital words split
# "Pakistan's economy is growing slowly."	8	punctuation + split words
# ⚡ Shortcut Formula
# 1 token ≈ 4 characters (English)
# 1 token ≈ 0.75 words


# 👉 Ab clear hai: ek token fixed size nahi hota, wo text ke fragments hote hain jo tokenizer banata hai.

# ? To calculate tokens there is a library name as **tiktoken**

# Import necessary modules
from ast import Store
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ModelSettings,
)
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
    agent = Agent(
        name="Assistant",
        instructions="Simple Ai assistant",
        model=gemini_model,
        model_settings=ModelSettings(include_usage=True, Store=True),
    )

    # Run the agent with a sample prompt
    result = await Runner.run(agent, "What is the capital of France?")
    # print(result.final_output)
    print(result.raw_responses[0].id)
    print(result.raw_responses[0].usage)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
