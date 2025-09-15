import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from agents.run import Runner
from pydantic import BaseModel, Field
from typing import Annotated

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
    # Define your agent

    class CalenderEvent(BaseModel):
        title: str
        time: str
        location: Annotated[str | None, Field(description="Event location")] = None

    agent = Agent(
        name="Calendar extractor",
        instructions="Extract calendar events from text. Location is optional if not provided return None",
        model=gemini_model,
        output_type=CalenderEvent,
    )

    event_text = """
Hey team! Just wanted to confirm our upcoming project kickoff meeting. 
It's scheduled for next Tuesday, March 12th from 10:00 AM to 11:30 AM PST.
We'll be discussing the Q2 roadmap and assigning initial tasks.
Please come prepared with your team's priorities and resource availability.

Meeting Title: Q2 Project Kickoff
Location: Virtual (Zoom link to follow)
"""
    result = await Runner.run(agent, event_text)
    print(result.final_output)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
