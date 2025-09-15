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


booking_agent = Agent(
    name="Booking Agent",
    instructions="You are a helpful travel booking assistant. Help users book flights and hotels.",
    model=gemini_model,
)
refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a customer service agent specializing in processing refunds. Assist users with their refund requests.",
    model=gemini_model,
)


# Main async function to run the agent
async def main():
    agent = Agent(
        name="Customer-facing agent",
        instructions=(
            "Handle all direct user communication. "
            "Call the relevant tools when specialized expertise is needed."
        ),
        model=gemini_model,
        tools=[
            booking_agent.as_tool(
                tool_name="Booking_expert",
                tool_description="Expert in booking flights and hotels",
            ),
            refund_agent.as_tool(
                tool_name="Refund_expert",
                tool_description="Expert in processing refunds",
            ),
        ],
    )

    # Run the agent with a sample prompt
    result = await Runner.run(agent, "What is the capital of France?")
    print(result.final_output)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
