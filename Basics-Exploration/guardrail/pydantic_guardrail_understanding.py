# Import necessary modules
from agents import Agent, OpenAIChatCompletionsModel, GuardrailFunctionOutput
from agents.run import Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel

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


class UserRequest(BaseModel):
    question: str
    age: int
    answer: str


async def age_guardrail(ctx, agent, input_data):
    if input_data.age < 18:
        return GuardrailFunctionOutput(
            tripwire_triggered=True, output_info="User is under 18"
        )
    return GuardrailFunctionOutput(tripwire_triggered=False)


# Main async function to run the agent
async def main():
    agent = Agent(
        name="Assistant",
        instructions="Simple Ai assistant",
        model=gemini_model,
        output_type=UserRequest,
    )

    # Run the agent with a sample prompt
    result = await Runner.run(agent, "What is the capital of France?")
    print(result.final_output)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
