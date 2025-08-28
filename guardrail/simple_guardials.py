# Import necessary modules
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
)
from agents.run import Runner
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
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


class HomeWorkCheck(BaseModel):
    is_homework: bool
    reason: str


# ? Input Guardrail


async def my_guardrail(ctx, agent, input_data):
    flag = "solve my homework" in input_data.lower()

    return GuardrailFunctionOutput(
        tripwire_triggered=flag, output_info=f"Homework request: {flag}"
    )


# ? Output Guardrail
async def block_email_output_guardrail(ctx, agent, output_data):
    if "@" in output_data:
        return GuardrailFunctionOutput(
            tripwire_triggered=True, output_info="Output contains an email address"
        )
    return GuardrailFunctionOutput(tripwire_triggered=False)


# Main async function to run the agent
async def main():
    agent = Agent(
        name="Tutor",
        instructions="Help but don't solve homework",
        input_guardrails=[InputGuardrail(guardrail_function=my_guardrail)],
        output_guardrails=[
            OutputGuardrail(guardrail_function=block_email_output_guardrail)
        ],
        model=gemini_model,
    )

    # Run the agent with a sample prompt
    try:
        result = await Runner.run(agent, "Give me email as huzaifa@gmail.com")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Input Guardrail blocked: ", e)

    except OutputGuardrailTripwireTriggered as e:
        print("Output Guardrail blocked: ", e)

    except Exception as e:
        print("Some other error occurred:", e)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
