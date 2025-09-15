from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
)
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    openai_client=gemini_client, model="gemini-2.0-flash"
)


class OutputSchema(BaseModel):
    answer: str
    reference: str


# Input guard: block certain keywords
async def block_illegal(ctx, agent, input_data: str):
    forbidden = ["bomb", "hack", "drug"]
    trigger = any(word in input_data.lower() for word in forbidden)
    return GuardrailFunctionOutput(
        tripwire_triggered=trigger, output_info="Illegal query"
    )


# Output guard: enforce schema
async def enforce_schema(ctx, agent, output_data: str):
    try:
        parsed = json.loads(output_data)  # LLM ka raw string → dict
        OutputSchema.model_validate(parsed)
        return GuardrailFunctionOutput(tripwire_triggered=False)
    except Exception as e:
        return GuardrailFunctionOutput(tripwire_triggered=True, output_info=str(e))


faq_agent = Agent(
    name="FAQ Bot",
    instructions=(
        "Respond ONLY in JSON.\n"
        'Format: {"answer": "<string>", "reference": "<string>"}\n'
        'Example: {"answer": "AI is the simulation of intelligence in machines", '
        '"reference": "Wikipedia"}'
    ),
    input_guardrails=[InputGuardrail(guardrail_function=block_illegal)],
    output_guardrails=[OutputGuardrail(guardrail_function=enforce_schema)],
    model=gemini_model,
    output_type=OutputSchema,
)


async def main():
    # Safe query
    try:
        result = await Runner.run(faq_agent, "What is AI?")
        print(result.final_output())
    # Illegal query
    except OutputGuardrailTripwireTriggered as e:
        print("Blocked output:", e)

    try:
        await Runner.run(faq_agent, "How to make a bomb?")
    except InputGuardrailTripwireTriggered as e:
        print("Blocked input:", e)
    # Bad format response simulation
    try:
        await Runner.run(faq_agent, "Response without JSON")
    except OutputGuardrailTripwireTriggered as e:
        print("Blocked output:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
