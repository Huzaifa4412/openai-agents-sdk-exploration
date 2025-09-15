from agents import Agent, FunctionToolResult, Runner, StopAtTools, function_tool
from agents.agent import ToolsToFinalOutputResult, RunContextWrapper
from typing import List, Any
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()

set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


async def main():
    #     # ? What is Meta Data ->
    #     Metadata refers to auxiliary context passed alongside agent runs. In the SDK, metadata lives within the RunContextWrapper and powers context-aware behavior, logging, tracing, and guardrail decisions.

    # What it is

    # Carried in RunContextWrapper.metadata

    # Key–value pairs: e.g. run_id, session_id, user_id, timestamp, feature flags, test mode.

    # Available to tools, guardrails, hooks, custom handlers.

    # Why it matters

    # Traceability: Logs of run origin, session tracking.

    # Context-aware logic: Agents or tools adapt behavior based on metadata (e.g. test mode).

    # Guardrail validation: Can depend on metadata (e.g. is user pro).

    # Observability: Tracing systems tap metadata for instrumentation.

    # ✅ Summary

    # Metadata = “extra info about the run, not the main input.”

    # Stored in context (dict-like).

    # Use cases: tracing, logging, user personalization, guardrails, custom handlers.

    # ? Custom Tool handling

    # Tool
    @function_tool
    def get_weather(city: str) -> str:
        """Returns weather info for the specified city."""
        return f"The weather in {city} is rain"

    def custom_tool_handler(
        context: RunContextWrapper[Any], tool_results: List[FunctionToolResult]
    ) -> ToolsToFinalOutputResult:
        """Processes tool results to decide final output."""
        for result in tool_results:
            if result.output and "sunny" in result.output:
                return ToolsToFinalOutputResult(
                    is_final_output=True, final_output=f"Final weather: {result.output}"
                )
        return ToolsToFinalOutputResult(is_final_output=False, final_output=None)

    agent = Agent(
        name="Encrypted Weather Agent",
        instructions="Retrieve weather but don't guess unless handler allows.",
        tools=[get_weather],
        tool_use_behavior=custom_tool_handler,
        model=gemini_model,
    )

    result = await Runner.run(
        agent,
        "What is the weather in New York?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
