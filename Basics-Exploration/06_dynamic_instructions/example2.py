import os
import asyncio
from dotenv import load_dotenv
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    enable_verbose_stdout_logging,
    RunContextWrapper,
)
from openai import AsyncOpenAI
from agents.run import Runner
from dataclasses import dataclass

# Debugging
# enable_verbose_stdout_logging()

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
    model="gemini-2.5-flash", openai_client=gemini_client
)


async def main():
    # context
    @dataclass
    class UserContext:
        name: str
        is_premium: bool

    @function_tool
    def support_user(context: RunContextWrapper[UserContext]) -> str:
        """Support the user based on their account type.

        Returns:
            String: Support message for both free and premium user
        """

        if not context.context.is_premium:
            return f"Hello, {context.name}! Bro please go and buy premium version."
        elif context.context.is_premium:
            return f"Hello, {context.context.name}! You are a premium user. How can i support your brother"
        else:
            return "Hello, user! How can i support you?"

    # Define your agent
    agent = Agent[UserContext](
        name="Support Agent",
        instructions="You are a Support agent. who supoort user based on their account type. that they are free or premium user. use tool for better results don't guess by your own. give support to both free and premium user. **note** don't give your personalize message always give message from tool.",
        tools=[support_user],  # Add the support_user tool to the agent
        model=gemini_model,
    )

    # Run agent with a prompt

    print("--- Free User ---")
    free_user = UserContext(name="john doe", is_premium=False)
    print(free_user)
    free = await Runner.run(agent, "Please support me", context=free_user)
    print(free.final_output)

    print("--- Pro User ---")
    pro_user = UserContext(name="Huzaifa", is_premium=True)
    print(pro_user)
    pro = await Runner.run(agent, "Please support me", context=pro_user)
    print(pro.final_output)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
