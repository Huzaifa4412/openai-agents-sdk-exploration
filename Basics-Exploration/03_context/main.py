from dotenv import load_dotenv
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    function_tool,
    # enable_verbose_stdout_logging,
    RunContextWrapper,
)
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")
from config import GEMINI_API_KEY

# enable_verbose_stdout_logging()

# Disable tracing (optional)
set_tracing_disabled(True)

# Load API Key
load_dotenv(ROOT_DIR / ".env")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key missing!")

# Setup Gemini Client
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=client)


# ? Creating Data class
class UserContext(BaseModel):
    uid: str
    is_pro_user: bool


@function_tool
def greet_user(context: UserContext) -> str:
    """Give a personalized greeting based on user's status."""
    if context.context.is_pro_user:
        return f"Hello Pro user {context.uid}, welcome back!"
    else:
        return f"Hello {context.uid}, upgrade to Pro for more features!"


agent = Agent[UserContext](
    name="Greeter",
    instructions="Greet the user based on their account type using the tool.",
    tools=[greet_user],
    model=model,
)

# Create different user contexts
pro_user = UserContext(uid="alice123", is_pro_user=True)
free_user = UserContext(uid="bob456", is_pro_user=False)

# Run with different contexts
print("\n--- Pro User ---")
result = Runner.run_sync(agent, "Please greet me", context=pro_user)
print(result.final_output)

print("\n--- Free User ---")
result = Runner.run_sync(agent, "Please greet me", context=free_user)
print(result.final_output)
