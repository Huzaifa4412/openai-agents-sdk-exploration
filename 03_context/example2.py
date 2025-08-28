import asyncio
from agents import Agent, Runner, function_tool, set_tracing_disabled, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import sys
from pathlib import Path
ROOT_DIR= Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from config import GEMINI_API_KEY
from dotenv import load_dotenv
load_dotenv(ROOT_DIR/".env")

# ✅ Gemini Client Setup
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ✅ OpenAgents compatible model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# ✅ Disable tracing
set_tracing_disabled(True)

# ✅ Tool: Save favorite movie using context.memory
@function_tool
async def remember_favorite_movie(movie: str, context):
    await context.memory.set("fav_movie", movie)
    return f"🎬 Got it! Your favorite movie is {movie}."

# ✅ Tool: Recall movie from memory
@function_tool
async def recall_movie(context):
    movie = await context.memory.get("fav_movie")
    if movie:
        return f"🍿 Your favorite movie is {movie}."
    else:
        return "I don't remember your favorite movie yet."

# ✅ Agent setup
agent = Agent(
    name="GeminiContextAgent",
    instructions="You remember and recall user's favorite movie using memory.",
    model=model,
    tools=[remember_favorite_movie, recall_movie],
)

# ✅ Main runner
async def main():
    print("🔴 REMEMBERING...")
    result1 = await Runner.run(agent, "My favorite movie is Interstellar.")
    print(result1.final_output)

    print("🟢 RECALLING...")
    result2 = await Runner.run(agent, "What's my favorite movie?")
    print(result2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
