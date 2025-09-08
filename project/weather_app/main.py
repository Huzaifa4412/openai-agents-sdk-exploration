# Import necessary modules
import requests
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ModelSettings,
    enable_verbose_stdout_logging,
)
from agents.run import Runner
from agents.tool import function_tool
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

set_tracing_disabled(True)
enable_verbose_stdout_logging()
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_AI_WEATHER_API_KEY = os.getenv("OPEN_AI_WEATHER_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
if not OPEN_AI_WEATHER_API_KEY:
    raise ValueError("OPEN_AI_WEATHER_API_KEY environment variable not set")

# Initialize Gemini client using OpenAI-compatible interface
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model using Gemini
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


@function_tool
def get_weather(city: str) -> str:
    """Get the weather in a given city"""
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPEN_AI_WEATHER_API_KEY}"

    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return f"The weather in {city} is {weather} with a temperature of {temperature} Kelvin"
    else:
        return f"Failed to get weather data for {city}."


# Main async function to run the agent
async def main():
    agent = Agent(
        name="weather_agent",
        instructions="Tell the weather in a given city in celsius and properly formatted",
        model=gemini_model,
        tools=[get_weather],
        model_settings=ModelSettings(tool_choice="required"),
    )

    # Run the agent with a sample prompt
    result = await Runner.run(agent, "What is the current weather in Karachi?")
    print(result.final_output)


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
