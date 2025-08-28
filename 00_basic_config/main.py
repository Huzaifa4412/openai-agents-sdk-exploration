import os
from agents import OpenAIChatCompletionsModel, Agent, set_tracing_disabled
from config import GEMINI_API_KEY
from openai import AsyncOpenAI
from agents.run import Runner
from dotenv import load_dotenv

set_tracing_disabled(True)


if not GEMINI_API_KEY:
    raise ValueError(
        "❌ Gemini API Key not found! Please set GOOGLE_GEMINI_API in your .env file."
    )

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash", openai_client=external_client
)


def main():

    print("Jawab kay lia kam chal raha hy brother")

    agent = Agent(
        name="Translator",
        instructions="Your are a power full ai assistant that translate each language into Roman urdu just me the roman urdu sentence ",
        model=model,
    )
    result = Runner.run_sync(agent, "Translate: Huzaifa is a good boy")
    print(result.final_output)

    print("Jawab aagaya brother")


if __name__ == "__main__":
    main()
