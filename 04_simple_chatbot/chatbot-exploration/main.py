from agents import Agent, OpenAIChatCompletionsModel, Runner, SQLiteSession
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
import os

load_dotenv()

gemini_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


async def main():
    agent = Agent(
        "assistant",
        "You are a helpful AI assistant. You are knowledgeable about various topics and aim to provide accurate, informative, and thoughtful responses. You should be polite, professional, and engaging in your interactions. Feel free to ask clarifying questions if needed to better assist the user.",
        model=gemini_model,
    )

    print('Enter your message (type "exit" to stop):')

    session = SQLiteSession("chat_history.db")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = await Runner.run(agent, user_input, session=session)
        print(r"agent: ", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
