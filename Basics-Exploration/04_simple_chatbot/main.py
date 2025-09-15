import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled, SQLiteSession
from openai import AsyncOpenAI
from agents.run import Runner


# Disable tracing (optional, for debugging/tracing tools)
set_tracing_disabled(True)

# Load environment variables from .env
load_dotenv()
GEMINI_API=os.getenv('GEMINI_API_KEY')
if not GEMINI_API:
    raise ValueError("Gemini API key not found")

# Set up Gemini client
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize Gemini model wrapper
gemini_model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=gemini_client
)

async def main():
    # Define your agent
    agent = Agent(
        name='Simple chat bot',
        instructions='You are a simple chat bot that chats with me',
        model=gemini_model
    )

    session = SQLiteSession("default_session")
    print(session)

    # Run agent with a prompt
    while True:
        prompt = input('You: ')
        result = await Runner.run(agent, prompt,  session=session)
        print(f"Bot: {result.final_output}")
        print('-'*10)
        if prompt.lower() == 'exit':
            break


# Run the async main function
if __name__ == '__main__':
    asyncio.run(main())