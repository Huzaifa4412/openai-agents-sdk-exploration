from agents import Agent, OpenAIChatCompletionsModel, AgentHooks
from agents.run import Runner
from openai import AsyncOpenAI
import os, asyncio
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)


class SupportAgentHooks(AgentHooks):

    async def on_before_prompt(self, context, agent):
        print(f"📌 Context injected for: {context.context.name}")

    async def on_prompt(self, context, agent, prompt):
        print("🧾 Final prompt to Gemini model:\n", prompt)

    async def on_completion(self, context, agent, completion):
        if "refund" in completion.lower() or "angry" in completion.lower():
            print("🚨 Alert: Refund or angry customer detected!")

    async def on_agent_finish(self, context, agent, output):
        print(f"✅ Final reply to {context.context.name}:\n{output}")


class UserContext(BaseModel):
    name: str
    order_id: str
    priority: str  # "high", "normal"


context = UserContext(name="Huzaifa", order_id="ORD4567", priority="high")


def dynamic_instructions(context, agent):
    return (
        f"You are a helpful support agent for an e-commerce store.\n"
        f"The customer's name is {context.context.name} and their order ID is {context.context.order_id}.\n"
        f"Respond politely and give helpful info based on their priority: {context.context.priority.upper()}."
    )


agent = Agent[UserContext](
    name="Support Agent",
    model=gemini_model,
    instructions=dynamic_instructions,
    hooks=SupportAgentHooks(),
)


async def main():

    result = await Runner.run(
        agent, "Where is my order? It's been 5 days!", context=context
    )
    print("\n🎉 Final Output:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
