# ✅ 1. AGENT LEVEL – Specific Agent ke liye Custom LLM
# 🔹 Concept (Simple Samajh)
# Tum ek specific agent ke liye Gemini set karte ho.

# Baaki agents default OpenAI use karenge.

# Best for Multi-Agent Systems jahan har agent apne kaam ke liye best model use kare.

# 🔹 Code (Tumhara hi example explained deeply)
# python

import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner

# Gemini ko OpenAI client ke through connect kar rahe hain
client = AsyncOpenAI(
    api_key="YOUR_GEMINI_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


async def main():
    # Sirf is agent ke liye Gemini set kiya
    agent = Agent(
        name="UrduTeacher",
        instructions="Tum sirf Urdu me jawab do.",
        model=OpenAIChatCompletionsModel(
            model="gemini-2.0-flash", openai_client=client
        ),
    )

    result = await Runner.run(agent, "AI agents seekhna acha idea hai?")
    print(result.final_output)


# if __name__ == "__main__":
#     asyncio.run(main())
# # 🔹 Output Example

# "Ji haan! AI agents seekhna aaj ke daur me bohot acha aur demand wala idea hai."
# 🔹 Use-Case
# ✔ Multi-Agent Projects – e.g.,

# Agent 1 (Coder) → OpenAI (reasoning aur coding ke liye best).

# Agent 2 (News Reporter) → Gemini (fresh knowledge aur updates ke liye).

# ✅ 2. RUN LEVEL – Sirf Ek Run ke liye LLM Change
# 🔹 Concept (Simple Samajh)
# Tumhara default agent OpenAI use karta hai, lekin sirf ek request ke liye Gemini use karwa sakte ho.

# Agent ki default setting change nahi hoti.

# 🔹 Code (Tumhara example with extra clarity)
# python

from agents import Agent, Runner
from agents.run import RunConfig
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

# Gemini Client
external_client = AsyncOpenAI(
    api_key="YOUR_GEMINI_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Gemini Model
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

# RunConfig (Temporary)
config = RunConfig(
    model=gemini_model, model_provider=external_client, tracing_disabled=True
)

# Agent normally OpenAI use karega
agent = Agent(name="DefaultAssistant", instructions="You are a helpful assistant.")

# Sirf is run ke liye Gemini use hoga
result = Runner.run_sync(agent, "Aaj Pakistan ka weather kaisa hai?", run_config=config)
print(result.final_output)
# 🔹 Output Example

"Aaj Pakistan ke zyadatar ilaqon me barish ka imkaan hai, temperature 30°C ke qareeb hai."
# 🔹 Use-Case
# ✔ Occasional Switch – Tumhara default agent OpenAI ho, lekin fresh news ya real-time info ke liye Gemini call ho.

# ✅ 3. GLOBAL LEVEL – Sab Agents ke liye Gemini
# 🔹 Concept (Simple Samajh)
# Poore app me default LLM ko Gemini bana dete ho.

# Har naya agent automatically Gemini use karega (jab tak tum agent level pe override na karo).

# 🔹 Code (Tumhara example with details)

from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

# Tracing band kar diya
set_tracing_disabled(True)

# Gemini Client
external_client = AsyncOpenAI(
    api_key="YOUR_GEMINI_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Global Default Gemini Set
set_default_openai_api("chat_completions")
set_default_openai_client(external_client)

# Ab har agent Gemini use karega
agent = Agent(
    name="GlobalAssistant",
    instructions="Tum helpful aur polite ho.",
    model="gemini-2.0-flash",
)

result = Runner.run_sync(agent, "AI agents future me kis field me zyada use honge?")
print(result.final_output)
# 🔹 Output Example

"AI agents healthcare, education aur business automation me sabse zyada use honge."
# 🔹 Use-Case
# ✔ Jab tumhara poora project Gemini pe depend ho, e.g., ek SaaS jo Google ecosystem ke liye banaya gaya ho.

# ✅ 📊 Summary Table
# Level	Kya karta hai?	Use Case
# Agent	Sirf ek agent ke liye LLM set	Multi-Agent Systems (ek coding agent OpenAI pe aur ek news agent Gemini pe)
# Run	Sirf ek request ke liye temporary LLM change	Kabhi kabar ek run ke liye Gemini use karna
# Global	Poore project ke liye default LLM set	Pure project ko Gemini pe shift karna
