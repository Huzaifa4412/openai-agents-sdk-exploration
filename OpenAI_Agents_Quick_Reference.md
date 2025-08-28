# OpenAI Agents Library - Quick Reference

## Essential Imports
```python
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner, function_tool,
    SQLiteSession, AgentHooks, RunContextWrapper, set_tracing_disabled
)
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio
```

## Basic Agent Setup
```python
# Model setup (Gemini)
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(model='gemini-2.0-flash', openai_client=client)

# Simple agent
agent = Agent(
    name='Assistant',
    instructions='You are a helpful assistant.',
    model=model
)

# Run agent
result = await Runner.run(agent, 'Hello!')
print(result.final_output)
```

## Context-Aware Agent
```python
class UserContext(BaseModel):
    name: str
    is_premium: bool

def dynamic_instructions(context: RunContextWrapper[UserContext], agent):
    user = context.context
    return f"User: {user.name}, Premium: {user.is_premium}. Respond accordingly."

agent = Agent[UserContext](
    name='Context Agent',
    instructions=dynamic_instructions,
    model=model
)

# Usage
context = UserContext(name="Alice", is_premium=True)
result = await Runner.run(agent, "Help me", context=context)
```

## Function Tools
```python
@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

agent = Agent(
    name='Weather Agent',
    instructions='Help with weather using the tool.',
    tools=[get_weather],
    model=model
)
```

## Agent Handoffs
```python
coding_agent = Agent(name="Coder", instructions="Code expert", model=model)
math_agent = Agent(name="Math", instructions="Math expert", model=model)

triage_agent = Agent(
    name="Triage",
    instructions="Route to appropriate specialist",
    handoffs=[coding_agent, math_agent],
    model=model
)
```

## Structured Output
```python
class Event(BaseModel):
    title: str
    date: str
    location: str = None

agent = Agent(
    name='Event Extractor',
    instructions='Extract events from text',
    model=model,
    output_type=Event
)
```

## Sessions & Memory
```python
session = SQLiteSession("user_123")
result = await Runner.run(agent, "Remember my name is Alice", session=session)
```

## Lifecycle Hooks
```python
class MyHooks(AgentHooks):
    async def on_before_prompt(self, context, agent):
        print("Processing...")
    
    async def on_completion(self, context, agent, completion):
        print(f"Response: {len(completion)} chars")

agent = Agent(
    name='Monitored Agent',
    instructions='Helpful assistant',
    model=model,
    hooks=MyHooks()
)
```

## Common Patterns

### Chatbot with Memory
```python
async def chatbot():
    agent = Agent(name='Bot', instructions='Friendly chatbot', model=model)
    session = SQLiteSession("chat_session")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        result = await Runner.run(agent, user_input, session=session)
        print(f"Bot: {result.final_output}")
```

### Multi-Step Workflow
```python
async def workflow(topic):
    research_agent = Agent(name="Researcher", instructions="Research topics", model=model)
    writer_agent = Agent(name="Writer", instructions="Write reports", model=model)
    
    # Step 1: Research
    research = await Runner.run(research_agent, f"Research: {topic}")
    
    # Step 2: Write report
    report = await Runner.run(writer_agent, f"Write report on: {research.final_output}")
    
    return report.final_output
```

### Context-Aware Tools
```python
@function_tool
def personalized_greeting(context: RunContextWrapper[UserContext]) -> str:
    """Greet user based on their context."""
    user = context.context
    return f"Hello {user.name}! Premium: {user.is_premium}"

agent = Agent[UserContext](
    name='Personal Assistant',
    instructions='Provide personalized assistance',
    tools=[personalized_greeting],
    model=model
)
```

## Debugging
```python
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()  # Enable detailed logs
set_tracing_disabled(True)       # Disable tracing for production
```

## Error Handling
```python
try:
    result = await Runner.run(agent, user_input, context=context)
    return result.final_output
except Exception as e:
    print(f"Error: {e}")
    return "Sorry, something went wrong."