# OpenAI Agents Library - Complete Guide

## Overview

The `openai-agents` library (version 0.2.3+) is a powerful, high-level abstraction built on top of the OpenAI SDK that provides sophisticated agent functionality. Unlike the standard OpenAI SDK which focuses on direct API calls, this library offers structured agent patterns, context management, tool integration, and advanced workflow capabilities.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Installation & Setup](#installation--setup)
3. [Basic Agent Creation](#basic-agent-creation)
4. [Models & Providers](#models--providers)
5. [Tools & Function Calling](#tools--function-calling)
6. [Context Management](#context-management)
7. [Dynamic Instructions](#dynamic-instructions)
8. [Agent Handoffs](#agent-handoffs)
9. [Output Types & Structured Responses](#output-types--structured-responses)
10. [Sessions & Memory](#sessions--memory)
11. [Lifecycle Event Hooks](#lifecycle-event-hooks)
12. [Debugging & Logging](#debugging--logging)
13. [Best Practices](#best-practices)
14. [Advanced Patterns](#advanced-patterns)

---

## Core Concepts

### 1. Agent
The central abstraction representing an AI assistant with specific instructions, capabilities, and behavior patterns.

### 2. Model
Wrapper around different AI providers (OpenAI, Gemini, etc.) providing a unified interface.

### 3. Runner
Execution engine that manages the conversation flow between user input and agent responses.

### 4. Tools
Functions that agents can call to perform specific tasks or access external data.

### 5. Context
Typed data structure that provides runtime information to agents and tools.

---

## Installation & Setup

```toml
# pyproject.toml
[project]
dependencies = [
    "openai-agents>=0.2.3",
    "dotenv>=0.9.9",
]
```

```python
# Basic imports
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    function_tool,
    SQLiteSession,
    AgentHooks,
    RunContextWrapper,
    enable_verbose_stdout_logging
)
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
```

---

## Basic Agent Creation

### Simple Agent

```python
import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents.run import Runner

# Load environment
load_dotenv()
GEMINI_API = os.getenv('GEMINI_API_KEY')

# Setup client
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create model wrapper
gemini_model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=gemini_client
)

async def main():
    # Create agent
    agent = Agent(
        name='Coding Assistant',
        instructions='You are a helpful coding assistant who provides clear, practical advice.',
        model=gemini_model
    )
    
    # Run agent
    result = await Runner.run(agent, 'Explain Python decorators')
    print(result.final_output)

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Models & Providers

### Supported Providers

#### 1. Gemini Integration
```python
# Gemini setup
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',  # or gemini-2.5-flash, gemini-2.5-pro
    openai_client=gemini_client
)
```

#### 2. OpenAI Integration
```python
# OpenAI setup
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_model = OpenAIChatCompletionsModel(
    model='gpt-4',  # or gpt-3.5-turbo, gpt-4-turbo
    openai_client=openai_client
)
```

### Model Configuration Options
- **Temperature**: Control randomness
- **Max Tokens**: Limit response length
- **Top P**: Nucleus sampling parameter
- **Frequency Penalty**: Reduce repetition

---

## Tools & Function Calling

### Basic Function Tool

```python
from agents import function_tool
from pydantic import BaseModel

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulate weather API call
    return f"The weather in {city} is sunny, 25°C"

# Agent with tools
agent = Agent(
    name='Weather Assistant',
    instructions='Help users get weather information using the weather tool.',
    tools=[get_weather],
    model=gemini_model
)
```

### Context-Aware Tools

```python
from agents import RunContextWrapper

class UserContext(BaseModel):
    user_id: str
    is_premium: bool

@function_tool
def get_user_info(context: RunContextWrapper[UserContext]) -> str:
    """Get user information based on context."""
    user = context.context
    status = "Premium" if user.is_premium else "Free"
    return f"User {user.user_id} has {status} account"

# Typed agent
agent = Agent[UserContext](
    name='User Assistant',
    instructions='Provide personalized assistance based on user context.',
    tools=[get_user_info],
    model=gemini_model
)
```

### Memory-Based Tools

```python
@function_tool
async def remember_preference(preference: str, context) -> str:
    """Save user preference to memory."""
    await context.memory.set("user_preference", preference)
    return f"Remembered your preference: {preference}"

@function_tool
async def recall_preference(context) -> str:
    """Recall user preference from memory."""
    pref = await context.memory.get("user_preference")
    return f"Your preference is: {pref}" if pref else "No preference saved"
```

---

## Context Management

### Defining Context Types

```python
from pydantic import BaseModel
from typing import TypedDict

# Using Pydantic (recommended)
class UserContext(BaseModel):
    name: str
    subscription_tier: str
    preferences: dict = {}

# Using TypedDict (alternative)
class SimpleContext(TypedDict):
    user_id: str
    language: str
```

### Using Context in Agents

```python
def context_aware_instructions(context: RunContextWrapper[UserContext], agent):
    user = context.context
    return f"""
    You are assisting {user.name} who has a {user.subscription_tier} subscription.
    Tailor your responses accordingly.
    """

agent = Agent[UserContext](
    name='Personal Assistant',
    instructions=context_aware_instructions,
    model=gemini_model
)

# Running with context
user_context = UserContext(
    name="Alice",
    subscription_tier="premium",
    preferences={"language": "python"}
)

result = await Runner.run(agent, "Help me with coding", context=user_context)
```

---

## Dynamic Instructions

### Function-Based Dynamic Instructions

```python
def dynamic_instructions(context: RunContextWrapper[UserContext], agent) -> str:
    user = context.context
    
    if user.subscription_tier == "premium":
        return f"Hello {user.name}! Provide detailed, premium-level assistance."
    else:
        return f"Hello {user.name}! Provide helpful but basic assistance."

agent = Agent[UserContext](
    name='Adaptive Assistant',
    instructions=dynamic_instructions,
    model=gemini_model
)
```

### Multi-Language Support

```python
def localized_instructions(context: RunContextWrapper, agent) -> str:
    lang = context.context.get('language', 'english')
    
    instructions = {
        'urdu': f"User ka naam {context.context['name']} hai. Urdu mein jawab do.",
        'english': f"User's name is {context.context['name']}. Respond in English.",
        'spanish': f"El nombre del usuario es {context.context['name']}. Responde en español."
    }
    
    return instructions.get(lang, instructions['english'])
```

---

## Agent Handoffs

### Creating Specialized Agents

```python
# Specialized agents
coding_agent = Agent(
    name="Coding Specialist",
    instructions="You are an expert coding assistant. Focus on code quality, best practices, and detailed explanations.",
    model=gemini_model
)

math_agent = Agent(
    name="Math Specialist", 
    instructions="You are a mathematics expert. Solve problems step-by-step with clear explanations.",
    model=gemini_model
)

history_agent = Agent(
    name="History Specialist",
    instructions="You are a history expert. Provide accurate historical information with context.",
    model=gemini_model
)
```

### Triage Agent with Handoffs

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="""
    You are a helpful assistant that routes user requests to appropriate specialists.
    - For coding questions: hand off to coding specialist
    - For math problems: hand off to math specialist  
    - For history questions: hand off to history specialist
    - For general questions: handle yourself
    """,
    handoffs=[coding_agent, math_agent, history_agent],
    model=gemini_model
)

# Usage
result = await Runner.run(triage_agent, "How do I implement a binary search in Python?")
```

---

## Output Types & Structured Responses

### Pydantic Output Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class CalendarEvent(BaseModel):
    title: str
    date: str
    time: str
    location: Optional[str] = Field(description="Event location if specified")
    attendees: List[str] = []

# Agent with structured output
agent = Agent(
    name="Calendar Extractor",
    instructions="Extract calendar events from text. Parse all relevant details.",
    model=gemini_model,
    output_type=CalendarEvent
)

# Usage
event_text = """
Team meeting scheduled for March 15th at 2:00 PM in Conference Room A.
Attendees: John, Sarah, Mike.
"""

result = await Runner.run(agent, event_text)
print(result.final_output)  # Returns CalendarEvent instance
```

### Complex Structured Outputs

```python
class TaskAnalysis(BaseModel):
    priority: str = Field(description="high, medium, or low")
    estimated_hours: float
    required_skills: List[str]
    dependencies: List[str] = []
    risk_factors: List[str] = []

class ProjectPlan(BaseModel):
    project_name: str
    description: str
    tasks: List[TaskAnalysis]
    total_estimated_hours: float
    timeline_weeks: int

agent = Agent(
    name="Project Planner",
    instructions="Analyze project requirements and create detailed project plans.",
    model=gemini_model,
    output_type=ProjectPlan
)
```

---

## Sessions & Memory

### SQLite Sessions

```python
from agents import SQLiteSession

async def chatbot_with_memory():
    agent = Agent(
        name='Memory Chatbot',
        instructions='You are a helpful chatbot that remembers our conversation.',
        model=gemini_model
    )
    
    # Create persistent session
    session = SQLiteSession("user_123_session")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        result = await Runner.run(agent, user_input, session=session)
        print(f"Bot: {result.final_output}")
```

### Custom Memory Operations

```python
@function_tool
async def save_user_data(key: str, value: str, context) -> str:
    """Save data to user's memory."""
    await context.memory.set(key, value)
    return f"Saved {key}: {value}"

@function_tool
async def get_user_data(key: str, context) -> str:
    """Retrieve data from user's memory."""
    value = await context.memory.get(key)
    return f"{key}: {value}" if value else f"No data found for {key}"

@function_tool
async def list_user_data(context) -> str:
    """List all saved user data."""
    # Implementation depends on memory backend
    return "Listed all user data"
```

---

## Lifecycle Event Hooks

### Custom Hook Implementation

```python
from agents import AgentHooks

class CustomAgentHooks(AgentHooks):
    
    async def on_before_prompt(self, context, agent):
        """Called before sending prompt to model."""
        print(f"🔄 Processing request for: {context.context.name}")
        
    async def on_prompt(self, context, agent, prompt):
        """Called with the final prompt sent to model."""
        print(f"📤 Sending prompt: {prompt[:100]}...")
        
    async def on_completion(self, context, agent, completion):
        """Called when model returns completion."""
        # Log sensitive content detection
        if any(word in completion.lower() for word in ['password', 'secret', 'token']):
            print("⚠️  Sensitive content detected in response")
            
    async def on_agent_finish(self, context, agent, output):
        """Called when agent finishes processing."""
        print(f"✅ Response ready for: {context.context.name}")
        
    async def on_error(self, context, agent, error):
        """Called when an error occurs."""
        print(f"❌ Error occurred: {str(error)}")
```

### Using Hooks with Agents

```python
class UserContext(BaseModel):
    name: str
    user_id: str

agent = Agent[UserContext](
    name="Monitored Agent",
    instructions="You are a helpful assistant.",
    model=gemini_model,
    hooks=CustomAgentHooks()
)

context = UserContext(name="Alice", user_id="user_123")
result = await Runner.run(agent, "Hello!", context=context)
```

### Advanced Hook Patterns

```python
class SecurityHooks(AgentHooks):
    
    async def on_completion(self, context, agent, completion):
        """Security monitoring."""
        # Check for potential data leaks
        sensitive_patterns = [
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',        # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, completion):
                print("🚨 SECURITY ALERT: Sensitive data detected!")
                # Log to security system
                break

class PerformanceHooks(AgentHooks):
    
    def __init__(self):
        self.start_time = None
        
    async def on_before_prompt(self, context, agent):
        """Start timing."""
        import time
        self.start_time = time.time()
        
    async def on_agent_finish(self, context, agent, output):
        """Log performance metrics."""
        import time
        duration = time.time() - self.start_time
        print(f"⏱️  Response time: {duration:.2f}s")
```

---

## Debugging & Logging

### Enable Verbose Logging

```python
from agents import enable_verbose_stdout_logging, set_tracing_disabled

# Enable detailed logging
enable_verbose_stdout_logging()

# Disable tracing (for production)
set_tracing_disabled(True)
```

### Custom Logging Setup

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class LoggingHooks(AgentHooks):
    
    async def on_prompt(self, context, agent, prompt):
        logger.info(f"Agent {agent.name} received prompt: {prompt[:100]}...")
        
    async def on_completion(self, context, agent, completion):
        logger.info(f"Agent {agent.name} generated response: {len(completion)} chars")
        
    async def on_error(self, context, agent, error):
        logger.error(f"Agent {agent.name} error: {str(error)}")
```

---

## Best Practices

### 1. Agent Design Patterns

```python
# ✅ Good: Specific, focused agents
coding_agent = Agent(
    name="Python Coding Assistant",
    instructions="""
    You are a Python coding expert. Focus on:
    - Writing clean, readable code
    - Following PEP 8 standards
    - Providing practical examples
    - Explaining complex concepts simply
    """,
    model=gemini_model
)

# ❌ Avoid: Overly broad agents
generic_agent = Agent(
    name="Do Everything Agent",
    instructions="You can help with anything",
    model=gemini_model
)
```

### 2. Context Management

```python
# ✅ Good: Structured context with validation
class UserContext(BaseModel):
    user_id: str = Field(..., min_length=1)
    subscription_tier: str = Field(..., regex="^(free|premium|enterprise)$")
    preferences: dict = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True

# ❌ Avoid: Unstructured context
context = {"user": "alice", "type": "premium"}  # No validation
```

### 3. Error Handling

```python
async def robust_agent_run():
    try:
        result = await Runner.run(agent, user_input, context=context)
        return result.final_output
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return "I apologize, but I encountered an error. Please try again."
```

### 4. Tool Design

```python
# ✅ Good: Clear, focused tools
@function_tool
def calculate_tax(amount: float, tax_rate: float) -> str:
    """Calculate tax amount for a given sum and tax rate.
    
    Args:
        amount: The base amount to calculate tax on
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%)
    
    Returns:
        Formatted string with tax calculation
    """
    tax = amount * tax_rate
    total = amount + tax
    return f"Tax: ${tax:.2f}, Total: ${total:.2f}"

# ❌ Avoid: Vague, multi-purpose tools
@function_tool
def do_math(operation: str, numbers: list) -> str:
    """Do some math operation."""  # Too vague
    pass
```

---

## Advanced Patterns

### 1. Multi-Agent Workflows

```python
class WorkflowContext(BaseModel):
    task_id: str
    current_step: str
    data: dict = {}

# Research agent
research_agent = Agent[WorkflowContext](
    name="Research Specialist",
    instructions="Research the given topic and provide comprehensive information.",
    model=gemini_model
)

# Analysis agent  
analysis_agent = Agent[WorkflowContext](
    name="Analysis Specialist", 
    instructions="Analyze the research data and provide insights.",
    model=gemini_model
)

# Report agent
report_agent = Agent[WorkflowContext](
    name="Report Writer",
    instructions="Create a professional report based on research and analysis.",
    model=gemini_model
)

async def workflow_pipeline(topic: str):
    context = WorkflowContext(task_id="task_001", current_step="research")
    
    # Step 1: Research
    research_result = await Runner.run(research_agent, f"Research: {topic}", context=context)
    context.data["research"] = research_result.final_output
    context.current_step = "analysis"
    
    # Step 2: Analysis
    analysis_result = await Runner.run(analysis_agent, "Analyze the research data", context=context)
    context.data["analysis"] = analysis_result.final_output
    context.current_step = "report"
    
    # Step 3: Report
    report_result = await Runner.run(report_agent, "Create final report", context=context)
    
    return report_result.final_output
```

### 2. Conditional Agent Selection

```python
def select_agent_by_complexity(query: str) -> Agent:
    """Select appropriate agent based on query complexity."""
    
    # Simple keyword-based routing
    if any(word in query.lower() for word in ['code', 'programming', 'function']):
        return coding_agent
    elif any(word in query.lower() for word in ['calculate', 'math', 'equation']):
        return math_agent
    elif len(query.split()) > 20:  # Complex query
        return advanced_agent
    else:
        return basic_agent

async def smart_routing(query: str):
    selected_agent = select_agent_by_complexity(query)
    result = await Runner.run(selected_agent, query)
    return result.final_output
```

### 3. Agent Composition

```python
class CompositeAgent:
    def __init__(self):
        self.agents = {
            'research': research_agent,
            'analysis': analysis_agent,
            'writing': writing_agent
        }
    
    async def process(self, task: str, workflow: List[str]):
        results = {}
        
        for step in workflow:
            if step in self.agents:
                agent = self.agents[step]
                # Pass previous results as context
                context_data = {"previous_results": results, "current_task": task}
                result = await Runner.run(agent, task, context=context_data)
                results[step] = result.final_output
        
        return results

# Usage
composite = CompositeAgent()
results = await composite.process(
    "Analyze market trends for AI startups",
    workflow=['research', 'analysis', 'writing']
)
```

---

## Performance Optimization

### 1. Connection Pooling

```python
# Reuse client connections
class ModelManager:
    def __init__(self):
        self.clients = {}
        self.models = {}
    
    def get_model(self, provider: str, model_name: str):
        key = f"{provider}:{model_name}"
        
        if key not in self.models:
            if provider == "gemini":
                client = AsyncOpenAI(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                self.models[key] = OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=client
                )
        
        return self.models[key]

# Global model manager
model_manager = ModelManager()
```

### 2. Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedAgent:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.cache = {}
    
    def _cache_key(self, prompt: str, context=None) -> str:
        """Generate cache key for prompt and context."""
        content = f"{prompt}:{str(context)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def run(self, prompt: str, context=None):
        cache_key = self._cache_key(prompt, context)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await Runner.run(self.agent, prompt, context=context)
        self.cache[cache_key] = result
        
        return result
```

---

## Integration Examples

### 1. FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session = SQLiteSession(f"user_{request.user_id}")
        result = await Runner.run(chatbot_agent, request.message, session=session)
        
        return ChatResponse(
            response=result.final_output,
            session_id=session.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Streamlit Integration

```python
import streamlit as st

st.title("AI Assistant")

# Initialize session state
if "session" not in st.session_state:
    st.session_state.session = SQLiteSession("streamlit_session")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = await Runner.run(
                chatbot_agent, 
                prompt, 
                session=st.session_state.session
            )
            response = result.final_output
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## Comparison with Standard OpenAI SDK

| Feature | OpenAI SDK | OpenAI Agents Library |
|---------|------------|----------------------|
| **Complexity** | Low-level API calls | High-level abstractions |
| **Agent Patterns** | Manual implementation | Built-in agent framework |
| **Context Management** | Manual state handling | Structured context types |
| **Tool Integration** | Function calling setup | Decorator-based tools |
| **Memory/Sessions** | Custom implementation | Built-in session management |
| **Multi-Agent** | Complex orchestration | Simple handoff patterns |
| **Lifecycle Hooks** | Not available | Comprehensive hook system |
| **Provider Support** | OpenAI only | Multiple providers (OpenAI, Gemini) |
| **Learning Curve** | Moderate | Higher (more concepts) |
| **Flexibility** | High (direct control) | Medium (abstracted patterns) |

---

## Migration Considerations

### From Standard OpenAI SDK

```python
# Standard OpenAI SDK
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

# OpenAI Agents equivalent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=openai_model
)

result = await Runner.run(agent, "Hello!")
```

### Benefits of Migration

1. **Structured Patterns**: Built-in agent patterns reduce boilerplate
2. **Context Management**: Type-safe context handling
3. **Tool Integration**: Simplified function calling
4. **Session Management**: Built-in conversation memory
5. **Multi-Provider**: Easy switching between AI providers
6. **Debugging**: Comprehensive logging and hooks

### Migration Challenges

1. **Learning Curve**: New concepts and patterns
2. **Abstraction Overhead**: Less direct control
3. **Dependency**: Additional library dependency
4. **Documentation**: Smaller community compared to OpenAI SDK

---

## Conclusion

The OpenAI Agents library provides a powerful, structured approach to building AI applications with sophisticated agent patterns. It excels in scenarios requiring:

- **Multi-agent workflows**
- **Complex context management** 
- **Tool integration**
- **Session persistence**
- **Provider flexibility**

While it adds complexity compared to the standard OpenAI SDK, it significantly reduces boilerplate code and provides robust patterns for production AI applications.

Choose OpenAI Agents when building complex, stateful AI applications that benefit from structured agent patterns. Stick with the standard OpenAI SDK for simple, direct API interactions or when maximum control is required.