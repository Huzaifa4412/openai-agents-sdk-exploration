# 📘 Project Best Practices

## 1. Project Purpose
This repository is a collection of Python examples exploring the OpenAI Agents SDK with Gemini via the OpenAI-compatible API. It demonstrates agent creation, running strategies (sync/async/streaming), structured outputs, multi-agent handoffs, dynamic instructions, lifecycle hooks, session persistence, and context-aware tools.

## 2. Project Structure
- Top-level workspace (managed via uv) with multiple numbered example projects:
  - 00_basic_config/ — minimal setup and configuration patterns
  - 01_Runner_exploration/ — sync, async, and streaming Runner usage
  - 02_handsoff/ — agent handoffs and triage agent
  - 03_context/ — context injection with Pydantic and function tools
  - 04_simple_chatbot/ — simple chatbot with SQLite-backed session
  - 04_simple_chatbot/chatbot-exploration/ — additional chatbot variant
  - 05_output_types/ — Pydantic-typed outputs from the agent
  - 06_dynamic_instructions/ — dynamic instruction generation from context
  - 07_Life_cycle_event_hook/ — lifecycle hook usage
- Core configuration at project root:
  - config.py — loads environment (.env) and exposes keys (e.g., GEMINI_API_KEY)
  - pyproject.toml — workspace definition and shared dependencies
  - .env — environment variables (not committed); loaded by config.py or per-example
  - README.md — high-level repo overview

Notes
- Each example typically has its own pyproject.toml and a main.py entry point.
- Avoid adding sys.path manipulations where possible; prefer importing shared config via proper package layout or relative imports.
- The workspace is defined using [tool.uv.workspace]. Ensure new example directories are added there when applicable.

## 3. Test Strategy
Current repo contains no tests; add them incrementally using pytest.

- Framework: pytest (+ pytest-asyncio for async tests)
- Layout:
  - tests/ at project root for shared utilities
  - Per-example tests can live under example_dir/tests/ to scope fixtures and data
- Naming conventions: test_*.py files; functions/classes also start with test_*
- Unit vs Integration:
  - Unit tests for helper functions, dynamic instruction logic, hook behavior, tool functions, context serialization
  - Integration tests for end-to-end agent runs using mocked LLM client
- Mocking guidelines:
  - Do not make real network calls; patch AsyncOpenAI or model wrapper objects
  - For streaming, simulate ResponseTextDeltaEvent yields
  - Use fixtures to inject fake clients/models and deterministic outputs
- Coverage targets:
  - ≥80% per example directory; focus on branching logic (dynamic instructions, hooks, handoffs)
- Example testing ideas:
  - 06_dynamic_instructions: assert different instruction strings for premium vs free users
  - 07_Life_cycle_event_hook: validate hooks fire in order and process outputs
  - 05_output_types: validate Pydantic schema parsing and error on invalid shape

## 4. Code Style
- Python version: 3.13 (as declared in pyproject)
- Async-first approach:
  - Prefer async main() + Runner.run for new examples; use Runner.run_sync only for simple demos or CLI sync flows
  - For streaming, consume async generators and handle incremental deltas
- Typing and data modeling:
  - Use Pydantic BaseModel for context and typed outputs
  - Annotate function_tool definitions carefully:
    - If the tool receives a RunContextWrapper, access user context via context.context
    - If the tool receives the data model directly, access fields directly (context.is_pro_user)
    - Pick one pattern per example and be consistent
- Naming conventions:
  - Files and modules: snake_case
  - Variables/functions: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
- Docstrings/comments:
  - Add short docstrings to tools and hooks; document non-obvious flows (handoffs, dynamic instruction logic)
  - Keep comments accurate and actionable; remove outdated TODOs
- Error handling:
  - Validate environment configuration early (e.g., raise ValueError if GEMINI_API_KEY missing)
  - Wrap network-bound runs with try/except in production examples; surface actionable errors
  - Prefer explicit exception types over bare except
- Imports/config:
  - Centralize env loading in config.py for examples that share root config
  - Avoid sys.path modification; if needed, prefer package initialization or relative imports
  - Reuse a single AsyncOpenAI client per process; pass via OpenAIChatCompletionsModel

## 5. Common Patterns
- Agent + Runner:
  - Agent(name, instructions, model, ...); execute via Runner.run / Runner.run_sync / Runner.run_streamed
- Model wrapper:
  - OpenAIChatCompletionsModel wraps an AsyncOpenAI client configured with Gemini base_url
- Context-aware agents:
  - Agent[UserContext] with Pydantic models for type-safe context
  - Dynamic instructions from context (function returning a string based on context)
- Tools (function_tool):
  - Decorate Python functions to expose capabilities; use Pydantic models for validated inputs
- Handoffs/triage:
  - Use handoffs=[...] to route user input to specialized agents (e.g., code, maths, history)
- Lifecycle hooks:
  - Subclass AgentHooks to inspect prompts, completions, and finishes for logging/alerts
- Sessions:
  - SQLiteSession for conversation persistence in chatbots
- Tracing/logging:
  - set_tracing_disabled(True) to disable tracing in examples
  - enable_verbose_stdout_logging() only for local debugging

## 6. Do's and Don'ts
- Do
  - Keep examples minimal and runnable end-to-end
  - Load environment variables from a single place where possible
  - Reuse AsyncOpenAI client + model wrapper; avoid re-instantiating per call
  - Use Pydantic models for structured input/output
  - Add type annotations and docstrings for tools/hooks
  - Prefer async patterns and non-blocking I/O
  - Validate inputs and fail fast with clear errors (e.g., missing keys)
- Don't
  - Hardcode API keys or model names scattered across files; centralize in config or constants
  - Modify sys.path unless absolutely necessary
  - Mix context access styles within a single example (choose either wrapper.context or direct model and stay consistent)
  - Leave verbose debug logging enabled in production-like examples
  - Make real network calls in tests

## 7. Tools & Dependencies
- Key libraries
  - openai-agents — Agent abstractions, Runner, hooks, tools, sessions
  - openai — AsyncOpenAI client to access the Gemini OpenAI-compatible endpoint
  - python-dotenv — environment variable management from .env
  - pydantic — structured data models for context and outputs
  - uv — workspace and dependency management (uv.lock present)
- Setup
  - Requirements: Python 3.13, uv (recommended)
  - Steps:
    1) Copy .env.example to .env (create one if not present) and set GEMINI_API_KEY
    2) Sync dependencies (via uv):
       - uv sync
    3) Run examples:
       - uv run python 01_Runner_exploration/main.py
       - uv run python 06_dynamic_instructions/main.py
       - uv run python 04_simple_chatbot/main.py
    4) Alternative without uv:
       - python -m venv .venv && .venv/Scripts/activate
       - pip install -e . (or pip install -r requirements if provided)
       - python 01_Runner_exploration/main.py

## 8. Other Notes
- LLM integration
  - Keep base_url set to https://generativelanguage.googleapis.com/v1beta/openai/
  - Choose an appropriate model (e.g., gemini-2.0-flash, gemini-2.5-flash, or gemini-2.5-pro for higher quality)
  - Reuse the AsyncOpenAI client across agents to minimize overhead
- Consistency tips
  - Prefer central config via config.py where feasible; de-duplicate env loading
  - Use one context access style per example for function_tool (either RunContextWrapper or direct model)
- Workspace hygiene
  - When adding new example directories, add them to [tool.uv.workspace].members
  - Keep per-example pyproject.toml small; inherit shared config from root where possible
- Security
  - Never commit secrets; .env is for local use only
  - Consider secret scanning and pre-commit hooks in the future
- Extensibility
  - Add adapters for additional providers using the same Agent/Runner abstractions
  - Expand tests around hooks, routing, and structured outputs to guard regressions
