How it works
When session memory is enabled:

Before each run: The runner automatically retrieves the conversation history for the session and prepends it to the input items.
After each run: All new items generated during the run (user input, assistant responses, tool calls, etc.) are automatically stored in the session.
Context preservation: Each subsequent run with the same session includes the full conversation history, allowing the agent to maintain context.
This eliminates the need to manually call .to_input_list() and manage conversation state between runs.
