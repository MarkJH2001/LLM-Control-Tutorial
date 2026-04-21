# LLM + Control Tutorial

An applied tutorial covering **LLM fundamentals**, **API usage**, and **agentic workflows**, with a control-systems angle.

The tutorial uses the `openai` Python SDK to reach **OpenAI**, **DeepSeek**, and **Qwen** through one client — code that differs per provider is shown in three tabs, so you can pick whichever you have a key for.

## What this site covers

- **[Getting Started](getting-started/index.md)** — conda environment, API keys.
- **[LLM Basics](llm-basics/index.md)** — what an LLM is, tokens, sampling, context window.
- **[Calling the API](api/index.md)** — first call, unified client, streaming, tool use.
- **[Agentic Workflows](agents/index.md)** — plan/act/observe loops, memory, multi-agent patterns.
- **[LLM + Control](control/index.md)** *(in progress)* — applying these patterns to control-systems problems.

## Running the code

Code samples are shown inline with their expected output. To run them yourself, follow [Setup](getting-started/setup.md) to create the `llm-tutorial` conda env, then execute the snippets locally.

!!! tip "Start here"
    New to all of this? Go to [Setup](getting-started/setup.md) first, then [LLM Basics](llm-basics/index.md).
