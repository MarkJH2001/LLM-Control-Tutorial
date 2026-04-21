# First Call

[![Open in Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/first_call.ipynb)
[![Open in Deepnote](../assets/deepnote-badge.svg)](https://deepnote.com/launch?url=https://github.com/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/first_call.ipynb)

The smallest possible round-trip to an LLM: send one message, print the reply. Shown for all three providers we cover — all of them use the same `openai` client.

## Prerequisites

- A key from at least one provider. If you don't have one, pick from the [Get an API Key](get-a-key/index.md) page.
- The key stored in your environment. We'll assume a `.env` file at the project root with whichever you have:

    ```bash
    OPENAI_API_KEY=sk-...
    DEEPSEEK_API_KEY=sk-...
    DASHSCOPE_API_KEY=sk-...
    ```

- The `llm-tutorial` conda env activated:

    ```bash
    conda activate llm-tutorial
    ```

The `openai` SDK is already installed as a project dependency. **DeepSeek and Qwen reuse the `openai` SDK** — only `base_url` and `model` change.

## Hello world

=== "OpenAI"

    ```python title="first_call_openai.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    Expected output:

    ```text
    Hi there! How can I help you today?
    ```

=== "DeepSeek"

    DeepSeek exposes an OpenAI-compatible endpoint, so the `openai` client works as-is — just override `base_url` and pick a DeepSeek model.

    ```python title="first_call_deepseek.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    Expected output:

    ```text
    Hello! Hope you're doing well.
    ```

    Use `model="deepseek-reasoner"` for the reasoning model; the response will additionally carry `resp.choices[0].message.reasoning_content`.

=== "Qwen"

    Qwen on Alibaba's DashScope also exposes an OpenAI-compatible endpoint.

    ```python title="first_call_qwen.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    Expected output:

    ```text
    Hi! Nice to meet you.
    ```

## Response shape

Most fields you'll touch:

```python
resp.choices[0].message.content    # the assistant's text
resp.choices[0].finish_reason      # "stop", "length", "tool_calls", ...
resp.model                         # resolved model ID
resp.usage.prompt_tokens           # input tokens billed
resp.usage.completion_tokens       # output tokens billed
resp.usage.total_tokens
```

## Adding a system prompt

Prepend a message with `role: "system"`:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",  # or "deepseek-chat", "qwen-plus"
    messages=[
        {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
        {"role": "user", "content": "What is a PID controller?"},
    ],
)
```

## Troubleshooting

- `AuthenticationError` / `401`: your key is wrong, revoked, or scoped to a different workspace / region.
- OpenAI `insufficient_quota`: the key has no credits; top up at [Billing](https://platform.openai.com/account/billing).
- `ConnectionError`: check your network / proxy; some provider endpoints are region-blocked.

## Next

- [Unified Client](unified-client.md) — wrap all three providers behind one function so you can swap with a single line.
- [Streaming](streaming.md) — return tokens as they're generated.
- [Tool Use](tool-use.md) — let the model call your Python functions.
