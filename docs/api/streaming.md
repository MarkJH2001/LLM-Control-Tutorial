# Streaming

[![Open in Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/streaming.ipynb)
[![Open in Deepnote](../assets/deepnote-badge.svg)](https://deepnote.com/launch?url=https://github.com/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/streaming.ipynb)

A non-streaming API call blocks until the model is finished, then returns the whole reply at once. **Streaming** returns tokens as they're generated, so a UI can display the answer progressively — the same effect you see in ChatGPT.

You enable streaming with one extra flag. Everything else stays the same.

## Streaming with the OpenAI SDK

Pass `stream=True` and iterate over the returned object. Each iteration yields a small chunk; the new text lives at `chunk.choices[0].delta.content`.

### Pick your provider

All three providers we cover use the `openai` SDK; only the client and model differ.

=== "OpenAI"

    ```python
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-4o-mini"
    ```

=== "DeepSeek"

    ```python
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat"
    ```

=== "Qwen"

    ```python
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = "qwen-plus"
    ```

### Shared streaming code

```python title="stream_chat.py"
import os
from dotenv import load_dotenv

load_dotenv()
# client and model come from one of the tabs above

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain a PID controller in three sentences."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

Two details that trip people up:

- **`flush=True`** is needed on `print` so the terminal doesn't buffer.
- **The first chunk's `delta.content` is often `None`** (it carries the role, not text). Always guard with `if delta:`.

### Token usage with streaming

By default, OpenAI does **not** include token counts in a streamed response. Opt in with `stream_options`:

```python
stream = client.chat.completions.create(
    model=model,
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},
)
```

The very last chunk will then carry `chunk.usage` with `prompt_tokens`, `completion_tokens`, `total_tokens`.

## Adding streaming to the unified client

Building on the `chat()` function from [unified-client.md](unified-client.md), expose a generator that yields chunks regardless of provider:

```python title="unified_stream.py"
from typing import Iterator
import os
from openai import OpenAI

# PROVIDERS dict from unified-client.md is reused as-is

def stream_chat(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    system: str | None = None,
) -> Iterator[str]:
    cfg = PROVIDERS[provider]
    client = OpenAI(api_key=os.environ[cfg.env_var], base_url=cfg.base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=model or cfg.default_model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Use it the same way for any provider:

```python
for token in stream_chat("Say hi.", provider="deepseek"):
    print(token, end="", flush=True)
print()
```

## Gotchas

- **Network buffering.** Some corporate proxies or load balancers buffer the response and defeat streaming — you'll get the full reply in one go. Test outside the proxy if you suspect this.
- **Tool use during streaming** uses different chunk shapes (start / delta / stop events for each tool block). Out of scope for this page; covered in [Tool Use](tool-use.md).
- **Don't mix `print` and a UI.** In a real app, push each chunk over a websocket / SSE / queue to the frontend instead of writing to stdout.
- **Errors mid-stream.** Network drops, rate limits, and content-filter triggers can interrupt a stream. Wrap the loop in `try/except` and decide whether to retry or surface the partial reply.

## Next

- [Tool Use](tool-use.md) — let the model call your Python functions.
