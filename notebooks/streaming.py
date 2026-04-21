# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Streaming — runnable notebook
#
# Companion to [**Streaming**](https://markjh2001.github.io/LLM-Control-Tutorial/api/streaming/).

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet openai python-dotenv

# %% [markdown]
# ## 2. Load an API key

# %%
import os

KEY_VARS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")

try:
    from google.colab import userdata
except ImportError:
    userdata = None

if userdata is not None:
    for k in KEY_VARS:
        try:
            v = userdata.get(k)
        except Exception:
            v = None
        if v:
            os.environ[k] = v

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not any(os.environ.get(k) for k in KEY_VARS):
    from getpass import getpass
    which = input("Which provider? (qwen / deepseek / openai) [qwen]: ").strip().lower() or "qwen"
    os.environ[{"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}[which]] = getpass("Paste your key: ")

print("Keys detected:", [k for k in KEY_VARS if os.environ.get(k)])

# %% [markdown]
# ## 3. Pick a provider

# %%
from openai import OpenAI

PROVIDERS = {
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",   "env_var": "DASHSCOPE_API_KEY", "model": "qwen-plus"},
    "deepseek": {"base_url": "https://api.deepseek.com",                            "env_var": "DEEPSEEK_API_KEY",  "model": "deepseek-chat"},
    "openai":   {"base_url": None,                                                 "env_var": "OPENAI_API_KEY",    "model": "gpt-4o-mini"},
}

provider = next((p for p, cfg in PROVIDERS.items() if os.environ.get(cfg["env_var"])), None)
if provider is None:
    raise RuntimeError(f"No API key loaded. Set one of {KEY_VARS} and re-run section 2.")

cfg = PROVIDERS[provider]
client = OpenAI(api_key=os.environ[cfg["env_var"]], base_url=cfg["base_url"])
model = cfg["model"]

print(f"Using provider={provider}, model={model}")

# %% [markdown]
# ## 4. Stream a response
#
# Pass `stream=True` and iterate. The first chunk's `delta.content` is often
# `None` (it carries the role), so always guard with `if delta:`.

# %%
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain a PID controller in three sentences."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()  # trailing newline

# %% [markdown]
# ## 5. Get token counts with a streamed response
#
# By default, OpenAI-compatible streams don't include usage stats. Opt in with
# `stream_options={"include_usage": True}` — the **last** chunk then carries
# `chunk.usage`.

# %%
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Give me one sentence on integral wind-up."}],
    stream=True,
    stream_options={"include_usage": True},
)

usage = None
for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    if chunk.usage:
        usage = chunk.usage
print()
print("usage:", usage)

# %% [markdown]
# ## 6. A unified streaming generator
#
# Same pattern wrapped in a function that yields text chunks regardless of provider.

# %%
from typing import Iterator

def stream_chat(prompt: str, provider: str = "qwen", model: str | None = None, system: str | None = None) -> Iterator[str]:
    cfg = PROVIDERS[provider]
    c = OpenAI(api_key=os.environ[cfg["env_var"]], base_url=cfg["base_url"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    s = c.chat.completions.create(model=model or cfg["model"], messages=messages, stream=True)
    for chunk in s:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

for token in stream_chat("Say hi in one sentence.", provider=provider):
    print(token, end="", flush=True)
print()

# %% [markdown]
# ## Next
#
# - [Tool Use](https://markjh2001.github.io/LLM-Control-Tutorial/api/tool-use/) — let the model call your functions.
