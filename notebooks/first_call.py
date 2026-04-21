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
# # First Call — runnable notebook
#
# Companion to the tutorial page [**Calling the API → First Call**](https://markjh2001.github.io/LLM-Control-Tutorial/api/first-call/).
#
# Pick one provider, paste your key (or set it in Colab Secrets), and run the cells top to bottom.

# %% [markdown]
# ## 1. Install dependencies
#
# Only needed on fresh Colab / Deepnote sessions.

# %%
# %pip install --quiet openai python-dotenv

# %% [markdown]
# ## 2. Load an API key
#
# - **Colab:** open the 🔑 **Secrets** panel in the left sidebar and add one of
#   `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `DASHSCOPE_API_KEY`. The cell below
#   picks it up automatically.
# - **Deepnote / local:** set the same env var in your workspace, or accept the
#   prompt below.
#
# The key is never written back to the notebook.

# %%
import os

try:
    from google.colab import userdata  # Colab only
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        v = userdata.get(k)
        if v:
            os.environ[k] = v
except Exception:
    pass

if not any(os.environ.get(k) for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")):
    from getpass import getpass
    os.environ["OPENAI_API_KEY"] = getpass("Paste your OPENAI_API_KEY: ")

# %% [markdown]
# ## 3. Pick a provider
#
# Uncomment exactly one block. All three use the `openai` SDK — only `base_url` and `model` change.

# %%
from openai import OpenAI

# ----- OpenAI (default) -----
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = "gpt-4o-mini"

# ----- DeepSeek -----
# client = OpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com",
# )
# model = "deepseek-chat"

# ----- Qwen -----
# client = OpenAI(
#     api_key=os.environ["DASHSCOPE_API_KEY"],
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# model = "qwen-plus"

# %% [markdown]
# ## 4. Hello world

# %%
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Say hi in one short sentence."},
    ],
)
print(resp.choices[0].message.content)

# %% [markdown]
# ## 5. Inspect the response

# %%
print("content:      ", resp.choices[0].message.content)
print("finish_reason:", resp.choices[0].finish_reason)
print("model:        ", resp.model)
print("usage:        ", resp.usage)

# %% [markdown]
# ## 6. Add a system prompt

# %%
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
        {"role": "user", "content": "What is a PID controller?"},
    ],
)
print(resp.choices[0].message.content)

# %% [markdown]
# ## Next
#
# - [Unified Client](https://markjh2001.github.io/LLM-Control-Tutorial/api/unified-client/) — one function covering all three providers.
# - [Streaming](https://markjh2001.github.io/LLM-Control-Tutorial/api/streaming/) — tokens as they're generated.
# - [Tool Use](https://markjh2001.github.io/LLM-Control-Tutorial/api/tool-use/) — let the model call your functions.
