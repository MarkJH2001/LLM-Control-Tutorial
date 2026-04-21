# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# - **Colab:** open the 🔑 **Secrets** panel in the left sidebar and add at least
#   one of `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `DASHSCOPE_API_KEY`.
#   **Also flip the "Notebook access" toggle on that secret** — the first time a
#   cell tries to read it, Colab pops a dialog asking for permission; if you
#   dismiss the dialog the secret stays inaccessible.
# - **Deepnote / local:** set the same env var in your workspace, or accept the
#   prompt below.
#
# The key is never written back to the notebook.

# %%
import os

KEY_VARS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")

# Try Colab Secrets. Each get() can raise if the secret is missing OR if the
# per-notebook access toggle is off — handle each key independently so one
# missing key doesn't skip the others.
try:
    from google.colab import userdata  # Colab only
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

# Local: pick up .env if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# If nothing was loaded, prompt for one (works in Deepnote / local too).
if not any(os.environ.get(k) for k in KEY_VARS):
    from getpass import getpass
    which = input("Which provider? (qwen / deepseek / openai) [qwen]: ").strip().lower() or "qwen"
    os.environ[{"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}[which]] = getpass(
        f"Paste your key: "
    )

print("Keys detected:", [k for k in KEY_VARS if os.environ.get(k)])

# %% [markdown]
# ## 3. Pick a provider
#
# Auto-detected from whichever key is loaded. Override `provider` below if you
# want to force a specific one.

# %%
from openai import OpenAI

PROVIDERS = {
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",   "env_var": "DASHSCOPE_API_KEY", "model": "qwen-plus"},
    "deepseek": {"base_url": "https://api.deepseek.com",                            "env_var": "DEEPSEEK_API_KEY",  "model": "deepseek-chat"},
    "openai":   {"base_url": None,                                                 "env_var": "OPENAI_API_KEY",    "model": "gpt-4o-mini"},
}

# Auto-detect based on which key is set. Set `provider = "qwen"` (etc.) to override.
provider = next((p for p, cfg in PROVIDERS.items() if os.environ.get(cfg["env_var"])), None)
if provider is None:
    raise RuntimeError(f"No API key loaded. Set one of {KEY_VARS} and re-run section 2.")

cfg = PROVIDERS[provider]
client = OpenAI(api_key=os.environ[cfg["env_var"]], base_url=cfg["base_url"])
model = cfg["model"]

print(f"Using provider={provider}, model={model}")

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
