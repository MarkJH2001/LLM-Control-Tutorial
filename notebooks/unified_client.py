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
# # Unified Client — runnable notebook
#
# Companion to [**Unified Client**](https://markjh2001.github.io/LLM-Control-Tutorial/api/unified-client/).

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet openai python-dotenv

# %% [markdown]
# ## 2. Load an API key
#
# - **Colab:** 🔑 panel → add one of `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`. Flip "Notebook access" ON.
# - **Deepnote:** project settings → Environment variables → add the same.
# - **Local:** put it in `.env`; `python-dotenv` picks it up.

# %%
import os

KEY_VARS = ("SJTU_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")

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
    which = input("Which provider? (sjtu / qwen / deepseek / openai) [sjtu]: ").strip().lower() or "sjtu"
    os.environ[{"sjtu": "SJTU_API_KEY", "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}[which]] = getpass("Paste your key: ")

print("Keys detected:", [k for k in KEY_VARS if os.environ.get(k)])

# %% [markdown]
# ## 3. Define the unified chat() function
#
# One `openai` client wrapping three providers.

# %%
from dataclasses import dataclass
from openai import OpenAI


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str | None
    env_var: str
    default_model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "sjtu":     ProviderConfig(base_url="https://models.sjtu.edu.cn/api/v1",                   env_var="SJTU_API_KEY",      default_model="deepseek-chat"),
    "qwen":     ProviderConfig(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",   env_var="DASHSCOPE_API_KEY", default_model="qwen-plus"),
    "deepseek": ProviderConfig(base_url="https://api.deepseek.com",                            env_var="DEEPSEEK_API_KEY",  default_model="deepseek-chat"),
    "openai":   ProviderConfig(base_url=None,                                                 env_var="OPENAI_API_KEY",    default_model="gpt-4o-mini"),
}


def chat(prompt: str, provider: str = "sjtu", model: str | None = None, system: str | None = None) -> str:
    cfg = PROVIDERS[provider]
    client = OpenAI(api_key=os.environ[cfg.env_var], base_url=cfg.base_url)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model or cfg.default_model,
        messages=messages,
    )
    return resp.choices[0].message.content

# %% [markdown]
# ## 4. Pick the provider we have a key for and call it
#
# The same function call works for any provider — only the `provider` arg changes.

# %%
provider = next((p for p, cfg in PROVIDERS.items() if os.environ.get(cfg.env_var)), None)
if provider is None:
    raise RuntimeError(f"No API key loaded. Set one of {KEY_VARS} and re-run section 2.")

print(f"--- plain chat ({provider}) ---")
print(chat("Say hi in one short sentence.", provider))

print()
print(f"--- with system prompt ({provider}) ---")
print(chat("What is a PID controller?", provider, system="Answer in one sentence."))

# %% [markdown]
# ## 5. Swapping providers is one argument
#
# If you have more than one key loaded, you can call `chat(..., "openai")`,
# `chat(..., "deepseek")`, `chat(..., "qwen")` — the client code is unchanged.

# %%
available = [p for p, cfg in PROVIDERS.items() if os.environ.get(cfg.env_var)]
print("Available providers in this session:", available)
for p in available:
    print(f"\n--- {p} ---")
    print(chat("Say hi in one short sentence.", p))

# %% [markdown]
# ## Next
#
# - [Streaming](https://markjh2001.github.io/LLM-Control-Tutorial/api/streaming/) — tokens as they're generated.
# - [Tool Use](https://markjh2001.github.io/LLM-Control-Tutorial/api/tool-use/) — let the model call your functions.
