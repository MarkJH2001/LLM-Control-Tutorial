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
# # Multi-agent — runnable notebook
#
# Companion to [**Agentic Workflows → Multi-agent**](https://markjh2001.github.io/LLM-Control-Tutorial/agents/multi-agent/).

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
# ## 4. Writer + Critic — structured handoff
#
# Two agents, two system prompts, one JSON handoff between them.

# %%
import json


def writer_agent(task: str, feedback: str | None = None) -> str:
    messages = [
        {
            "role": "system",
            "content": "You write concise, accurate technical explanations. One paragraph, no bullets.",
        },
        {"role": "user", "content": task},
    ]
    if feedback:
        messages.append(
            {"role": "user", "content": f"Revise the previous draft. Issues to fix: {feedback}"}
        )
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content


def critic_agent(task: str, draft: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You review technical explanations for accuracy and clarity. "
                'Reply with JSON: {"approved": bool, "issues": [string]}. '
                "Approve only if the draft is factually correct and clear. "
                "Issues must be specific and actionable."
            ),
        },
        {"role": "user", "content": f"Task: {task}\n\nDraft: {draft}"},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

# %% [markdown]
# ## 5. Glue: iterate until approved or cap

# %%
def collaborate(task: str, max_rounds: int = 3) -> str:
    feedback = None
    draft = ""
    for round_num in range(max_rounds):
        draft = writer_agent(task, feedback)
        print(f"--- round {round_num + 1} draft ---")
        print(draft)
        review = critic_agent(task, draft)
        print(f"--- round {round_num + 1} review --- {review}")
        if review.get("approved"):
            return draft
        feedback = "; ".join(review.get("issues", []))
    return draft  # give up after max_rounds — return last draft

# %% [markdown]
# ## 6. Run it

# %%
final = collaborate("Explain a PID controller in one paragraph.")
print()
print("=== FINAL ===")
print(final)

# %% [markdown]
# ## Next
#
# - [LLM + Control](https://markjh2001.github.io/LLM-Control-Tutorial/control/) — applying these patterns to control-systems problems.
