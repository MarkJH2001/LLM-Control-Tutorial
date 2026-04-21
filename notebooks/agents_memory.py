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
# # Agent Memory — runnable notebook
#
# Companion to [**Agentic Workflows → Memory**](https://markjh2001.github.io/LLM-Control-Tutorial/agents/memory/).

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
# ## 4. Working memory — window + summarize
#
# `summarize()` uses the model to condense older turns; `compact_messages()`
# replaces everything except the last `keep_last` turns with that summary.

# %%
def summarize(old_messages: list[dict]) -> str:
    """Ask the model for a short summary of a run of messages."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following assistant/user/tool exchange in under 200 words. "
                    "Preserve any numbers, decisions, and named entities verbatim."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    f"[{m['role']}] {m.get('content') or m.get('tool_calls')}"
                    for m in old_messages
                ),
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def compact_messages(
    messages: list[dict],
    keep_last: int = 6,
    threshold: int = 20,
) -> list[dict]:
    """If history is longer than `threshold`, summarize everything except the last `keep_last`."""
    if len(messages) <= threshold:
        return messages

    system = messages[0]
    middle = messages[1:-keep_last]
    tail = messages[-keep_last:]

    summary = summarize(middle)
    return [
        system,
        {"role": "user", "content": f"[summary of earlier turns]\n{summary}"},
        *tail,
    ]

# %% [markdown]
# ## 5. Demo compaction on a fake long history

# %%
fake_history = [{"role": "system", "content": "You are a control-systems agent."}]
for i in range(25):
    fake_history.append({"role": "user", "content": f"Turn {i}: tune the PID with Kp={i * 0.1:.1f}."})
    fake_history.append({"role": "assistant", "content": f"Kp={i * 0.1:.1f} acknowledged."})

print(f"Original history length: {len(fake_history)} messages")
compacted = compact_messages(fake_history, keep_last=4, threshold=10)
print(f"Compacted length:        {len(compacted)} messages")
print()
print("Summary injected:")
print(compacted[1]["content"][:400], "...")

# %% [markdown]
# ## 6. External memory — JSON-file state
#
# For facts that shouldn't live in context (user preferences, tuned gains, etc.)
# expose a simple key-value store as two tools the agent can call.

# %%
import json
from pathlib import Path

STATE_FILE = Path("./.agent_state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def get_memory(key: str) -> str:
    """Look up a saved value by key."""
    return str(load_state().get(key, ""))


def set_memory(key: str, value: str) -> str:
    """Save a value under a key. Overwrites any existing value."""
    state = load_state()
    state[key] = value
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return "ok"

# Demo (no API calls needed for this cell).
print(set_memory("preferred_gains", "Kp=1.2, Ki=0.1, Kd=0.01"))
print("preferred_gains =", get_memory("preferred_gains"))
print("unknown_key     =", repr(get_memory("unknown_key")))

# Clean up the demo file.
STATE_FILE.unlink(missing_ok=True)

# %% [markdown]
# ## Next
#
# - [Multi-agent](https://markjh2001.github.io/LLM-Control-Tutorial/agents/multi-agent/) — splitting work across specialized agents.
