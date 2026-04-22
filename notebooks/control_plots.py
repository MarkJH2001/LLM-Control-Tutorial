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
# # Control Plots — runnable notebook
#
# Companion to [**LLM + Control → Control Plots**](https://markjh2001.github.io/LLM-Control-Tutorial/control/plots/).
#
# Give the model three plotting tools (root locus, Bode, Nyquist) backed by `python-control`. The model parses a natural-language request — "draw the Bode plot of $G(s) = (s+2)/(s^3 + 5s^2 + 8s + 4)$" — into numerator/denominator coefficient lists and calls the matching tool. This is the first page in the LLM + Control section where a tool is genuinely required: plain prompts can do the algebra, but they can't produce a plot.

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet openai python-dotenv control matplotlib

# %% [markdown]
# ## 2. Load an API key

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
# ## 3. Pick a provider

# %%
from openai import OpenAI

PROVIDERS = {
    "sjtu":     {"base_url": "https://models.sjtu.edu.cn/api/v1",                   "env_var": "SJTU_API_KEY",      "model": "deepseek-chat"},
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
# ## 4. Three plotting tools backed by `python-control`
#
# Each tool takes numerator and denominator coefficient lists (in descending powers of $s$),
# builds a transfer function, renders the plot inline, and returns a short confirmation
# string for the model to read.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from control import bode, nyquist, root_locus, tf

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


def _fmt(coeffs: list[float]) -> str:
    return "[" + ", ".join(f"{c:g}" for c in coeffs) + "]"


def plot_root_locus(numerator: list[float], denominator: list[float], title: str = "Root Locus") -> str:
    system = tf(numerator, denominator)
    fig, ax = plt.subplots(figsize=(6, 5))
    root_locus(system, ax=ax)
    ax.set_title(title)
    fig.savefig(PLOT_DIR / "root_locus.svg", bbox_inches="tight", dpi=120)
    plt.show()
    return f"Root locus rendered for numerator={_fmt(numerator)}, denominator={_fmt(denominator)}."


def plot_bode(numerator: list[float], denominator: list[float], title: str = "Bode Plot") -> str:
    system = tf(numerator, denominator)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))
    bode(system, ax=axes)
    fig.suptitle(title)
    fig.savefig(PLOT_DIR / "bode.svg", bbox_inches="tight", dpi=120)
    plt.show()
    return f"Bode plot rendered for numerator={_fmt(numerator)}, denominator={_fmt(denominator)}."


def plot_nyquist(numerator: list[float], denominator: list[float], title: str = "Nyquist Plot") -> str:
    system = tf(numerator, denominator)
    fig, ax = plt.subplots(figsize=(6, 5))
    nyquist(system, ax=ax)
    ax.set_title(title)
    fig.savefig(PLOT_DIR / "nyquist.svg", bbox_inches="tight", dpi=120)
    plt.show()
    return f"Nyquist plot rendered for numerator={_fmt(numerator)}, denominator={_fmt(denominator)}."


TOOLS_BY_NAME = {
    "plot_root_locus": plot_root_locus,
    "plot_bode": plot_bode,
    "plot_nyquist": plot_nyquist,
}

# %% [markdown]
# ## 5. Tool schemas
#
# The three schemas are parallel — only the name, description, and default title differ.

# %%
def _schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numerator polynomial coefficients, descending powers of s.",
                    },
                    "denominator": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Denominator polynomial coefficients, descending powers of s.",
                    },
                    "title": {"type": "string", "description": "Optional plot title."},
                },
                "required": ["numerator", "denominator"],
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("plot_root_locus", "Plot the root locus of a transfer function G(s) = N(s)/D(s)."),
    _schema("plot_bode",       "Plot the Bode magnitude and phase of a transfer function."),
    _schema("plot_nyquist",    "Plot the Nyquist diagram of a transfer function."),
]

# %% [markdown]
# ## 6. The agent loop
#
# Same shape as the loop in [Tool Use](https://markjh2001.github.io/LLM-Control-Tutorial/api/tool-use/):
# iterate `chat.completions.create(... tools=TOOL_SCHEMAS ...)`, execute any tool calls the
# model returns, feed the results back, stop when the model answers without a tool call.

# %%
import json


SYSTEM = (
    "You are a control-systems assistant with three plotting tools: "
    "plot_root_locus, plot_bode, and plot_nyquist. "
    "When the user gives you a transfer function G(s) = N(s)/D(s), "
    "parse N and D into polynomial coefficient lists in descending powers of s, "
    "then call the appropriate tool. "
    "After the tool runs, briefly describe what the plot shows."
)


def run_agent(user_message: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = TOOLS_BY_NAME[tc.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return messages[-1].get("content", "")

# %% [markdown]
# ## 7. Example 1 — Root locus

# %%
print(run_agent("Plot the root locus of G(s) = 1 / (s^2 + 2s + 1)."))

# %% [markdown]
# ## 8. Example 2 — Bode plot

# %%
print(run_agent("Draw the Bode plot of G(s) = (s + 2) / (s^3 + 5 s^2 + 8 s + 4)."))

# %% [markdown]
# ## 9. Example 3 — Nyquist plot

# %%
print(run_agent("Generate the Nyquist plot of G(s) = 10 / (s (s + 1) (s + 5))."))

# %% [markdown]
# ## Next
#
# Back to the [LLM + Control overview](https://markjh2001.github.io/LLM-Control-Tutorial/control/).
