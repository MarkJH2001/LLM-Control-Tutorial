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
# # Agent Loops — runnable notebook
#
# Companion to [**Agentic Workflows → Loops**](https://markjh2001.github.io/LLM-Control-Tutorial/agents/loops/).

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
    which = input("Which provider? (openai / deepseek / qwen) [openai]: ").strip().lower() or "openai"
    os.environ[{"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY"}[which]] = getpass("Paste your key: ")

print("Keys detected:", [k for k in KEY_VARS if os.environ.get(k)])

# %% [markdown]
# ## 3. Pick a provider

# %%
from openai import OpenAI

PROVIDERS = {
    "openai":   {"base_url": None,                                                 "env_var": "OPENAI_API_KEY",    "model": "gpt-4o-mini"},
    "deepseek": {"base_url": "https://api.deepseek.com",                            "env_var": "DEEPSEEK_API_KEY",  "model": "deepseek-chat"},
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",   "env_var": "DASHSCOPE_API_KEY", "model": "qwen-plus"},
}

provider = next((p for p, cfg in PROVIDERS.items() if os.environ.get(cfg["env_var"])), None)
if provider is None:
    raise RuntimeError(f"No API key loaded. Set one of {KEY_VARS} and re-run section 2.")

cfg = PROVIDERS[provider]
client = OpenAI(api_key=os.environ[cfg["env_var"]], base_url=cfg["base_url"])
model = cfg["model"]

print(f"Using provider={provider}, model={model}")

# %% [markdown]
# ## 4. Define the math tools

# %%
import json


def add(a: float, b: float) -> float:      return a + b
def subtract(a: float, b: float) -> float: return a - b
def multiply(a: float, b: float) -> float: return a * b
def divide(a: float, b: float) -> float:   return a / b

TOOLS_BY_NAME = {
    "add": add, "subtract": subtract, "multiply": multiply, "divide": divide,
}


def _two_num_schema(name: str, desc: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }

TOOL_SCHEMAS = [
    _two_num_schema("add", "Add two numbers."),
    _two_num_schema("subtract", "Subtract b from a."),
    _two_num_schema("multiply", "Multiply two numbers."),
    _two_num_schema("divide", "Divide a by b."),
]


def dispatch(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    return str(TOOLS_BY_NAME[name](**args))

# %% [markdown]
# ## 5. The agent loop
#
# Plan → act → observe → repeat, with an **iteration cap** so a confused
# model can't loop forever.

# %%
def run_agent(goal: str, max_iterations: int = 10) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math agent. Use the tools to compute the answer. "
                "Think briefly before each tool call and give a short final answer."
            ),
        },
        {"role": "user", "content": goal},
    ]

    for step in range(max_iterations):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content  # model decided it is done

        for tc in msg.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": dispatch(tc),
                }
            )

    raise RuntimeError(f"agent exceeded {max_iterations} iterations without finishing")

# %% [markdown]
# ## 6. Run it
#
# Multi-step arithmetic: the error of a system with current state `3.2`
# relative to setpoint `5.0`, as a percentage.

# %%
goal = (
    "A system has a current state of 3.2 and a setpoint of 5.0. "
    "What is the error as a percentage of the setpoint?"
)
print(run_agent(goal))

# %% [markdown]
# ## Next
#
# - [Memory](https://markjh2001.github.io/LLM-Control-Tutorial/agents/memory/) — what to do when tool outputs start eating the context window.
# - [Multi-agent](https://markjh2001.github.io/LLM-Control-Tutorial/agents/multi-agent/) — coordinating specialized agents.
