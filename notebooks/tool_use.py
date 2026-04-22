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
# # Tool Use — runnable notebook
#
# Companion to [**Tool Use**](https://markjh2001.github.io/LLM-Control-Tutorial/api/tool-use/).

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet openai python-dotenv

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
# ## 4. Define a tool
#
# The model doesn't know the current time, so it must call our function.

# %%
import json
from datetime import datetime, timezone


def get_current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


TOOLS_BY_NAME = {"get_current_time": get_current_time}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current UTC time as an ISO-8601 string.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def dispatch(tool_call) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    return str(TOOLS_BY_NAME[name](**args))

# %% [markdown]
# ## 5. Run the conversation loop
#
# Keep calling the model, executing tools it requests, and feeding results
# back until it stops asking for tools.

# %%
messages = [{"role": "user", "content": "What time is it right now?"}]

for step in range(5):  # iteration cap
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,
    )
    msg = resp.choices[0].message
    messages.append(msg.model_dump(exclude_none=True))

    if not msg.tool_calls:
        break  # model is done

    for tc in msg.tool_calls:
        result = dispatch(tc)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )

print(messages[-1].get("content", "<no content>"))

# %% [markdown]
# ## 6. Add a second tool
#
# The same pattern scales: add one function, one schema, one registry entry.

# %%
def add(a: float, b: float) -> float:
    return a + b

TOOLS_BY_NAME["add"] = add
TOOL_SCHEMAS.append(
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First addend."},
                    "b": {"type": "number", "description": "Second addend."},
                },
                "required": ["a", "b"],
            },
        },
    },
)

messages = [{"role": "user", "content": "What is 17.3 + 5.9, and what time is it?"}]

for step in range(5):
    resp = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
    msg = resp.choices[0].message
    messages.append(msg.model_dump(exclude_none=True))
    if not msg.tool_calls:
        break
    for tc in msg.tool_calls:
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": dispatch(tc)})

print(messages[-1].get("content", "<no content>"))

# %% [markdown]
# ## Next
#
# - [Agentic Workflows → Loops](https://markjh2001.github.io/LLM-Control-Tutorial/agents/loops/) — wrap this pattern into a reusable agent with an iteration cap.
