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
# # Laplace Transforms — runnable notebook
#
# Companion to [**LLM + Control → Laplace Transforms**](https://markjh2001.github.io/LLM-Control-Tutorial/control/laplace/).
#
# Three canonical textbook problems, each solved with a plain prompt at `temperature=0`:
#
# - **Homework 2, Problem 3.3(b)** — forward Laplace transform
# - **Homework 3, Problem 3.7(h)** — inverse via partial-fraction expansion
# - **Homework 3, Problem 3.9(e)** — second-order ODE, solved via Laplace

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
# ## 4. Forward transforms — Homework 2, Problem 3.3(b)
#
# Find the Laplace transform F(s) = L{f(t)} for:
#
#     f(t) = cos(2t) + 4 sin(5t) + e^(-2t) cos(7t)
#
# Textbook answer (from the standard Laplace table plus the frequency-shift property):
#
#     F(s) = s/(s² + 4) + 20/(s² + 25) + (s + 2)/((s + 2)² + 49)

# %%
EXPR_33B = "cos(2t) + 4 sin(5t) + e^(-2t) cos(7t)"

# %% [markdown]
# ## 5. Ask the model
#
# Pose the problem in the textbook's own wording — no custom system prompt, no
# formatting instructions. Just the question as a student would ask it.

# %%
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Find the Laplace transform of the following time function: f(t) = {EXPR_33B}"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)

# %% [markdown]
# The model's final boxed F(s) should be algebraically equivalent to the textbook answer. On a canonical problem like this, `temperature=0` plain prompting is sufficient, and the match is its own proof.

# %% [markdown]
# ## 6. Inverse transforms — Homework 3, Problem 3.7(h)
#
# Same textbook, reversed direction: given F(s), find f(t) via partial-fraction
# decomposition.
#
#     F(s) = (3s + 2) / (s² - s - 2)
#
# Textbook answer (s² - s - 2 factors as (s+1)(s-2); decomposes as (1/3)/(s+1) + (8/3)/(s-2)):
#
#     f(t) = (1/3) e^(-t) + (8/3) e^(2t)

# %%
FS_37H = "(3s + 2) / (s^2 - s - 2)"

# %% [markdown]
# ## 7. Ask the model — inverse direction
#
# Same approach as Problem 3.3: use the textbook's own wording. The problem
# statement already says *"using partial-fraction expansions"*, so the method
# requirement is baked into the prompt itself — no system-prompt engineering
# needed to force the model to show PF.

# %%
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Find the time function corresponding to the following Laplace transform using partial-fraction expansion: F(s) = {FS_37H}"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)

# %% [markdown]
# The model's response includes the partial-fraction decomposition alongside the final f(t). Both should match the textbook — so the *method* is verified, not just the answer.

# %% [markdown]
# ## 8. Solving ODEs — Homework 3, Problem 3.9(e)
#
# Step up: solve an ODE using Laplace transforms. The pattern chains three stages
# — transform both sides, apply initial conditions and solve for Y(s),
# partial-fraction, then inverse-transform back to y(t).
#
#     y''(t) + 2 y'(t) + 2 y(t) = 5 sin(t),  y(0) = y'(0) = 0
#
# Textbook answer:
#
#     y(t) = sin(t) - 2 cos(t) + 2 e^(-t) cos(t) + e^(-t) sin(t)

# %%
ODE_39E = "y''(t) + 2 y'(t) + 2 y(t) = 5 sin(t), given y(0) = y'(0) = 0"

# %% [markdown]
# ## 9. Ask the model — ODE solver

# %%
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Solve the following ODE using Laplace transforms: {ODE_39E}"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)

# %% [markdown]
# The model chains Laplace → partial fractions → inverse Laplace and arrives at the textbook answer, and typically volunteers a verification that both initial conditions hold — a detail you'd want on a homework submission but didn't prompt for.
#
# ## Next
#
# Back to the [LLM + Control overview](https://markjh2001.github.io/LLM-Control-Tutorial/control/).
