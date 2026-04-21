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
# # PID Tuning — runnable notebook
#
# Companion to [**LLM + Control → PID Tuning**](https://markjh2001.github.io/LLM-Control-Tutorial/control/pid/).
#
# Two runs side-by-side on the same problem:
#
# - **Open-loop**: one LLM call proposes (Kp, Ki). We evaluate after the fact. No feedback, no correction — like a chat window.
# - **Closed-loop**: the same model, looped with a deterministic evaluator. Each proposal is scored against the spec; if it fails, the metrics and the failure reason are fed back into the next turn. Same shape as closed-loop control: propose → measure → correct → propose again.
#
# Plant: $G(s) = 1/(s+1)$. PI controller. Spec: settling time ≤ 1 s, overshoot ≤ 5 %, zero steady-state error.

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet openai python-dotenv control matplotlib

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
# ## 4. Plant, spec, and the evaluator
#
# Plant $G(s) = 1/(s+1)$. We want the closed loop under a PI controller
# $K(s) = K_p + K_i/s$ to hit settling time ≤ 1 s and overshoot ≤ 5 %,
# with zero steady-state error (the integral term handles that automatically).
#
# `evaluate_pi(kp, ki)` builds the closed loop, checks stability, and reads
# settling time + overshoot from `control.step_info`.

# %%
import numpy as np
import control as ctrl

PLANT_NUM = [1]
PLANT_DEN = [1, 1]
SPECS = {"settling_time_max": 1.0, "overshoot_max": 5.0}


def evaluate_pi(kp: float, ki: float) -> dict:
    G = ctrl.TransferFunction(PLANT_NUM, PLANT_DEN)
    K = ctrl.TransferFunction([kp, ki], [1, 0])
    T = ctrl.feedback(K * G, 1)
    poles = ctrl.poles(T)
    stable = bool(np.all(np.real(poles) < 0))
    if not stable:
        return {"stable": False, "settling_time": None, "overshoot": None,
                "steady_state_error": None, "closed_loop": T}
    info = ctrl.step_info(T)
    return {
        "stable": True,
        "settling_time": float(info["SettlingTime"]),
        "overshoot": float(info["Overshoot"]),
        "steady_state_error": float(1.0 - ctrl.dcgain(T)),
        "closed_loop": T,
    }


def passes(perf: dict) -> bool:
    return (perf["stable"]
            and perf["settling_time"] <= SPECS["settling_time_max"]
            and perf["overshoot"] <= SPECS["overshoot_max"])

# %% [markdown]
# ## 5. Shared LLM setup
#
# Same system prompt for both runs. The only thing that changes is whether we
# loop with evaluator feedback.

# %%
import json

SYSTEM_PROMPT = """You are a control-systems engineer. You are given a first-order plant
G(s) = 1 / (s + 1) and must design a PI controller K(s) = Kp + Ki/s that drives the
closed-loop step response to meet:

  - settling time <= 1 second (2% criterion)
  - overshoot <= 5%
  - zero steady-state error

Respond ONLY with JSON of the form {"kp": <number>, "ki": <number>}. No prose."""


def propose(messages: list) -> tuple[float, float, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    obj = json.loads(raw)
    return float(obj["kp"]), float(obj["ki"]), raw


def feedback_msg(iter_num: int, kp: float, ki: float, perf: dict) -> str:
    if not perf["stable"]:
        return (f"Iteration {iter_num}: Kp={kp}, Ki={ki} is UNSTABLE. Propose new (Kp, Ki).")
    issues = []
    if perf["settling_time"] > SPECS["settling_time_max"]:
        issues.append(f"settling time {perf['settling_time']:.3f} s exceeds spec {SPECS['settling_time_max']} s")
    if perf["overshoot"] > SPECS["overshoot_max"]:
        issues.append(f"overshoot {perf['overshoot']:.2f}% exceeds spec {SPECS['overshoot_max']}%")
    if not issues:
        return (f"Iteration {iter_num}: PASSES. settling={perf['settling_time']:.3f} s, "
                f"overshoot={perf['overshoot']:.2f}%.")
    return (f"Iteration {iter_num}: Kp={kp}, Ki={ki} failed — "
            f"{'; '.join(issues)}. Propose new (Kp, Ki) to fix this.")

# %% [markdown]
# ## 6. Helper: plot the step response

# %%
import matplotlib.pyplot as plt


def plot_step(T, title: str):
    t = np.linspace(0, 5.0, 600)
    t_out, y_out = ctrl.step_response(T, T=t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_out, y_out, lw=2)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.axhline(1.05, ls=":", color="red", lw=0.6)
    ax.axhline(0.95, ls=":", color="red", lw=0.6)
    ax.axvline(SPECS["settling_time_max"], ls="--", color="green", lw=0.8,
               label=f"settling spec ({SPECS['settling_time_max']} s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y(t)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.show()

# %% [markdown]
# ## 7. Run A — open-loop (single LLM call, no feedback)
#
# One call. Whatever the model proposes, we evaluate and report.
# There is no mechanism for the LLM to see the result or correct it.

# %%
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Propose (Kp, Ki)."},
]
kp_A, ki_A, raw_A = propose(messages)
perf_A = evaluate_pi(kp_A, ki_A)

print(f"LLM proposed:    Kp={kp_A}, Ki={ki_A}")
print(f"Evaluator says:  stable={perf_A['stable']}, "
      f"settling={perf_A['settling_time']:.3f} s, overshoot={perf_A['overshoot']:.2f}%")
print(f"Specs met?       {passes(perf_A)}")

plot_step(perf_A["closed_loop"], f"Open-loop run: Kp={kp_A}, Ki={ki_A}")

# %% [markdown]
# ## 8. Run B — closed-loop (agent loop with evaluator feedback)
#
# Same system prompt, same first proposal. The difference is that after each
# attempt we pipe the evaluator's verdict back into the conversation as a
# new user message, and the model reads it before proposing again.
# This is the LLM analogue of closed-loop control: propose → measure → correct.

# %%
MAX_ITER = 8

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Propose (Kp, Ki)."},
]

history = []
converged_perf = None

for i in range(1, MAX_ITER + 1):
    kp, ki, raw = propose(messages)
    perf = evaluate_pi(kp, ki)

    history.append({
        "iter": i, "kp": kp, "ki": ki,
        "stable": perf["stable"],
        "settling_time": perf["settling_time"],
        "overshoot": perf["overshoot"],
    })
    print(f"Iter {i}: Kp={kp}, Ki={ki} → stable={perf['stable']}, "
          f"settling={perf['settling_time']!s:<10s} overshoot={perf['overshoot']!s:<8s} "
          f"pass={passes(perf)}")

    if passes(perf):
        converged_perf = perf
        break

    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": feedback_msg(i, kp, ki, perf)})

# %%
if converged_perf is not None:
    last = history[-1]
    plot_step(converged_perf["closed_loop"],
              f"Closed-loop run (converged at iter {last['iter']}): Kp={last['kp']}, Ki={last['ki']}")
else:
    print(f"Did not converge in {MAX_ITER} iterations.")

# %% [markdown]
# ## 9. What changed
#
# Both runs started from the same (Kp, Ki) — the model's first guess. In Run A
# that guess is final. In Run B, the evaluator reports "settling time 1.61 s
# exceeds spec 1 s", the model reads that, and its next proposal (Kp=6, Ki=5)
# halves the settling time and passes. Same model, same system prompt, same
# temperature. The only difference is the **loop**.
#
# ## Next
#
# Back to the [LLM + Control overview](https://markjh2001.github.io/LLM-Control-Tutorial/control/).
