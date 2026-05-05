# Beginner Template

A reference for students who are new to Python or programming in general. It walks from "I have a laptop" to "my Streamlit or Gradio app is running on `localhost`". This is **only a reference** — you are welcome to use a different IDE, project layout, or web framework. What matters is that the LLM-agent loop closes through a deterministic Python evaluator.

## 1. Install the tools

### Python 3.10 or newer

| OS | How |
|---|---|
| **Windows** | Download the installer from [python.org/downloads](https://www.python.org/downloads/). On the first install screen, **tick "Add python.exe to PATH"** before clicking Install. |
| **macOS** | Download from [python.org/downloads](https://www.python.org/downloads/), or `brew install python@3.12` if you already use Homebrew. |
| **Linux** | Use your distro's package manager: `sudo apt install python3 python3-venv python3-pip` (Ubuntu/Debian) / `sudo dnf install python3` (Fedora). |

Verify in a fresh terminal:

```bash
python --version          # Windows usually
python3 --version         # macOS / Linux usually
```

You should see `Python 3.10.x` or higher.

### VS Code + Python extension

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Open VS Code, click the Extensions icon in the left sidebar, search **Python**, install the one from Microsoft.
3. Open the folder you'll work in: *File → Open Folder…*

### Pick the Python interpreter inside VS Code

Open the Command Palette (**Ctrl + Shift + P** on Windows/Linux, **Cmd + Shift + P** on macOS) → type *Python: Select Interpreter* → choose the one you'll create in step 3. (Comes back to bite you if you skip it — VS Code will run scripts against the system Python instead of your project's venv.)

## 2. Project file structure

```
aircraft-control-agent/
├── .env                  # your SJTU_API_KEY, never committed to git
├── requirements.txt      # Python dependencies
├── backend.py            # SJTU client + python-control evaluator + agent loop
└── app.py                # Streamlit OR Gradio UI
```

Four files, that's the whole project.

## 3. Set up the environment

In a terminal, inside the project folder:

```bash
# Create a virtual environment (an isolated Python install for this project)
python -m venv venv
```

Activate it — the command depends on your OS / shell:

```bash
# macOS / Linux (bash, zsh):
source venv/bin/activate

# Windows (cmd.exe):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should now see `(venv)` at the start of your prompt. Then install the deps:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
openai>=1.60
python-dotenv>=1.0
control>=0.10
matplotlib>=3.9
streamlit>=1.30
# Or, if you choose Gradio instead:
# gradio>=4.0
```

### `.env`

Create a file literally named `.env` (no extension) in the project folder, with your SJTU API key:

```
SJTU_API_KEY=sk-...your-key...
```

`python-dotenv` will pick it up automatically.

## 4. `backend.py` — skeleton

```python
"""
Backend: SJTU client + python-control evaluator + agent loop.
Fill in the TODOs for your chosen sub-option (a)–(d).
"""
import json
import os
import numpy as np
import control as ctrl
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- SJTU client ----------
client = OpenAI(
    api_key=os.environ["SJTU_API_KEY"],
    base_url="https://models.sjtu.edu.cn/api/v1",
)
MODEL = "deepseek-chat"

# ---------- Plant ----------
def build_plant() -> ctrl.TransferFunction:
    """
    TODO: build the aircraft attitude-control plant from Figure 2 of the project page,
    using the parameter table (Ks, K, K1, K2, Kt, Ra, La, Ki [motor torque constant],
    Kb, Jm, JL, Bm, BL, N). Return a control.TransferFunction — the open-loop G(s)
    the controller sees. Note: this Ki (torque constant) is the plant parameter; the
    controller's integral gain is a different number you'll let the agent propose.
    """
    raise NotImplementedError

# ---------- Spec for the chosen sub-option ----------
SPECS = {
    # TODO: fill in the targets for your sub-option, e.g. for (a):
    # "ess_ramp_max": 0.000443,
    # "overshoot_max": 5.0,
    # "rise_time_max": 0.005,
    # "settling_time_max": 0.005,
}

# ---------- Evaluator ----------
def evaluate(gains: dict) -> dict:
    """
    Build the closed loop, check stability, return a dict with the metrics
    your sub-option cares about plus a "passes" boolean.

    TODO:
      1. Build controller K(s) from `gains` (e.g. {"kp": ..., "ki": ..., "kd": ...})
      2. Form the closed-loop T(s) = K G / (1 + K G) using ctrl.feedback
      3. Check stability via ctrl.poles(T)
      4. Extract metrics:
         - sub-options (a)–(c): use ctrl.step_info(T) for settling, overshoot, rise time
         - sub-option (d):     use ctrl.margin(L) and ctrl.bandwidth(T)
      5. Compare each metric to SPECS, set "passes" to True iff every spec is met.
    """
    raise NotImplementedError

def passes(perf: dict) -> bool:
    return bool(perf.get("passes"))

# ---------- Prompts (this is where the project actually happens) ----------
#
# The loop above is mechanical. The interesting part of the project is what
# you put HERE — your control-theory expertise, encoded as text the model
# can read on every iteration.

SYSTEM_PROMPT = """You are a control-systems engineer.

TODO — fill this in with everything the model needs to design the controller well:

  - Describe the plant (transfer function, order, dominant dynamics from Figure 2)
  - State the controller form (PD / PI / PID) and which gains to propose
  - List the design spec for your chosen sub-option
  - Encode the control-theory intuition you learned in class — how each gain
    shapes the closed-loop response (rise time, overshoot, steady-state error,
    phase margin, etc.). The model can only reason from what you write here.

End with:
Respond ONLY with JSON of the form {"kp": <number>, "ki": <number>, "kd": <number>}.
No prose.
"""


def feedback_msg(i: int, gains: dict, perf: dict) -> str:
    """
    Turn the evaluator's verdict into a SHORT, ACTIONABLE user message.

    What to put here:
      - How far each metric is from its target (the spec gap).
      - A direction grounded in your control-theory training — which gain to
        nudge, which way, and roughly by how much.

    The model only proposes new gains from this text plus the running history.
    Keep it tight: it needs the next move, not a textbook.
    """
    # TODO
    return f"Iter {i}: {gains} -> {perf}. Propose new values."

# ---------- Agent loop ----------
MAX_ITER = 10

def run_agent():
    """
    Generator. Yields one dict per iteration; the frontend renders these live.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Propose initial gains."},
    ]
    for i in range(1, MAX_ITER + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = resp.choices[0].message.content
        gains = json.loads(raw)
        perf = evaluate(gains)
        yield {"iter": i, "gains": gains, "perf": perf, "raw": raw}
        if passes(perf):
            return
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback_msg(i, gains, perf)})
```

## 5. `app.py` — Streamlit skeleton

Run with `streamlit run app.py`. It opens a browser at `http://localhost:8501`.

```python
"""
Streamlit frontend. Run:  streamlit run app.py
"""
import streamlit as st
import matplotlib.pyplot as plt
import control as ctrl
from backend import run_agent, passes

st.title("Aircraft Attitude Control Agent")

# TODO: render a description of the sub-option you chose and the targets.
st.markdown("""
**Sub-option chosen**: TODO  
**Spec**: TODO (settling time, overshoot, …)
""")

if st.button("Run agent"):
    placeholder = st.empty()  # for the live iteration log
    log_lines = []
    converged = False
    for step in run_agent():
        log_lines.append(f"Iter {step['iter']}: gains={step['gains']} → {step['perf']}")
        placeholder.code("\n".join(log_lines))
        if passes(step["perf"]):
            converged = True
            st.success(f"Converged at iteration {step['iter']}")
            # TODO: plot the final step response with ctrl.step_response + matplotlib,
            # then st.pyplot(fig).
            break
    if not converged:
        st.error("Did not converge within MAX_ITER iterations.")
```

## 6. `app.py` — Gradio skeleton

Run with `python app.py`. Gradio prints the local URL (usually `http://127.0.0.1:7860`).

```python
"""
Gradio frontend. Run:  python app.py
"""
import gradio as gr
import matplotlib.pyplot as plt
import control as ctrl
from backend import run_agent, passes

def run_and_format():
    log_lines = []
    final_fig = None
    for step in run_agent():
        log_lines.append(f"Iter {step['iter']}: gains={step['gains']} → {step['perf']}")
        if passes(step["perf"]):
            log_lines.append(f"Converged at iteration {step['iter']}")
            # TODO: build the step-response figure and assign to final_fig.
            # fig, ax = plt.subplots()
            # ...
            # final_fig = fig
            break
    return "\n".join(log_lines), final_fig

iface = gr.Interface(
    fn=run_and_format,
    inputs=[],
    outputs=[
        gr.Textbox(label="Iteration log", lines=20),
        gr.Plot(label="Step response"),
    ],
    title="Aircraft Attitude Control Agent",
    description="TODO: describe your chosen sub-option and the spec.",
)

if __name__ == "__main__":
    iface.launch()
```

## 7. Run it

With your venv activated and `.env` filled in:

```bash
# Streamlit:
streamlit run app.py
# (opens http://localhost:8501)

# Gradio:
python app.py
# (URL printed in the terminal, usually http://127.0.0.1:7860)
```

Click **Run agent** and watch the iteration log fill in.

## Reminder

This page is a reference template — nothing here is mandatory. PyCharm instead of VS Code, a different project layout, FastAPI + React, or even an `ipywidgets` Jupyter notebook are all acceptable. The only real requirements are the ones on the [project page](llm-agent.md): an LLM-agent loop closed by a Python evaluator, with a UI exposing it.
