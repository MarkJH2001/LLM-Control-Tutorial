"""Backend: SJTU client + python-control evaluator + PI agent loop.

Specs are passed in by the frontend so the user can edit them; the system prompt
holds the design intuition and the per-run target values are sent in the first
user message.
"""
import json
import os
import numpy as np
import control as ctrl
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["SJTU_API_KEY"],
    base_url="https://models.sjtu.edu.cn/api/v1",
)
MODEL = "deepseek-chat"


# ---------- Plant: G(s) = 1 / ((s+1)(s+2)) ----------
def build_plant() -> ctrl.TransferFunction:
    return ctrl.TransferFunction([1], [1, 3, 2])


G = build_plant()

# ---------- Default spec ----------
DEFAULT_SPECS = {
    "settling_time_max": 5.0,    # seconds (2 % criterion)
    "overshoot_max":     10.0,   # %
    "rise_time_max":     1.0,    # seconds
}


# ---------- Evaluator ----------
def evaluate(gains: dict, specs: dict) -> dict:
    kp = float(gains.get("kp", 0))
    ki = float(gains.get("ki", 0))
    K = ctrl.TransferFunction([kp, ki], [1, 0])
    T = ctrl.feedback(K * G, 1)
    if not np.all(np.real(ctrl.poles(T)) < 0):
        return {"stable": False, "passes": False, "T": T,
                "settling_time": None, "overshoot": None, "rise_time": None}
    info = ctrl.step_info(T)
    perf = {
        "stable": True,
        "settling_time": float(info["SettlingTime"]),
        "overshoot": float(info["Overshoot"]),
        "rise_time": float(info["RiseTime"]),
        "T": T,
    }
    perf["passes"] = (perf["settling_time"] <= specs["settling_time_max"]
                      and perf["overshoot"]     <= specs["overshoot_max"]
                      and perf["rise_time"]     <= specs["rise_time_max"])
    return perf


def passes(perf: dict) -> bool:
    return bool(perf.get("passes"))


# ---------- Prompts ----------
SYSTEM_PROMPT = """You are a control-systems engineer. Design a PI controller
G_c(s) = Kp + Ki/s for the plant

  G(s) = 1 / ((s+1)(s+2)) = 1 / (s^2 + 3 s + 2).

The user will give you specific design targets in the first message; aim to meet
all of them and return zero step steady-state error (Ki > 0 handles this when
the loop is stable).

Design intuition:

  - Closed-loop polynomial is s^3 + 3 s^2 + (2 + Kp) s + Ki.
  - The PI controller adds a zero at s = -Ki/Kp. Placing it near a plant pole
    (-1 or -2) gives a near-cancellation that keeps the integral mode FAST.
    Target Ki/Kp ~ 1-2.
  - Kp speeds the dominant complex pair: small Kp -> slow rise; large Kp ->
    underdamped, overshoot.
  - WARNING: if Ki is much smaller than Kp, the integral pole drifts toward
    s = 0 and settling time blows up even with no overshoot.
  - Useful starting region: Kp in [1, 8], Ki in [0.5, 5], Ki not far below Kp.

Respond ONLY with JSON of the form {"kp": <number>, "ki": <number>}. No prose."""


def initial_user_msg(specs: dict) -> str:
    return (f"Design targets for this run:\n"
            f"  - settling time <= {specs['settling_time_max']} s\n"
            f"  - overshoot     <= {specs['overshoot_max']} %\n"
            f"  - rise time     <= {specs['rise_time_max']} s\n"
            f"  - zero step steady-state error\n"
            f"Propose initial (Kp, Ki).")


def feedback_msg(i: int, gains: dict, perf: dict, specs: dict) -> str:
    kp, ki = gains["kp"], gains["ki"]
    if not perf["stable"]:
        return f"Iter {i}: Kp={kp}, Ki={ki} is UNSTABLE. Reduce Ki and/or Kp."
    issues = []
    if perf["settling_time"] > specs["settling_time_max"]:
        if perf["overshoot"] < 3.0 and ki < kp:
            issues.append(f"settling={perf['settling_time']:.2f}s with little overshoot — "
                          f"the slow mode is the PI integral pole. INCREASE Ki toward Kp.")
        else:
            issues.append(f"settling={perf['settling_time']:.2f}s > {specs['settling_time_max']}s.")
    if perf["overshoot"] > specs["overshoot_max"]:
        issues.append(f"overshoot={perf['overshoot']:.2f}% > {specs['overshoot_max']}%; "
                      f"REDUCE Kp, keep Ki/Kp similar.")
    if perf["rise_time"] > specs["rise_time_max"]:
        issues.append(f"rise={perf['rise_time']:.3f}s > {specs['rise_time_max']}s; "
                      f"INCREASE Kp (and Ki proportionally).")
    return (f"Iter {i}: Kp={kp}, Ki={ki}. {'; '.join(issues)} "
            f"Hint: keep Ki/Kp around 1-2 so the PI zero stays near the slow plant pole.")


# ---------- Agent loop ----------
MAX_ITER = 8


def run_agent(specs: dict | None = None):
    """Generator. Yields one dict per iteration; the frontend renders these live."""
    specs = specs if specs is not None else DEFAULT_SPECS
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_msg(specs)},
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
        perf = evaluate(gains, specs)
        yield {"iter": i, "gains": gains, "perf": perf, "raw": raw, "specs": specs}
        if passes(perf):
            return
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback_msg(i, gains, perf, specs)})


# ---------- Plot helper (returns an SVG string for crisp browser rendering) ----------
def step_plot_svg(T, gains: dict, specs: dict) -> str:
    import io
    import matplotlib.pyplot as plt
    t = np.linspace(0, 8, 800)
    t_out, y_out = ctrl.step_response(T, T=t)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_out, y_out, lw=2)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.axhline(1.0 + specs["overshoot_max"] / 100, ls=":", color="red", lw=0.7,
               label=f"overshoot spec ({specs['overshoot_max']:.0f} %)")
    ax.axvline(specs["settling_time_max"], ls="--", color="green", lw=0.8,
               label=f"settling spec ({specs['settling_time_max']:.1f} s)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y(t)")
    ax.set_title(f"Converged step response — Kp={gains['kp']}, Ki={gains['ki']}")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
