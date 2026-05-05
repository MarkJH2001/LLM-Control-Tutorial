"""Gradio frontend. Run: python app_gradio.py"""
import gradio as gr

from backend import (DEFAULT_SPECS, MAX_ITER, SYSTEM_PROMPT,
                     passes, run_agent, step_plot_svg)


def run_with_specs(settling_max: float, overshoot_max: float, rise_max: float):
    specs = {
        "settling_time_max": float(settling_max),
        "overshoot_max":     float(overshoot_max),
        "rise_time_max":     float(rise_max),
    }
    log = []
    final_svg = ""
    for step in run_agent(specs):
        i = step["iter"]; g = step["gains"]; p = step["perf"]
        log.append(f"Iter {i}: Kp={g['kp']}, Ki={g['ki']} → "
                   f"settling={p['settling_time']!s:<8s} ovr={p['overshoot']!s:<7s} "
                   f"rise={p['rise_time']!s:<8s} pass={p['passes']}")
        if passes(p):
            log.append(f"\nConverged at iteration {i}.")
            final_svg = step_plot_svg(p["T"], g, specs)
            final_svg = f'<div style="max-width:760px;margin:auto">{final_svg}</div>'
            break
    else:
        log.append(f"\nDid not converge within {MAX_ITER} iterations.")
    return "\n".join(log), final_svg


with gr.Blocks(title="Course Project Demo") as iface:
    gr.Markdown("# Course Project Demo")
    gr.Markdown(
        r"""
**Plant**: $G(s) = \dfrac{1}{s^2 + 3s + 2}$.

**Controller**: $G_c(s) = K_p + K_i / s$.

**Goal**: design $K_p$ and $K_i$ so the unit-feedback closed-loop step response
hits the targets below. An LLM agent proposes gains, a `python-control` evaluator
scores them, and the loop iterates until every spec passes (or the iteration
cap fires). Step input zero steady-state error is automatic when the loop is
stable (PI integral action handles it).
""",
        latex_delimiters=[{"left": "$", "right": "$", "display": False}],
    )

    gr.Markdown("### Design targets")
    with gr.Row():
        settling_in = gr.Number(label="Settling time max (s)",
                                value=DEFAULT_SPECS["settling_time_max"])
        overshoot_in = gr.Number(label="Overshoot max (%)",
                                 value=DEFAULT_SPECS["overshoot_max"])
        rise_in = gr.Number(label="Rise time max (s)",
                            value=DEFAULT_SPECS["rise_time_max"])

    with gr.Accordion("Show system prompt (LLM reasoning context)", open=False):
        gr.Textbox(value=SYSTEM_PROMPT, lines=18, interactive=False, show_label=False)

    run_btn = gr.Button("Run agent", variant="primary")

    log_out = gr.Textbox(label="Iteration log", lines=12)
    plot_out = gr.HTML(label="Step response")

    run_btn.click(
        fn=run_with_specs,
        inputs=[settling_in, overshoot_in, rise_in],
        outputs=[log_out, plot_out],
    )


if __name__ == "__main__":
    iface.launch()
