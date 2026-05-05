"""Streamlit frontend. Run: streamlit run app_streamlit.py"""
import streamlit as st

from backend import (DEFAULT_SPECS, MAX_ITER, SYSTEM_PROMPT,
                     passes, run_agent, step_plot_svg)

st.set_page_config(page_title="Course Project Demo", layout="wide")
st.title("Course Project Demo")

# ---- Problem description ----
st.markdown(r"""
**Plant**: $G(s) = \dfrac{1}{s^2 + 3s + 2}$.

**Controller**: $G_c(s) = K_p + K_i / s$.

**Goal**: design $K_p$ and $K_i$ so the unit-feedback closed-loop step response
hits the targets below. An LLM agent proposes gains, a `python-control`
evaluator scores them, and the loop iterates until every spec passes (or the
iteration cap fires). Step input is assumed to have zero steady-state error
when the loop is stable (PI handles this automatically).
""")

# ---- Editable spec block ----
st.subheader("Design targets")
col1, col2, col3 = st.columns(3)
with col1:
    settling_max = st.number_input("Settling time max (s)",
                                   value=DEFAULT_SPECS["settling_time_max"],
                                   min_value=0.1, step=0.5, format="%.2f")
with col2:
    overshoot_max = st.number_input("Overshoot max (%)",
                                    value=DEFAULT_SPECS["overshoot_max"],
                                    min_value=0.0, step=1.0, format="%.2f")
with col3:
    rise_max = st.number_input("Rise time max (s)",
                               value=DEFAULT_SPECS["rise_time_max"],
                               min_value=0.05, step=0.1, format="%.2f")

specs = {
    "settling_time_max": float(settling_max),
    "overshoot_max":     float(overshoot_max),
    "rise_time_max":     float(rise_max),
}

# ---- Optional: show the system prompt the agent sees ----
with st.expander("Show system prompt (LLM reasoning context)"):
    st.code(SYSTEM_PROMPT, language="text")

# ---- Run button + live log + final plot ----
if st.button("Run agent", type="primary"):
    log_placeholder = st.empty()
    log_lines = []
    converged_step = None
    for step in run_agent(specs):
        i = step["iter"]; g = step["gains"]; p = step["perf"]
        log_lines.append(f"Iter {i}: Kp={g['kp']}, Ki={g['ki']} → "
                         f"settling={p['settling_time']!s:<8s} ovr={p['overshoot']!s:<7s} "
                         f"rise={p['rise_time']!s:<8s} pass={p['passes']}")
        log_placeholder.code("\n".join(log_lines))
        if passes(p):
            converged_step = step
            st.success(f"Converged at iteration {i}.")
            break
    if converged_step is None:
        st.error(f"Did not converge within {MAX_ITER} iterations.")
    else:
        svg = step_plot_svg(converged_step["perf"]["T"],
                            converged_step["gains"],
                            specs)
        st.markdown(
            f'<div style="max-width:760px;margin:auto">{svg}</div>',
            unsafe_allow_html=True,
        )
