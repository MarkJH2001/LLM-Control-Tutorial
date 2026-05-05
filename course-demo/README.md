# Course Project Demo — PI Tuning Agent

End-to-end worked example for the [Course Project — Demo](https://markjh2001.github.io/LLM-Control-Tutorial/course-project/demo/) tutorial page.

PI design on $G(s) = 1 / ((s+1)(s+2))$ via an LLM agent loop running against the SJTU API, with a Gradio or Streamlit web UI on top.

## Files

- `backend.py` — SJTU client, plant, `python-control` evaluator, agent loop
- `app_streamlit.py` — Streamlit frontend (default)
- `app_gradio.py` — Gradio frontend (alternative)
- `requirements.txt` — Python dependencies
- `.env.example` — copy to `.env` and paste your `SJTU_API_KEY`

## Setup

```bash
cd course-demo
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp .env.example .env               # then edit .env to add your key
```

SJTU API access requires the SJTU campus network or VPN.

## Run it

Pick one of the two frontends.

```bash
# Streamlit (default)
streamlit run app_streamlit.py
# opens http://localhost:8501

# Gradio (alternative)
python app_gradio.py
# opens http://127.0.0.1:7860
```

Click **Run agent** (Gradio: **Submit**) to start the loop. The agent proposes (Kp, Ki), the evaluator scores them against the spec, and feedback is sent back until convergence (typically 1–4 iterations) or the iteration cap.
