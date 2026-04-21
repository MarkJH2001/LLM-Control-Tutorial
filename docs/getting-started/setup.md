# Setup

Everything in this tutorial runs in a single conda environment named `llm-tutorial`. Create it once and every subsequent page assumes it's active.

## 1. Create the conda environment

```bash
conda create -n llm-tutorial python=3.12 -y
conda activate llm-tutorial
```

## 2. Install project dependencies

From the repository root:

```bash
pip install -e .
```

This installs everything listed in [`pyproject.toml`](https://github.com/MarkJH2001/LLM-Control-Tutorial/blob/main/pyproject.toml) — `mkdocs-material`, the `openai` SDK, `numpy` / `scipy` / `matplotlib` / `control` for the control-systems examples, plus dev helpers.

## 3. Verify

```bash
python -c "import openai; print('openai', openai.__version__)"
python -c "import tiktoken; print('tiktoken', tiktoken.__version__)"
python -c "import control; print('control', control.__version__)"
```

All three should print a version number, not raise `ImportError`.

## 4. Add an API key

You need at least one provider key to run any code that hits an LLM. Copy the template:

```bash
cp .env.example .env
```

…then edit `.env` and fill in whichever of `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` you have. `.env` is git-ignored.

Where to get a key → [API Keys](api-keys.md).

## 5. Serve the site locally (optional)

If you want to read the tutorial pages offline or edit them:

```bash
mkdocs serve
```

Open <http://127.0.0.1:8000>. Saved edits to any markdown file reload the browser automatically.

## Troubleshooting

- **`conda: command not found`** — install [Miniforge](https://github.com/conda-forge/miniforge) (or Anaconda / Miniconda) first.
- **`pip install -e .` fails on a dependency** — make sure the env is activated (`conda activate llm-tutorial`); a wrong Python version in `(base)` is the usual culprit.
- **`mkdocs serve` doesn't pick up changes** — see the note in the project README about file watchers and `~/Desktop` / iCloud-synced folders; move the project out of iCloud if affected.
