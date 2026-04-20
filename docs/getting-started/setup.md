# Setup

## Conda environment

```bash
conda create -n llm-tutorial python=3.12 -y
conda activate llm-tutorial
pip install -e .
```

## Verify

```bash
python -c "import anthropic, openai; print(anthropic.__version__, openai.__version__)"
```

## Serving the site locally

```bash
mkdocs serve
```
