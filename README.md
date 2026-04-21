# LLM + Control Tutorial

An applied tutorial covering **LLM fundamentals**, **API usage**, and **agentic workflows**, with a control-systems angle.

Built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

📖 **Live site:** <https://markjh2001.github.io/LLM-Control-Tutorial/>

## What's covered

- **Getting Started** — conda environment, API keys.
- **LLM Basics** — what an LLM is, tokens, sampling, context windows.
- **Calling the API** — OpenAI, DeepSeek, and Qwen via one SDK; first call, unified client, streaming, tool use.
- **Agentic Workflows** — plan/act/observe loops, memory, multi-agent patterns.
- **LLM + Control** (in progress) — applying these patterns to control-systems problems.

## Providers

The tutorial uses the `openai` Python SDK as the common wire format for **OpenAI**, **DeepSeek**, and **Qwen / DashScope**. Code that differs per provider is shown in three tabs, so you can pick whichever you have a key for.

## Local development

```bash
conda activate llm-tutorial
pip install -e .
mkdocs serve
```

Open <http://127.0.0.1:8000>.

First-time setup (create the conda env):

```bash
conda create -n llm-tutorial python=3.12 -y
conda activate llm-tutorial
pip install -e .
```

## Building

```bash
mkdocs build --strict
```

Static site is emitted to `site/`. Pushes to `main` trigger the [GitHub Actions workflow](.github/workflows/deploy.yml) which builds and publishes to GitHub Pages.

## API keys

Copy `.env.example` to `.env` and fill in whichever provider key(s) you have. The `.env` file is git-ignored. See [Getting Started → API Keys](docs/getting-started/api-keys.md) for details.

## License

Content and code in this repository are released under the terms set by the repository owner; check with the owner before reusing.
