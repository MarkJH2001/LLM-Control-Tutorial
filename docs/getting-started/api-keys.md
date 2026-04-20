# API Keys

You will need keys for the providers whose APIs you want to call.

- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- OpenAI: [platform.openai.com](https://platform.openai.com)

## Storing keys locally

Copy `.env.example` to `.env` and fill in:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The `.env` file is git-ignored. Load it in Python with `os.environ` or `python-dotenv`.

!!! warning "Do not commit keys"
    Never paste a key into a notebook cell that will be committed. Read from env vars.
