# API Keys

You will need a key from at least one provider to run the tutorials that call an LLM.

## Pick a provider

Each provider has its own console, sign-up flow, and model catalog. Detailed walkthroughs live in the **Calling the API** section:

- [Compare providers](../api/get-a-key/index.md) — quick matrix to help you choose.
- [OpenAI](../api/get-a-key/openai.md)
- [DeepSeek](../api/get-a-key/deepseek.md)
- [Qwen](../api/get-a-key/qwen.md)

You only need one to start — the tutorials run against whichever provider(s) you have keys for.

## Storing keys locally

Regardless of provider, keep keys out of your source tree. Copy `.env.example` to `.env` and fill in whichever ones apply:

```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
```

The `.env` file is git-ignored. Load it in Python with `os.environ` directly, or with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()

import os
key = os.environ["OPENAI_API_KEY"]
```

!!! warning "Do not commit keys"
    Never paste a key into a notebook cell that will be committed. Read from env vars. If a key has ever been pasted into a shared document, notebook cell that was pushed, or screenshot, rotate it immediately from the provider's console.
