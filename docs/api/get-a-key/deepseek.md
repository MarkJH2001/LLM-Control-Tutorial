# DeepSeek

DeepSeek exposes an **OpenAI-compatible** API — you can reuse the `openai` Python client and just swap `base_url` and model name.

## Sign up and get a key

1. Create an account at [platform.deepseek.com](https://platform.deepseek.com).
2. Top up credits: **Billing → Top up** (very low minimum compared to US providers).
3. Create a key: **[platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) → Create new API key**. Copy it immediately.

## Install

You don't need a DeepSeek-specific SDK — just the OpenAI client:

```bash
pip install openai
```

## Hello world

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
)
print(resp.choices[0].message.content)
```

The only differences from the OpenAI example are `base_url` and the `model` ID.

## Loading the key

```bash
DEEPSEEK_API_KEY=sk-...
```

## Model catalog

Popular IDs as of April 2026:

- `deepseek-chat` — general-purpose chat model (points at the current DeepSeek-V family).
- `deepseek-reasoner` — reasoning model (DeepSeek-R family), emits `reasoning_content` alongside the answer.

Canonical catalog: [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing).

## Further reading

- Docs: [api-docs.deepseek.com](https://api-docs.deepseek.com)
- Pricing: [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing) — notably cheaper than OpenAI / Anthropic, with off-peak discounts.
