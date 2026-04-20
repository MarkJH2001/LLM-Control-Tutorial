# OpenAI

## Sign up and get a key

1. Create an account at [platform.openai.com](https://platform.openai.com).
2. Add a payment method: **Settings → Billing → Payment methods**.
3. Buy credits: **Settings → Billing → Add to credit balance** (minimum top-up applies; see the billing page for the current floor).
4. Create a key: **[platform.openai.com/api-keys](https://platform.openai.com/api-keys) → Create new secret key**. Copy it immediately — you cannot view it again.

!!! warning "Billing is required"
    OpenAI's free trial credits have been retired. Keys created without a funded balance will return `insufficient_quota` on every request.

## Install

```bash
pip install openai
```

## Hello world

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
)
print(resp.choices[0].message.content)
```

## Loading the key

Put the key in `.env` (git-ignored):

```bash
OPENAI_API_KEY=sk-...
```

Then either `export $(cat .env | xargs)` before running, or use `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Model catalog

Popular IDs as of April 2026:

- `gpt-4o`, `gpt-4o-mini` — general-purpose, fast, good price/performance.
- `o1`, `o3-mini` — reasoning models for math, code, hard logic.
- `gpt-4.1` — long context.

Canonical catalog (always current): [platform.openai.com/docs/models](https://platform.openai.com/docs/models).

## Further reading

- Pricing: [openai.com/api/pricing](https://openai.com/api/pricing)
- Rate limits: **Settings → Limits** in your dashboard (tiers scale with spend).
- Python SDK source: [github.com/openai/openai-python](https://github.com/openai/openai-python)
