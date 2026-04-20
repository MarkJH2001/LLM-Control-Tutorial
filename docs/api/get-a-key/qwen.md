# Qwen (Alibaba)

Qwen is served via Alibaba's **DashScope** / **Model Studio** platform. It supports both a native `dashscope` SDK **and** an OpenAI-compatible endpoint. We'll use the OpenAI-compatible endpoint so the same `openai` client covers OpenAI, DeepSeek, and Qwen.

## Sign up and get a key

1. Sign up at [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) with an Aliyun account.
2. Complete real-name verification.
3. Create a key: **API-KEY management → Create new API-KEY**. Copy it.

New accounts get a free token allowance — enough for learning and prototyping.

## Install

```bash
pip install openai
```

(Use `pip install dashscope` instead if you want the native SDK with Alibaba-specific features.)

## Hello world

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
)
print(resp.choices[0].message.content)
```

## Loading the key

```bash
DASHSCOPE_API_KEY=sk-...
```

## Model catalog

Popular IDs as of April 2026:

- `qwen-max` — flagship, highest capability.
- `qwen-plus` — balanced.
- `qwen-turbo` — fastest, cheapest.
- `qwen3-*` — latest open-weight generation (also available on Hugging Face for local inference).

Canonical catalog: [help.aliyun.com/zh/dashscope](https://help.aliyun.com/zh/dashscope/).

## Further reading

- Native SDK (optional): [github.com/dashscope/dashscope-sdk-python](https://github.com/dashscope/dashscope-sdk-python)
- Open-weight Qwen models on Hugging Face: [huggingface.co/Qwen](https://huggingface.co/Qwen) — useful if you want to run locally rather than via the API.
