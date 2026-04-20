# Unified Client

OpenAI, DeepSeek, and Qwen all speak the **OpenAI wire format**, which means one `openai` Python client covers all three — you only swap `base_url` and `model`. This page wraps that insight into a small helper so your application code doesn't have to care which provider is underneath.

## The three providers

| Provider | `base_url` | Env var | A good default model |
|---|---|---|---|
| OpenAI   | *(default)* `https://api.openai.com/v1` | `OPENAI_API_KEY`   | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com`              | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen     | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen-plus` |

## A minimal unified function

```python title="unified_chat.py"
import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str | None   # None → use the SDK's default (OpenAI)
    env_var: str
    default_model: str


PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url=None,
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "qwen": ProviderConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_var="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
    ),
}


def chat(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    system: str | None = None,
) -> str:
    cfg = PROVIDERS[provider]
    client = OpenAI(
        api_key=os.environ[cfg.env_var],
        base_url=cfg.base_url,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model or cfg.default_model,
        messages=messages,
    )
    return resp.choices[0].message.content
```

Usage:

```python
print(chat("Say hi in one sentence."))                      # OpenAI
print(chat("Say hi in one sentence.", "deepseek"))          # DeepSeek
print(chat("Say hi in one sentence.", "qwen", "qwen-max"))  # Qwen
```

The *only* line that differs between calls is the provider argument — the actual request code is reused.

## Pitfalls to watch for

- **Feature parity is not guaranteed.** Tool use, JSON mode, and vision input sometimes differ or are missing on DeepSeek / Qwen. Treat "OpenAI-compatible" as "the plain chat endpoint works" — verify anything beyond that per-provider.
- **Rate limits are per provider.** A unified client does not give you a unified quota. If one provider throttles you, retry logic needs to know which one failed.
- **Model IDs drift.** Provider catalogs change faster than this tutorial. Pass an explicit `model` when you need a specific one; fall back to `cfg.default_model` for convenience.

## Next

- [Streaming](streaming.md) — return tokens as they're generated (the unified pattern extends naturally).
- [Tool Use](tool-use.md) — let the model call your Python functions.
