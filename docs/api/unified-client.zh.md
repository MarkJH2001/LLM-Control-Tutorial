# 统一客户端

OpenAI、DeepSeek 和 Qwen 全都遵循 **OpenAI 的接口协议**，也就是说一份 `openai` Python 客户端代码可以同时覆盖这三家 —— 你只需要换 `base_url` 和 `model`。本页把这个观察封装成一个小辅助函数，这样你的应用代码就不必关心底下用的是哪一家。

## 三家服务商

| 服务商 | `base_url` | 环境变量 | 推荐的默认模型 |
|---|---|---|---|
| OpenAI   | *（默认）* `https://api.openai.com/v1` | `OPENAI_API_KEY`   | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com`              | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen     | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen-plus` |

## 一个最小化的统一函数

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

用法：

```python
print(chat("Say hi in one sentence."))                      # OpenAI
print(chat("Say hi in one sentence.", "deepseek"))          # DeepSeek
print(chat("Say hi in one sentence.", "qwen", "qwen-max"))  # Qwen
```

三次调用中 *唯一* 不同的是 provider 参数 —— 实际的请求代码是复用的。

## 需要留意的坑

- **功能并不是完全对齐的。** 工具调用、JSON 模式、视觉输入在 DeepSeek / Qwen 上有时表现不同或直接缺失。把 "OpenAI 兼容" 理解成 "普通 chat 接口能用" —— 其它任何功能都要按服务商逐个验证。
- **限流是按服务商计的。** 用了统一客户端不代表你有一份统一的额度。如果某一家限流，重试逻辑需要知道失败的是哪一家。
- **模型 ID 会漂移。** 服务商的模型目录更新速度比本教程快。需要某个特定模型时，显式传 `model`；不需要时再退回 `cfg.default_model`。

## 下一步

- [流式输出](streaming.md) —— 让词元在生成的同时返回（统一客户端的模式可以自然延伸过去）。
- [工具调用](tool-use.md) —— 让模型调用你的 Python 函数。
