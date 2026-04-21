# 流式输出

一次非流式的 API 调用会一直阻塞到模型生成完毕，然后把整段回复一次性返回。**流式** 则让词元在生成的同时返回，UI 就能一边接收一边展示 —— 就是你在 ChatGPT 里看到的那种效果。

只需多加一个参数就能启用流式。其余逻辑完全一致。

## 用 OpenAI SDK 做流式

把 `stream=True` 传进去，再对返回的对象做迭代。每次迭代会拿到一小段 chunk；新出现的文字在 `chunk.choices[0].delta.content` 上。

### 选择你的服务商

我们覆盖的三家服务商都使用 `openai` SDK；只有客户端和模型名不同。

=== "OpenAI"

    ```python
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-4o-mini"
    ```

=== "DeepSeek"

    ```python
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    model = "deepseek-chat"
    ```

=== "Qwen"

    ```python
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = "qwen-plus"
    ```

### 共用的流式代码

```python title="stream_chat.py"
import os
from dotenv import load_dotenv

load_dotenv()
# client and model come from one of the tabs above

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain a PID controller in three sentences."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

两处经常绊倒人的地方：

- **`flush=True`** 必须设，否则终端会缓冲输出。
- **首个 chunk 的 `delta.content` 通常是 `None`**（它携带的是 role 而不是正文）。用 `if delta:` 护一下总是对的。

### 在流式输出中读取 token 用量

默认情况下，OpenAI **不会** 在流式响应里附带 token 计数。通过 `stream_options` 手动开启：

```python
stream = client.chat.completions.create(
    model=model,
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},
)
```

最后一个 chunk 的 `chunk.usage` 会带上 `prompt_tokens`、`completion_tokens`、`total_tokens`。

## 把流式接入统一客户端

在 [unified-client.md](unified-client.md) 里的 `chat()` 基础上，暴露一个与服务商无关的生成器：

```python title="unified_stream.py"
from typing import Iterator
import os
from openai import OpenAI

# PROVIDERS dict from unified-client.md is reused as-is

def stream_chat(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    system: str | None = None,
) -> Iterator[str]:
    cfg = PROVIDERS[provider]
    client = OpenAI(api_key=os.environ[cfg.env_var], base_url=cfg.base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=model or cfg.default_model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

对任何一家服务商都能一样地用：

```python
for token in stream_chat("Say hi.", provider="deepseek"):
    print(token, end="", flush=True)
print()
```

## 那些容易踩坑的地方

- **网络缓冲。** 某些企业代理或负载均衡器会把响应攒起来一起发，从而把流式打回一次性返回 —— 你会一次拿到完整回复。如果怀疑是这个原因，把测试放到代理之外再试。
- **流式中的工具调用** 使用不同的 chunk 结构（每个工具块会有 start / delta / stop 事件）。这里不展开；在 [工具调用](tool-use.md) 里涉及。
- **不要把 `print` 和 UI 混在一起用。** 真实应用里，应该把每个 chunk 通过 websocket / SSE / 队列推到前端，而不是直接写标准输出。
- **流式中途的错误。** 网络断开、限流、内容过滤触发都可能中断一次流。在循环外套一层 `try/except`，再决定是重试还是把已有的一部分回复暴露给用户。

## 下一步

- [工具调用](tool-use.md) —— 让模型调用你的 Python 函数。
