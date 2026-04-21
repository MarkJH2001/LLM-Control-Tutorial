# 首次调用

向 LLM 发起最小化的一次往返：发送一条消息，打印回复。下面对我们覆盖的三家服务商各演示一次 —— 它们都使用同一个 `openai` 客户端。

## 前置条件

- 至少一家服务商的密钥。如果还没有，请前往 [申请 API 密钥](get-a-key/index.md) 页面挑一家。
- 密钥已存入你的环境。我们默认你在项目根目录有一个 `.env` 文件，里面填入了你手头已有的那几项：

    ```bash
    OPENAI_API_KEY=sk-...
    DEEPSEEK_API_KEY=sk-...
    DASHSCOPE_API_KEY=sk-...
    ```

- `llm-tutorial` 这一 conda 环境已激活：

    ```bash
    conda activate llm-tutorial
    ```

`openai` SDK 已作为项目依赖安装完毕。**DeepSeek 和 Qwen 复用 `openai` SDK** —— 只有 `base_url` 与 `model` 需要改动。

## Hello world

=== "OpenAI"

    ```python title="first_call_openai.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    预期输出：

    ```text
    Hi there! How can I help you today?
    ```

=== "DeepSeek"

    DeepSeek 提供了 OpenAI 兼容的接口，所以 `openai` 客户端可以原样使用 —— 只需要覆盖 `base_url` 并换成 DeepSeek 的模型名。

    ```python title="first_call_deepseek.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    预期输出：

    ```text
    Hello! Hope you're doing well.
    ```

    要使用推理模型，把 `model` 设为 `"deepseek-reasoner"`；返回体中会额外带上 `resp.choices[0].message.reasoning_content`。

=== "Qwen"

    阿里云 DashScope 上的 Qwen 也提供了 OpenAI 兼容的接口。

    ```python title="first_call_qwen.py"
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
    )

    print(resp.choices[0].message.content)
    ```

    预期输出：

    ```text
    Hi! Nice to meet you.
    ```

## 返回体的形态

你最常访问的字段：

```python
resp.choices[0].message.content    # the assistant's text
resp.choices[0].finish_reason      # "stop", "length", "tool_calls", ...
resp.model                         # resolved model ID
resp.usage.prompt_tokens           # input tokens billed
resp.usage.completion_tokens       # output tokens billed
resp.usage.total_tokens
```

## 加入 system 提示

在前面再塞一条 `role: "system"` 的消息：

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",  # or "deepseek-chat", "qwen-plus"
    messages=[
        {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
        {"role": "user", "content": "What is a PID controller?"},
    ],
)
```

## 故障排查

- `AuthenticationError` / `401`：密钥错误、已被吊销，或被绑定在另一个工作区 / 区域上。
- OpenAI `insufficient_quota`：密钥没有余额；到 [Billing](https://platform.openai.com/account/billing) 充值。
- `ConnectionError`：检查网络 / 代理；部分服务商的端点在某些地区被屏蔽。

## 下一步

- [统一客户端](unified-client.md) —— 把三家服务商封装到同一个函数后，切换只需一行。
- [流式输出](streaming.md) —— 让词元在生成的同时返回。
- [工具调用](tool-use.md) —— 让模型调用你写的 Python 函数。
