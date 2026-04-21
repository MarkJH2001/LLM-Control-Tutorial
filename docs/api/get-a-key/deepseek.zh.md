# DeepSeek

DeepSeek 提供 **OpenAI 兼容** 的 API —— 你可以复用 `openai` Python 客户端，只需换 `base_url` 和模型名。

## 注册并获取密钥

1. 在 [platform.deepseek.com](https://platform.deepseek.com) 注册账号。
2. 充值：**Billing → Top up**（最低额度远低于美国服务商）。
3. 创建密钥：**[platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) → Create new API key**。立刻复制保存。

## 安装

不需要 DeepSeek 专属 SDK —— 用 OpenAI 客户端即可：

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

与 OpenAI 示例相比，不同之处只有 `base_url` 和 `model` ID。

## 加载密钥

```bash
DEEPSEEK_API_KEY=sk-...
```

## 模型目录

截至 2026 年 4 月，常用的 ID：

- `deepseek-chat` —— 通用聊天模型（指向当前的 DeepSeek-V 系列）。
- `deepseek-reasoner` —— 推理模型（DeepSeek-R 系列），会在回答之外额外带出 `reasoning_content`。

规范目录：[api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing)。

## 进一步阅读

- 文档：[api-docs.deepseek.com](https://api-docs.deepseek.com)
- 定价：[api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing) —— 明显比 OpenAI / Anthropic 便宜，还有非高峰时段折扣。
