# Qwen（通义千问）

Qwen 通过阿里云的 **DashScope** / **Model Studio** 平台提供。它既支持原生的 `dashscope` SDK，**也** 支持 OpenAI 兼容端点。我们选择 OpenAI 兼容端点，这样同一份 `openai` 客户端代码就能覆盖 OpenAI、DeepSeek 和 Qwen。

## 注册并获取密钥

1. 用阿里云账号登录 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) 注册。
2. 完成实名认证。
3. 创建密钥：**API-KEY management → Create new API-KEY**。立刻复制保存。

新账号会获赠一份免费 token 额度 —— 做学习和原型足够用。

## 安装

```bash
pip install openai
```

（如果你想使用原生 SDK 以获得阿里云专属功能，改用 `pip install dashscope`）。

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

## 加载密钥

```bash
DASHSCOPE_API_KEY=sk-...
```

## 模型目录

截至 2026 年 4 月，常用的 ID：

- `qwen-max` —— 旗舰模型，能力最强。
- `qwen-plus` —— 平衡款。
- `qwen-turbo` —— 最快、最便宜。
- `qwen3-*` —— 最新的开源权重系列（Hugging Face 上也可下载本地推理）。

规范目录：[help.aliyun.com/zh/dashscope](https://help.aliyun.com/zh/dashscope/)。

## 进一步阅读

- 原生 SDK（可选）：[github.com/dashscope/dashscope-sdk-python](https://github.com/dashscope/dashscope-sdk-python)
- Hugging Face 上的 Qwen 开源权重：[huggingface.co/Qwen](https://huggingface.co/Qwen) —— 如果你希望本地推理而不是走 API，可以用这里的权重。
