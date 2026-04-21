# API 密钥

要运行任何调用 LLM 的教程代码，你至少需要一家服务商的密钥。

## 选择一家服务商

每家服务商都有各自的控制台、注册流程和模型目录。详细的分步指南在 **调用 API** 章节里：

- [服务商对比](../api/get-a-key/index.md) —— 一个速查矩阵，帮你决定选哪家。
- [OpenAI](../api/get-a-key/openai.md)
- [DeepSeek](../api/get-a-key/deepseek.md)
- [Qwen](../api/get-a-key/qwen.md)

入门只需要一家 —— 本教程的代码会根据你手头有哪家的密钥而运行。

## 在本地保存密钥

无论选哪家，都要把密钥放在源码树之外。把 `.env.example` 复制为 `.env`，并填入你实际拥有的那几项：

```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
```

`.env` 文件已被 git 忽略。在 Python 中可以直接用 `os.environ` 读取，或使用 `python-dotenv`：

```python
from dotenv import load_dotenv
load_dotenv()

import os
key = os.environ["OPENAI_API_KEY"]
```

!!! warning "不要提交密钥"
    永远不要把密钥粘到一个会被提交的 notebook 单元格里。要通过环境变量读取。如果某把密钥曾经出现在任何共享文档、已推送到远端的 notebook 单元格或截图里，请立即到服务商控制台轮换这把密钥。
