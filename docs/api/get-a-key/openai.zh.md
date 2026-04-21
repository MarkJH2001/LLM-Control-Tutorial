# OpenAI

## 注册并获取密钥

1. 在 [platform.openai.com](https://platform.openai.com) 注册账号。
2. 添加支付方式：**Settings → Billing → Payment methods**。
3. 充值：**Settings → Billing → Add to credit balance**（有最低充值金额；最新下限以账单页为准）。
4. 创建密钥：**[platform.openai.com/api-keys](https://platform.openai.com/api-keys) → Create new secret key**。立刻复制保存 —— 之后你将无法再次查看完整密钥。

!!! warning "必须先充值"
    OpenAI 已经取消了免费试用额度。账户没有余额时，新建的密钥每次请求都会返回 `insufficient_quota`。

## 安装

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

## 加载密钥

把密钥写进 `.env`（该文件已被 git 忽略）：

```bash
OPENAI_API_KEY=sk-...
```

运行前执行 `export $(cat .env | xargs)`，或者直接用 `python-dotenv`：

```python
from dotenv import load_dotenv
load_dotenv()
```

## 模型目录

截至 2026 年 4 月，常用的 ID：

- `gpt-4o`、`gpt-4o-mini` —— 通用，速度快，性价比好。
- `o1`、`o3-mini` —— 推理模型，适合数学、代码、困难的逻辑题。
- `gpt-4.1` —— 长上下文。

规范目录（始终最新）：[platform.openai.com/docs/models](https://platform.openai.com/docs/models)。

## 进一步阅读

- 定价：[openai.com/api/pricing](https://openai.com/api/pricing)
- 限流：你控制台里的 **Settings → Limits**（分级随消费增加）。
- Python SDK 源码：[github.com/openai/openai-python](https://github.com/openai/openai-python)
