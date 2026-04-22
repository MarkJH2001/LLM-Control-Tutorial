# SJTU (致远一号 AI 模型 API)

**For Shanghai Jiao Tong University affiliates only.** SJTU's High-Performance Computing Center runs an on-campus LLM gateway at `models.sjtu.edu.cn` that hosts several leading Chinese / open-source models behind a single OpenAI-compatible endpoint. If you're an SJTU student or staff member, this is often the fastest path to experimenting with Chinese-ecosystem models without paying per-token — the test-phase quota is large and the API shape is the same as every other provider in this tutorial.

## Sign up and get a key

1. Log in to **My SJTU** (交我办) at <https://my.sjtu.edu.cn/> — web or mobile app.
2. Search for the workflow **"致远一号" AI 模型 API 申请（测试）** ("Zhiyuan-1" AI Model API Application — Test) and start an application.
3. Fill out the form (contact email, which models you want, intended use) and submit.
4. Wait for approval. Once granted, the `base_url` and `api-key` are emailed to you and also delivered to your Jiao-Wo-Ban inbox.

Keys are personal — selling or leaking the key cancels your access. **The test-phase key expires 2026-06-30.**

## Quota (test phase)

- **100 requests per minute**
- **100,000 tokens per minute**
- **1,000,000,000 tokens per week**

That's an enormous allowance by individual-user standards — enough to run the entire tutorial's agent loops many times over. Larger campus applications (research groups, classroom deployments) can request a higher quota by emailing `hpc@sjtu.edu.cn`.

## Install

```bash
pip install openai
```

Nothing SJTU-specific — the endpoint speaks the OpenAI wire format.

## Hello world

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SJTU_API_KEY"],
    base_url="https://models.sjtu.edu.cn/api/v1",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
)
print(resp.choices[0].message.content)
```

## Loading the key

```bash
SJTU_API_KEY=sk-...
```

## Model catalog

Use the exact `id` returned by `GET /api/v1/models`, **not** the human-readable label you saw on the application form. Current catalog:

| `model` id | Underlying model | Context | Notes |
|---|---|---|---|
| `deepseek-chat` | DeepSeek V3.2 (685B) | 32k | General-purpose. **Default for the tutorial's notebooks.** |
| `deepseek-reasoner` | DeepSeek V3.2 reasoning | 32k | Returns `reasoning_content` alongside `content`. |
| `glm-5` | Zhipu GLM-5 (744B) | 32k | Strong on coding and agent tasks. |
| `minimax` / `minimax-m2.5` | MiniMax M2.5 (230B) | **192k** | Use this when you need a long context. |
| `qwen3coder` | Qwen3-Coder (30B) | 32k | Coding-focused. |
| `qwen3vl` | Qwen3-VL (30B) | 32k | Vision / multimodal. |

You can verify the live list any time with:

```bash
curl 'https://models.sjtu.edu.cn/api/v1/models' -H 'Authorization: Bearer your-api-key'
```

## Gotchas

- **Network**: the endpoint is **only reachable from the SJTU campus network, or from off-campus via the SJTU VPN.** Colab, Deepnote, or any non-SJTU cloud instance will timeout against this URL. For those environments, fall back to one of the public providers.
- **V3.2 requires a `user` message**: if your `messages` list contains only a `system` entry with no `user` message, `deepseek-chat` and `deepseek-reasoner` return an empty response. Always include at least one `{"role": "user", ...}` turn. (The tutorial's code always does, so nothing to change — just worth knowing if you hand-roll a request.)
- **Key expiry**: the test-phase key is valid through **2026-06-30**. Renew when announced.
- **Use the `id`, not the form label**: e.g. the form may show "DeepSeek-V3.2", but the API call needs `model="deepseek-chat"`.

## Further reading

- Official SJTU docs: <https://claw.sjtu.edu.cn/guide/sjtu-api/>
- SJTU HPC home: <https://hpc.sjtu.edu.cn/>
