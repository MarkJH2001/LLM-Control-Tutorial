# Get an API Key

You need at least one provider's API key to run the tutorials in this section. All three providers we cover speak the **OpenAI wire format**, so one `openai` Python client works for all of them — you just swap `base_url` and `model`.

## Provider at a glance

| Provider | SDK to install                   | Best for                          | Notes                                      |
|----------|----------------------------------|-----------------------------------|--------------------------------------------|
| OpenAI   | `openai`                         | Broadest ecosystem, most examples | Requires credit top-up before use          |
| DeepSeek | `openai` (compat endpoint)       | Low cost, strong reasoning        | Reuses the `openai` client, swap `base_url`|
| Qwen     | `openai` (compat) or `dashscope` | Free tier available, open models  | Served via DashScope (Aliyun account)      |

!!! tip "Pick one to start"
    If you are unsure, start with **OpenAI** (most examples online) or **DeepSeek** (cheapest). You can add more providers later — the [unified client](../unified-client.md) pattern uses one codebase across all three.

## Safe key handling

Regardless of provider, the rules are the same:

- Store keys in environment variables or a git-ignored `.env` file — never hard-coded.
- Rotate a key immediately if it has ever been pasted into a shared document, notebook cell that was committed, or screenshot.
- Use separate keys per project where possible so revocation is cheap.

See [API Keys](../../getting-started/api-keys.md) in Getting Started for the `.env` pattern used across this site.

## Per-provider guides

- [OpenAI](openai.md)
- [DeepSeek](deepseek.md)
- [Qwen](qwen.md)
