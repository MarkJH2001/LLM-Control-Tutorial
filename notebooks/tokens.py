# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tokens — runnable notebook
#
# Companion to [**LLM Basics → Tokens**](https://markjh2001.github.io/LLM-Control-Tutorial/llm-basics/tokens/).
#
# No API key needed — this one is pure client-side tokenization with `tiktoken`.

# %% [markdown]
# ## 1. Install dependencies

# %%
# %pip install --quiet tiktoken

# %% [markdown]
# ## 2. Inspect a sentence

# %%
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

text = "The quick brown fox jumps over the lazy dog."
tokens = enc.encode(text)

print(f"{len(tokens)} tokens")
print([enc.decode([t]) for t in tokens])

# %% [markdown]
# ## 3. Gotchas: same word, different tokenization
#
# The same root word tokenizes differently depending on capitalization, leading
# space, and suffix.

# %%
for s in ["color", "colorful", " color", "Color"]:
    ids = enc.encode(s)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{s!r:12} -> {len(ids)} tokens: {pieces}")

# %% [markdown]
# ## 4. Real BPE splits for less-common words

# %%
for word in ["controllability", "colorful", "tokenization", "aerospace"]:
    ids = enc.encode(word)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{word:18} -> {len(ids)} tokens: {pieces}")

# %% [markdown]
# ## 5. Numbers chunk in groups of three

# %%
for n in ["42", "123", "1234", "12345", "1000000", "12345678", "3.14159"]:
    ids = enc.encode(n)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{n:10} -> {len(ids)} tokens: {pieces}")

# %% [markdown]
# ## 6. Cost estimate for a workload

# %%
def cost_estimate(text: str, price_per_1m_in: float, price_per_1m_out: float, expected_out_tokens: int = 200) -> None:
    in_tokens = len(enc.encode(text))
    in_cost = in_tokens * price_per_1m_in / 1_000_000
    out_cost = expected_out_tokens * price_per_1m_out / 1_000_000
    print(f"input  {in_tokens:>5} tokens  ~ ${in_cost:.6f}")
    print(f"output {expected_out_tokens:>5} tokens  ~ ${out_cost:.6f}")
    print(f"total                    ~ ${in_cost + out_cost:.6f}")

# Example: a short prompt at hypothetical $0.15 / $0.60 per 1M input/output tokens.
cost_estimate("Summarize the Laplace transform in two sentences.", 0.15, 0.60)

# %% [markdown]
# ## Next
#
# - [Sampling](https://markjh2001.github.io/LLM-Control-Tutorial/llm-basics/sampling/) — turning the next-token distribution into one token.
# - [Context Window](https://markjh2001.github.io/LLM-Control-Tutorial/llm-basics/context-window/) — the token budget that caps every call.
