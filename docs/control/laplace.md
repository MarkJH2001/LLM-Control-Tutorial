# Laplace Transforms

[![Open in Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/control_laplace.ipynb)
[![Open in Deepnote](../assets/deepnote-badge.svg)](https://deepnote.com/launch?url=https://github.com/MarkJH2001/LLM-Control-Tutorial/blob/main/notebooks/control_laplace.ipynb)

The classic first exercises in a controls course: work the Laplace transform table, run partial-fraction expansions, solve ODEs in the $s$-domain. This page takes three textbook problems — **Homework 2, Problem 3.3(b)** (forward transform), **Homework 3, Problem 3.7(h)** (inverse via partial-fraction expansion), and **Homework 3, Problem 3.9(e)** (a second-order ODE solved with Laplace) — and sends each to the model with a plain prompt.

## Forward transforms — Homework 2, Problem 3.3(b)

Find the Laplace transform $F(s) = \mathcal{L}\{f(t)\}$ for:

$$
f(t) = \cos 2t + 4\sin 5t + e^{-2t}\cos 7t
$$

Textbook answer, from the standard Laplace table plus the frequency-shift property $\mathcal{L}\{e^{-at}f(t)\} = F(s+a)$:

$$
F(s) = \frac{s}{s^2+4} + \frac{20}{s^2+25} + \frac{s+2}{(s+2)^2+49}
$$

Pose the problem in the textbook's own wording — no custom system prompt, no formatting instructions, just the question as a student would ask it. The [First Call](../api/first-call.md) pattern:

```python
expr = "cos(2t) + 4 sin(5t) + e^(-2t) cos(7t)"

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Find the Laplace transform of the following time function: f(t) = {expr}"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)
```

??? example "Actual output — qwen-plus (2026-04-21)"

    We are asked to find the **Laplace transform** of

    \[
    f(t) = \cos(2t) + 4\sin(5t) + e^{-2t}\cos(7t)
    \]

    Recall standard Laplace transforms (for \( \operatorname{Re}(s) > \text{appropriate value} \)):

    1. \( \mathcal{L}\{\cos(at)\} = \dfrac{s}{s^2 + a^2}, \quad \text{Re}(s) > 0 \)

    2. \( \mathcal{L}\{\sin(at)\} = \dfrac{a}{s^2 + a^2}, \quad \text{Re}(s) > 0 \)

    3. \( \mathcal{L}\{e^{at}\cos(bt)\} = \dfrac{s - a}{(s - a)^2 + b^2}, \quad \text{Re}(s) > a \)
       (Note: For \( e^{-at}\cos(bt) \), replace \( a \to -a \), so it becomes \( \dfrac{s + a}{(s + a)^2 + b^2} \))

    Let's compute term-by-term.

    ---

    **Term 1: \( \cos(2t) \)**

    \[
    \mathcal{L}\{\cos(2t)\} = \frac{s}{s^2 + 2^2} = \frac{s}{s^2 + 4}
    \]

    ---

    **Term 2: \( 4\sin(5t) \)**

    \[
    \mathcal{L}\{4\sin(5t)\} = 4 \cdot \frac{5}{s^2 + 5^2} = \frac{20}{s^2 + 25}
    \]

    ---

    **Term 3: \( e^{-2t}\cos(7t) \)**

    Use the **s-shift property**:
    If \( \mathcal{L}\{g(t)\} = G(s) \), then \( \mathcal{L}\{e^{at}g(t)\} = G(s - a) \).
    Here, \( g(t) = \cos(7t) \), so \( G(s) = \dfrac{s}{s^2 + 49} \), and \( a = -2 \), so:

    \[
    \mathcal{L}\{e^{-2t}\cos(7t)\} = G(s - (-2)) = G(s + 2) = \frac{s + 2}{(s + 2)^2 + 49}
    \]

    ---

    **Combine all terms:**

    \[
    \boxed{
    \mathcal{L}\{f(t)\} = \frac{s}{s^2 + 4} + \frac{20}{s^2 + 25} + \frac{s + 2}{(s + 2)^2 + 49}
    }
    \]

    **Region of convergence (ROC):**
    - First term: Re(s) > 0
    - Second term: Re(s) > 0
    - Third term: Re(s) > −2 (since exponential decay rate is −2 ⇒ need Re(s) > −2 for convergence)
    So overall ROC is Re(s) > 0 (the intersection, since 0 > −2).

    ✅ Final answer:

    \[
    \boxed{
    F(s) = \frac{s}{s^2 + 4} + \frac{20}{s^2 + 25} + \frac{s + 2}{s^2 + 4s + 53}, \quad \text{Re}(s) > 0
    }
    \]

The final boxed expression matches the textbook term-for-term. Note that the model even volunteers the region-of-convergence discussion — behavior you'd reward in a homework submission but didn't have to prompt for. At `temperature=0` the model isn't deriving the transform from first principles; it's recalling it, and recalling deterministically.

## Inverse transforms — Homework 3, Problem 3.7(h)

Same textbook, reversed direction. You're given $F(s)$ and asked for $f(t)$, using partial-fraction decomposition:

$$
F(s) = \frac{3s+2}{s^2-s-2}
$$

Textbook answer — both the partial-fraction decomposition and the final $f(t)$:

| Partial-fraction form | $f(t)$ |
|---|---|
| $\dfrac{1/3}{s+1} + \dfrac{8/3}{s-2}$ | $\dfrac{1}{3}e^{-t} + \dfrac{8}{3}e^{2t}$ |

This is a different skill from Problem 3.3 — it requires actual algebra (factoring the denominator $s^2 - s - 2 = (s+1)(s-2)$, solving for residues), not just a table lookup. Notice that the textbook's problem statement already says *"using partial-fraction expansions"* — so the method requirement is baked into the prompt itself, and no custom system prompt is needed to force it. Using the textbook wording verbatim:

```python
Fs = "(3s + 2) / (s^2 - s - 2)"

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Find the time function corresponding to the following Laplace transform using partial-fraction expansion: F(s) = {Fs}"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)
```

??? example "Actual output — qwen-plus (2026-04-21)"

    We are given the Laplace transform:

    \[
    F(s) = \frac{3s + 2}{s^2 - s - 2}
    \]

    We need to find the **inverse Laplace transform**, i.e., find \( f(t) = \mathcal{L}^{-1}\{F(s)\} \), using **partial-fraction expansion**.

    ---

    **Step 1: Factor the denominator**

    \[
    s^2 - s - 2 = (s - 2)(s + 1)
    \]
    (Since roots of \(s^2 - s - 2 = 0\) are \(s = 2\) and \(s = -1\).)

    So,

    \[
    F(s) = \frac{3s + 2}{(s - 2)(s + 1)}
    \]

    ---

    **Step 2: Partial fraction decomposition**

    Assume:

    \[
    \frac{3s + 2}{(s - 2)(s + 1)} = \frac{A}{s - 2} + \frac{B}{s + 1}
    \]

    Multiply both sides by \((s - 2)(s + 1)\):

    \[
    3s + 2 = A(s + 1) + B(s - 2)
    \]

    Expand RHS:

    \[
    A(s + 1) + B(s - 2) = As + A + Bs - 2B = (A + B)s + (A - 2B)
    \]

    Match coefficients with LHS \(3s + 2\):

    - Coefficient of \(s\): \(A + B = 3\)
    - Constant term: \(A - 2B = 2\)

    Solve the system:

    From first equation: \(A = 3 - B\)

    Substitute into second:

    \[
    (3 - B) - 2B = 2 \Rightarrow 3 - 3B = 2 \Rightarrow -3B = -1 \Rightarrow B = \frac{1}{3}
    \]

    Then \(A = 3 - \frac{1}{3} = \frac{8}{3}\)

    So,

    \[
    F(s) = \frac{8/3}{s - 2} + \frac{1/3}{s + 1}
    \]

    ---

    **Step 3: Inverse Laplace transform**

    Recall:

    \[
    \mathcal{L}^{-1}\left\{\frac{1}{s - a}\right\} = e^{at}, \quad \text{for } t \ge 0
    \]

    Thus:

    \[
    f(t) = \frac{8}{3} e^{2t} + \frac{1}{3} e^{-t}, \quad t \ge 0
    \]

    ---

    **✅ Final Answer:**

    \[
    \boxed{f(t) = \frac{8}{3}e^{2t} + \frac{1}{3}e^{-t}, \quad t \ge 0}
    \]

The response includes the PF decomposition alongside the final inverse transform — the model respects the textbook's wording and shows the work. Both the decomposition and $f(t)$ match the textbook. Plain prompt, textbook wording verbatim, zero machinery.

## Solving ODEs with Laplace — Homework 3, Problem 3.9(e)

Step up: solve an ODE using Laplace transforms. The pattern is longer — Laplace-transform both sides, apply the initial conditions, solve algebraically for $Y(s)$, partial-fraction, then inverse-transform back to $y(t)$:

$$
y''(t) + 2y'(t) + 2y(t) = 5\sin t, \quad y(0) = y'(0) = 0
$$

Textbook answer:

$$
y(t) = \sin t - 2\cos t + 2e^{-t}\cos t + e^{-t}\sin t
$$

More steps than a transform lookup or a pure partial-fraction problem, but every step is still within the well-trodden textbook repertoire. Same prompt shape, textbook wording:

```python
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Solve the following ODE using Laplace transforms: y''(t) + 2 y'(t) + 2 y(t) = 5 sin(t), given y(0) = y'(0) = 0"},
    ],
    temperature=0,
)
print(resp.choices[0].message.content)
```

??? example "Actual output — qwen-plus (2026-04-21)"

    We are given the second-order linear nonhomogeneous ODE:

    \[
    y''(t) + 2y'(t) + 2y(t) = 5 \sin t, \quad y(0) = 0,\; y'(0) = 0
    \]

    We will solve it using **Laplace transforms**.

    ---

    **Step 1: Take Laplace transform of both sides**

    Recall Laplace transforms:

    - \(\mathcal{L}\{y'(t)\} = sY(s) - y(0)\)
    - \(\mathcal{L}\{y''(t)\} = s^2 Y(s) - s y(0) - y'(0)\)
    - \(\mathcal{L}\{\sin(at)\} = \frac{a}{s^2 + a^2}\)

    Given \(y(0) = y'(0) = 0\), we have:

    - \(\mathcal{L}\{y'(t)\} = sY(s)\)
    - \(\mathcal{L}\{y''(t)\} = s^2 Y(s)\)

    Also, \(\mathcal{L}\{5 \sin t\} = 5 \cdot \frac{1}{s^2 + 1} = \frac{5}{s^2 + 1}\)

    Now apply Laplace transform to the ODE:

    \[
    s^2 Y(s) + 2s Y(s) + 2 Y(s) = \frac{5}{s^2 + 1}
    \]

    Factor left-hand side:

    \[
    (s^2 + 2s + 2) Y(s) = \frac{5}{s^2 + 1}
    \]

    So,

    \[
    Y(s) = \frac{5}{(s^2 + 2s + 2)(s^2 + 1)}
    \]

    ---

    **Step 2: Partial fraction decomposition**

    Complete the square: \(s^2 + 2s + 2 = (s+1)^2 + 1\).

    Decompose:

    \[
    \frac{5}{(s^2 + 1)\big((s+1)^2 + 1\big)} = \frac{As + B}{s^2 + 1} + \frac{Cs + D}{(s+1)^2 + 1}
    \]

    Equate coefficients; solving the 4×4 system gives \(A = -2,\; B = 1,\; C = 2,\; D = 3\).

    \[
    Y(s) = \frac{-2s + 1}{s^2 + 1} + \frac{2s + 3}{(s+1)^2 + 1}
    \]

    ---

    **Step 3: Inverse Laplace transform**

    - \(\frac{-2s + 1}{s^2 + 1} \;\rightarrow\; -2\cos t + \sin t\)
    - For the second term, substitute \(u = s+1\): \(\frac{2s + 3}{(s+1)^2 + 1} = \frac{2u + 1}{u^2 + 1}\), whose inverse (with the \(e^{-t}\) shift) is \(e^{-t}(2\cos t + \sin t)\).

    Combine:

    \[
    \boxed{y(t) = -2 \cos t + \sin t + e^{-t}(2 \cos t + \sin t)}
    \]

    **Verify initial conditions:**

    - \(y(0) = -2(1) + 0 + 1 \cdot (2 + 0) = 0\) ✅
    - \(y'(0)\): after differentiating and evaluating at 0, get \(1 - 1 = 0\) ✅

    Both initial conditions satisfied.

The model chains all three stages (Laplace → partial fractions → inverse Laplace) and arrives at the textbook answer, then volunteers a verification that both initial conditions are satisfied — a detail you'd want on a homework submission but didn't prompt for. Still a plain prompt, still deterministic at `temperature=0`.

## Next

Back to the [LLM + Control overview](index.md), or forward to the upcoming Routh–Hurwitz page.
