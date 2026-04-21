# LLM + Control

Apply the LLM patterns from earlier sections — plain prompting, tool use, and multi-agent loops — to concrete control-theory problems. The pages in this section walk through representative tasks in rough order of difficulty. Early problems (Laplace transforms) a capable model handles fine with a plain prompt; later problems (Routh–Hurwitz, Nyquist, PID tuning) force you up the staircase.

The goal isn't to replace SymPy or `python-control` with a language model — they're better at the math than any model is. The goal is to see, problem by problem, *where each layer of LLM machinery starts to become necessary*, and to build an intuition for when to reach for which tool.

## The staircase

Every page follows the same recipe: state the problem, pick a deterministic oracle for ground truth, run all three approaches, and compare.

```mermaid
flowchart LR
    P[Control problem] --> A1[Plain prompt]
    P --> A2[With deterministic tool]
    P --> A3[Writer and checker loop]
    A1 --> G[Ground-truth oracle]
    A2 --> G
    A3 --> G
    G --> R[Accuracy vs. cost]
```

Each rung reuses a pattern from earlier: plain prompting from [Calling the API](../api/index.md), deterministic tools from [Tool Use](../api/tool-use.md), and writer/checker from [Multi-agent](../agents/multi-agent.md). The staircase is the unifying story — every additional rung costs more tokens and more latency, but buys you something specific.

## Pages

- **[Laplace Transforms](laplace.md)** — Homework 2 Problem 3.3 and Homework 3 Problems 3.7 / 3.9. A plain prompt already matches the textbook for forward transforms, inverse via partial fractions, and ODE solving — no tool needed.
- **[Control Plots](plots.md)** — give the model three `python-control`-backed tools (root locus, Bode, Nyquist) and let it parse a natural-language transfer function into coefficient lists and call the matching tool. First page where a tool is genuinely required (plain prompts cannot emit plots).
- **[PID Tuning](pid.md)** — an agent with a deterministic step-response evaluator in the loop. Compares a single chat-style call (open-loop) against the full iterative loop (closed-loop) on a first-order PI tuning problem. The analogy to open-loop vs. closed-loop *control* is the pedagogical spine of the page.
- **[Real-World Applications](applications.md)** — three deployed projects (drone swarm choreography, autonomous spacecraft in KSPDG, long-horizon robotic manipulation) that put an LLM in the control stack, each with a different role: planner, fine-tuned controller, reflective agent. Videos and architecture diagrams, no code.

Planned:

- **Routh–Hurwitz stability** — plain-prompt accuracy starts to slip on higher-order characteristic polynomials.
