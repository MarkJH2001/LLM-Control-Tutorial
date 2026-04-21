# LLM + 控制教程

一份应用教程，涵盖 **LLM 基础**、**API 使用** 以及 **智能体工作流**，带有控制系统视角。

本教程使用 `openai` Python SDK，通过同一个客户端访问 **OpenAI**、**DeepSeek** 和 **Qwen** —— 不同服务商之间有差异的代码会以三个选项卡展示，因此你可以选择手头已有密钥的那一个。

## 本站涵盖的内容

- **[开始入门](getting-started/index.md)** —— conda 环境、API 密钥。
- **[LLM 基础](llm-basics/index.md)** —— LLM 是什么、词元、采样、上下文窗口。
- **[调用 API](api/index.md)** —— 首次调用、统一客户端、流式输出、工具调用。
- **[智能体工作流](agents/index.md)** —— 规划 / 执行 / 观察循环、记忆、多智能体模式。
- **[LLM + 控制](control/index.md)** *（进行中）* —— 将这些模式应用到控制系统问题。

## 运行代码

代码示例与其预期输出一并展示在正文中。若要自行运行，请按照 [环境配置](getting-started/setup.md) 创建 `llm-tutorial` 这一 conda 环境，然后在本地执行这些代码片段。

!!! tip "从这里开始"
    第一次接触这些内容？先去 [环境配置](getting-started/setup.md)，然后进入 [LLM 基础](llm-basics/index.md)。
