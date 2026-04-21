# 环境配置

本教程的所有内容都在一个名为 `llm-tutorial` 的 conda 环境里运行。创建一次即可，后续每一页都假设它已经激活。

## 1. 创建 conda 环境

```bash
conda create -n llm-tutorial python=3.12 -y
conda activate llm-tutorial
```

## 2. 安装项目依赖

在仓库根目录下执行：

```bash
pip install -e .
```

这会安装 [`pyproject.toml`](https://github.com/MarkJH2001/LLM-Control-Tutorial/blob/main/pyproject.toml) 中列出的所有依赖 —— `mkdocs-material`、`openai` SDK、用于控制系统示例的 `numpy` / `scipy` / `matplotlib` / `control`，以及开发辅助工具。

## 3. 验证安装

```bash
python -c "import openai; print('openai', openai.__version__)"
python -c "import tiktoken; print('tiktoken', tiktoken.__version__)"
python -c "import control; print('control', control.__version__)"
```

三条命令都应该打印版本号，而不是抛出 `ImportError`。

## 4. 添加 API 密钥

要运行任何涉及 LLM 调用的代码，你至少需要一家服务商的密钥。复制模板文件：

```bash
cp .env.example .env
```

——然后编辑 `.env`，在 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY` 中填入你持有的任意一项。`.env` 已被 git 忽略。

如何申请密钥 —— 参见 [API 密钥](api-keys.md)。

## 5. 本地启动站点（可选）

如果你想离线阅读本教程或者修改页面：

```bash
mkdocs serve
```

然后打开 <http://127.0.0.1:8000>。任何 markdown 文件保存后浏览器会自动刷新。

## 故障排查

- **`conda: command not found`** —— 先安装 [Miniforge](https://github.com/conda-forge/miniforge)（或 Anaconda / Miniconda）。
- **`pip install -e .` 在某个依赖上失败** —— 确认环境已激活（`conda activate llm-tutorial`），常见原因是仍停留在 `(base)` 且 Python 版本不对。
- **`mkdocs serve` 不响应文件变更** —— 参见项目 README 中关于文件监听器与 `~/Desktop` / iCloud 同步目录的说明，如有影响请把项目移出 iCloud 同步目录。
