# 🚀 开始使用

欢迎来到 **LazyLLM**！🎉

LazyLLM 是构建和优化多 Agent 应用的一站式开发工具，为应用开发过程中的全部环节（包括应用搭建、数据准备、模型部署、模型微调、评测等）提供了大量的工具，协助开发者用极低的成本构建 AI 应用，并可以持续的迭代优化效果。

## 🛠️ 环境准备

LazyLLM 使用 Python 开发，因此请确保您的计算机上已经安装了 3.10 以上的 Python，并安装了 pip 作为 Python 的包管理工具。

如果您是从 GitHub 上下载的 LazyLLM，您需要先初始化一下 LazyLLM 的运行环境，确保它的依赖都被正确的安装。我们在 `requirements.txt` 和 `requirements.full.txt` 中提供了运行 LazyLLM 的基础依赖包的集合。

### 基础依赖包安装

如果您所使用的计算机没有 GPU，仅希望基于在线的模型服务和应用 API 来搭建自己的 AI 应用，则您仅需安装 `requirements.txt` 中的基础包即可。您可以进入 LazyLLM 的目录，使用命令 `pip install -r requirements.txt` 安装这些依赖包。

> **注意**：
> 如果您遇到权限问题，您可能需要在命令前添加 `sudo`，或者在命令后添加 `--user`，以确保 pip 有足够的权限安装这些包。

## 🚀 部署基本功能

基础包安装完成之后，您就可以使用 LazyLLM 的基本功能搭建服务。下列 Python 脚本可以部署一个具有简单 web 界面的服务:

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
t = lazyllm.OnlineChatModule(source="openai", stream=True)
w = lazyllm.WebModule(t)
w.start().wait()
```

这个 Python 脚本将调用 OpenAI 的模型服务，并启动一个带多轮对话界面的 web 服务运行在本地的 20570 端口上。服务启动之后，使用浏览器访问 [http://localhost:20570](http://localhost:20570)，通过页面上的聊天机器人组件调用后台的大模型服务，LazyLLM 会将模型的返回结果打印在聊天机器人组件中。

> **注意**：
> 如果端口 20570 被占用，则 LazyLLM 会自动为您寻找下一个可用的端口，请留意系统的日志输出。

## 🧩 全量依赖包安装

对于想在本地进行模型的训练或推理，或是需要搭建 RAG 应用的用户，我们在 `requirements.full.txt` 中提供了完整使用 LazyLLM 的全部功能所需要的依赖库。同样地，您可以使用 `pip install -r requirements.full.txt` 来安装全量依赖包。安装完成之后，LazyLLM 可以实现微调，部署，推理，评测，RAG 等等基于大模型的高级功能。下列 Python 脚本可以启动一个大模型服务，并且与这个大模型对话：

```python
import lazyllm

t = lazyllm.TrainableModule('internlm2-chat-7b')
w = lazyllm.WebModule(t)
w.start().wait()
```

> **注意**：
> 如果您本地没有该模型的数据文件，LazyLLM 会自动为您下载至 `~/.lazyllm/model` 下。

祝您使用愉快！🌟