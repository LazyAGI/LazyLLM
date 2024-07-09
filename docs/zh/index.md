开始使用

欢迎来到LazyLLM！

LazyLLM是构建和优化多Agent应用的一站式开发工具，为应用开发过程中的全部环节（包括应用搭建、数据准备、模型部署、模型微调、评测等）提供了大量的工具，协助开发者用极低的成本构建AI应用，并可以持续的迭代优化效果。

LazyLLM使用Python开发，因此请确保您的计算机上已经安装了3.10以上的Python，并安装了pip作为Python的包管理工具。

如果您是从github上下载的LazyLLM，您需要先初始化一下LazyLLM的运行环境，确保它的依赖都被正确的安装。我们在 ``requirements.txt`` 和 ``requirements.full.txt`` 中提供了运行LazyLLM的基础依赖包的集合。

如果您所使用的计算机没有GPU，仅希望基于在线的模型服务和应用API来搭建自己的AI应用，则您仅需安装 ``requirements.txt`` 中通过的基础包即可。您可以进入LazyLLM的目录，使用命令 ``pip install -r requirements.txt`` 安装这些依赖包。

> **注意**：
    如果您遇到权限问题，您可能需要在命令前添加sudo，或者在命令后添加--user，以确保pip有足够的权限安装这些包


基础包安装完成之后，您就可以使用LazyLLM的基本功能搭建服务。下列python脚本可以部署一个具有简单web界面的服务:

```python

    # set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
    # or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
    import lazyllm
    t = lazyllm.OnlineChatModule(source="openai", stream=True)
    w = lazyllm.WebModule(t)
    w.start().wait()
```

这个python脚本将调用OpenAI的模型服务，并启动一个带多轮对话界面的web服务运行在本地的20570端口上。服务启动之后，使用浏览器访问http://localhost:20570，通过页面上的聊天机器人组件调用后台的大模型服务，LazyLLM会将模型的返回结果打印在聊天机器人组件中。

> **注意**：
    如果端口20570被占用，则LazyLLM会自动为您寻找下一个可用的端口，请留意系统的日志输出。


对于想在本地进行模型的训练或推理，或是需要搭建RAG应用的用户，我们在requirements.full.txt中提供了完整使用LazyLLM的全部功能所需要的依赖库。
同样地，您可以使用 ``pip install -r requirements.full.txt`` 来安装全量依赖包。安装完成之后，LazyLLM可以实现微调，部署，推理，评测，RAG等等基于大模型的高级功能。
下列python脚本可以启动一个大模型服务，并且与这个大模型对话：

```python

    import lazyllm

    t = lazyllm.TrainableModule('internlm2-chat-7b')
    w = lazyllm.WebModule(t)
    w.start().wait()
```
> **注意**：
    如果您本地没有该模型的数据文件，LazyLLM会自动为您下载至~/.lazyllm/model下。

祝您使用愉快。