# 开始使用

欢迎来到 **LazyLLM**！

`LazyLLM` 是构建和优化多 Agent 应用的一站式开发工具，为应用开发过程中的全部环节（包括应用搭建、数据准备、模型部署、模型微调、评测等）提供了大量的工具，协助开发者用极低的成本构建 AI 应用，并可以持续地迭代优化效果。

## 环境准备

可以用以下任一方式搭建 `LazyLLM` 开发环境。

### 手动配置

`LazyLLM` 基于 Python 开发，我们需要保证系统中已经安装好了 `Python`， `Pip` 和 `Git`。

首先准备一个名为 `lazyllm-venv` 的虚拟环境并激活：

```bash
python3 -m venv lazyllm-venv
source lazyllm-venv/bin/activate
```

如果运行正常，你可以在命令行的开头看到 `(lazyllm-venv)` 的提示。接下来我们的操作都在这个虚拟环境中进行。

从 GitHub 下载 `LazyLLM` 的代码：

```bash
git clone https://github.com/LazyAGI/LazyLLM.git
```

并切换到下载后的代码目录：

```bash
cd LazyLLM
```

安装基础依赖：

```bash
pip3 install -r requirements.txt
```

把 `LazyLLM` 加入到模块搜索路径中：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

这样我们在任意目录下都可以找到它。

### 拉取 Docker 镜像

我们提供了包含最新版本的 `LazyLLM` 的 docker 镜像，开箱即用：

```bash
docker pull lazyllm/lazyllm
```

也可以从 [https://hub.docker.com/r/lazyllm/lazyllm/tags](https://hub.docker.com/r/lazyllm/lazyllm/tags) 查看并拉取需要的版本。

### 从 Pip 安装

`LazyLLM` 支持用 `pip` 直接安装,下面三种安装方式分别对应不同功能的使用

1. 安装 `LazyLLM` 基础功能的最小依赖包。可以支持线上各类模型的微调和推理。

    ```bash
    pip3 install lazyllm
    ```

2. 安装 `LazyLLM` 的所有功能最小依赖包。不仅支持线上模型的微调和推理，而且支持离线模型的微调（主要依赖 `LLaMA-Factory`）和推理（主要依赖 `vLLM`）。

    ```bash
    pip3 install lazyllm
    lazyllm install standard
    ```

3. 安装 `LazyLLM` 的所有依赖包，所有功能以及高级功能都支持，比如自动框架选择（`AutoFinetune`、`AutoDeploy` 等）、更多的离线推理工具（如增加 `LightLLM` 等工具）、更多的离线训练工具（如增加 `AlpacaloraFinetune`、`CollieFinetune` 等工具）。

    ```bash
    pip3 install lazyllm
    lazyllm install full
    ```

## Hello, world!

为了让大家对 `LazyLLM` 有个初步的认识，我们基于下面提供的 [平台](#platform) 提供的对话功能，使用 `LazyLLM` 来实现一个聊天机器人。

首先如果没有对应平台的账号，就需要先在平台注册一个账号，然后根据下面平台对应的获取 [API key](#platform) 的链接获取所需要的key(注意:sensenova需要获取两个key)，并设置对应的环境变量：

```bash
export LAZYLLM_<使用的平台环境变量名称，大写>_API_KEY=<申请到的api key>
```

接着打开编辑器输入以下代码，保存为 `chat.py`：

```python
import lazyllm                                          #(1)

chat = lazyllm.OnlineChatModule()                       #(2)
while True:
    query = input("query(enter 'quit' to exit): ")      #(3)
    if query == "quit":                                 #(4)
        break
    res = chat.forward(query)                           #(5)
    print(f"answer: {res}")                             #(6)
```

最后运行我们的 demo：

```bash
python3 chat.py
```

当出现输入提示之后，输入我们的问题并回车，稍等一会就可以看到回答了。

我们来简单介绍一下这段代码的功能。

首先语句 1 引入了模块 `lazyllm`，并且在语句 2 中生成了一个在线聊天服务的实例 `chat`。接着我们进入一个无限循环，只有当接收到 "quit" 这个字符串才会退出（语句 4）。语句 3 打印输入提示，并把用户的输入保存在 `query` 这个变量中。语句 5 把用户的输入内容传给聊天模块，由聊天模块向日日新模型在线服务发送请求，并把日日新返回的回复保存在变量 `res` 中。语句 6 把收到的回复打印到屏幕上。

`LazyLLM` 内建了以下平台的支持：
[](){#platform}

| 平台     | 获取 api key                         | 需要设置的环境变量                                           |
| :------- | :----------------------------------- | :----------------------------------------------------------- |
| [日日新](https://platform.sensenova.cn/)   | [获取访问密钥](https://platform.sensenova.cn/doc?path=/platform/helpdoc/help.md)       | `LAZYLLM_SENSENOVA_API_KEY`,  `LAZYLLM_SENSENOVA_SECRET_KEY` |
| [OpenAI](https://openai.com/index/openai-api/)   | [获取访问密钥](https://platform.openai.com/api-keys) | `LAZYLLM_OPENAI_API_KEY`                                     |
| [智谱](https://open.bigmodel.cn/)     | [获取访问密钥](https://open.bigmodel.cn/usercenter/apikeys)            | `LAZYLLM_GLM_API_KEY`                                        |
| [Kimi](https://platform.moonshot.cn/)     | [获取访问密钥](https://platform.moonshot.cn/console/api-keys)        | `LAZYLLM_KIMI_API_KEY`                                       |
| [通义千问](https://help.aliyun.com/zh/dashscope/developer-reference/use-qwen-by-api) | [获取访问密钥](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key)     | `LAZYLLM_QWEN_API_KEY`                                       |

可以通过设置不同的环境变量来使用对应的平台。

## 再多一点：多轮对话

上面的例子演示的是一问一答，每个问题都是新一轮的对话，不会接着前面的回答结果来继续推导。接下来我们稍加改动，让机器人支持多轮对话：

```python
import lazyllm                                           #(1)

chat = lazyllm.OnlineChatModule()                        #(2)

# history has the form of [[query1, answer1], [query2, answer2], ...]
history = []                                             #(7)

while True:
    query = input("query(enter 'quit' to exit): ")       #(3)
    if query == "quit":                                  #(4)
        break
    res = chat(query, llm_chat_history=history)          #(5')
    print(f"answer: {res}")                              #(6)
    history.append([query, res])                         #(8)
```

对应标号的语句和前面一问一答的版本一样，运行方式也一样。代码中不同的地方主要有下面这些：

* 语句 7 加了个 `history` 字段，用来保存对话的历史；
* 语句 5' 传给远程服务器的内容，除了当前 `query` 外，还把历史内容 `history` 也传进去了；
* 语句 8 把本次对话的问答内容拼接到 `history` 之后。

## 使用 web 界面

`LazyLLM` 内建了一个 web 界面模块 `WebModule`，方便快速搭建各类常见的应用：

```python
import lazyllm

chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat, port=23333).start().wait()
```

`WebModule` 接受两个参数：用于对话的模块 `chat` 和作为 web server 监听的端口号 `port`。调用成员函数 `start()` 启动成功之后，接着调用 `wait()` 阻塞等待用户在 web 界面上的操作。我们可以使用浏览器访问 [http://localhost:23333](http://localhost:23333)，通过页面上的聊天机器人组件调用后台的大模型服务，`LazyLLM` 会将模型的返回结果展示在页面上。

!!! Note "注意"

    如果启动报错或者网页访问出错，请查看终端窗口的错误信息，检查是否端口被其它应用占用，或者启用了代理，或者被防火墙拦截等。

## 使用命令行工具

如果你是使用 `pip` 安装的 `lazyllm` ，并且保证python环境的`bin`目录已经在`$PATH`中，则你可以通过执行:

```bash
lazyllm run chatbot
```

来快速启动一个对话机器人。如果你想使用本地模型，则需要用`--model`参数指定模型名称，例如你可以用:

```bash
lazyllm run chatbot --model=internlm2-chat-7b
```

来启动基于本地模型的对话机器人。

-----

以上就是 `LazyLLM` 的入门介绍，接下来的章节会从不同的方面来探索 `LazyLLM` 的强大功能。
