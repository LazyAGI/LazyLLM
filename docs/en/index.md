# Getting Started

Welcome to **LazyLLM**!

`LazyLLM` is an all-in-one development tool for building and optimizing multi-Agent applications. It provides a wealth of tools for all stages of application development (including application construction, data preparation, model deployment, model fine-tuning, evaluation, etc.), helping developers to build AI applications at a very low cost and continuously iterate and optimize the effects.

## Environment Preparation

You can set up the `LazyLLM` development environment using any of the following methods:

### Manual Configuration

`LazyLLM` is developed in Python, and we need to ensure that `Python`, `Pip`, and `Git` are already installed on our system.

First, prepare and activate a virtual environment named `lazyllm-venv`:

```bash
python3 -m venv lazyllm-venv
source lazyllm-venv/bin/activate
```

If everything runs normally, you should see the prompt `(lazyllm-venv)` at the beginning of the command line. All our subsequent operations will be conducted within this virtual environment.

Download the code for `LazyLLM` from GitHub:

```bash
git clone https://github.com/LazyAGI/LazyLLM.git
```

And switch to the downloaded code directory:

```bash
cd LazyLLM
```

Install the basic dependencies:

```bash
pip3 install -r requirements.txt
```

Add `LazyLLM` to the module search path:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

So that we can find it from any directory.

### Using the Docker Image

We provide a Docker image containing the latest version of `LazyLLM`, ready to use right out of the box:

```bash
docker pull lazyllm/lazyllm
```

You can also view and pull the required version from [https://hub.docker.com/r/lazyllm/lazyllm/tags](https://hub.docker.com/r/lazyllm/lazyllm/tags).

### Installing from Pip

`LazyLLM` supports direct installation via `pip`, the following three installation methods correspond to the use of different features:

1. Install the minimum dependency package for the basic functionality of `LazyLLM`. This can support the fine-tuning and inference of various online models.

    ```bash
    pip3 install lazyllm
    ```

2. Install the minimum dependency package for all features of `LazyLLM`. Not only does it support the fine-tuning and inference of online models, but it also supports the fine-tuning (mainly dependent on `LLaMA-Factory`) and inference (mainly dependent on `vLLM`) of offline models.

    ```bash
    pip3 install lazyllm
    lazyllm install standard
    ```

3. Install all dependency packages of `LazyLLM`, all features as well as advanced features are supported, such as automatic framework selection (`AutoFinetune`, `AutoDeploy`, etc.), more offline inference tools (such as `LightLLM`), and more offline training tools (such as `AlpacaloraFinetune`, `CollieFinetune`, etc.).

    ```bash
    pip3 install lazyllm
    lazyllm install full
    ```

## Hello, world!

To give you a basic understanding of `LazyLLM`, we will use it to create a chatbot based on the conversation capabilities provided by the [platform](#platform) below.

First, if you don't have an account on the corresponding platform, you need to register an account on the platform first, then get the required key according to the link to get the [API key](#platform) of the platform below(not: sensenova needs to get two keys), and set the corresponding evironment variables:

```bash
export LAZYLLM_<platform environment variable name in uppercase>_API_KEY=<your obtained api key>
```

Next, open an editor and enter the following code, save it as `chat.py`:

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

Finally, run our demo:

```bash
python3 chat.py
```

Once the input prompt appears, type your question and press Enter, and you should see the answer after a short wait.

Let’s briefly introduce the functionality of this code.

First, statement 1 imports the `lazyllm` module, and in statement 2, an instance of the online chat service named `chat` is created. We then enter an infinite loop that will only exit when the string “quit” is received (statement 4). Statement 3 prints the input prompt and saves the user’s input in the variable `query`. Statement 5 passes the user’s input to the chat module, which sends a request to the Nova model’s online service and saves the reply returned by Nova in the variable `res`. Statement 6 prints the received reply on the screen.

`LazyLLM` has built-in support for the following platforms:
[](){#platform}

| Platform | API Key Acquisition URL              | Environment Variables to Set                            |
|:---------|:-------------------------------------|:--------------------------------------------------------|
| [Nova](https://platform.sensenova.cn/)     | [API Keys](https://platform.sensenova.cn/doc?path=/platform/helpdoc/help.md)       | LAZYLLM_SENSENOVA_API_KEY, LAZYLLM_SENSENOVA_SECRET_KEY |
| [OpenAI](https://openai.com/index/openai-api/)   | [API Keys](https://platform.openai.com/api-keys) | LAZYLLM_OPENAI_API_KEY                                  |
| [Zhipu](https://open.bigmodel.cn/)    | [API Keys](https://open.bigmodel.cn/usercenter/apikeys)            | LAZYLLM_GLM_API_KEY                                     |
| [Kimi](https://platform.moonshot.cn/)     | [API Keys](https://platform.moonshot.cn/console/api-keys)        | LAZYLLM_KIMI_API_KEY                                    |
| [Qwen](https://help.aliyun.com/zh/dashscope/developer-reference/use-qwen-by-api)     | [API Keys](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key)     | LAZYLLM_QWEN_API_KEY                                    |

You can use the corresponding platform by setting different environment variables.

## Going Further: Multi-turn Dialogue

The example above demonstrates a question-and-answer format where each question starts a new dialogue and does not continue from the previous answer. Let’s modify it slightly to enable the robot to support multi-turn dialogue:

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

The corresponding statements with numbers are the same as in the previous question-and-answer version, and the running method is the same. The main differences in the code are as follows:

* Statement 7 adds a `history` field to keep track of the conversation history;
* Statement 5' passes the current `query` and the `history` to the remote server;
* Statement 8 appends the current dialogue's question and answer to the `history` field.

## Using the Web Interface

`LazyLLM` comes with a built-in web interface module `WebModule`, which facilitates the quick setup of various common applications:

```python
import lazyllm

chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat, port=23333).start().wait()
```

The `WebModule` accepts two parameters: the chat module for conversation and the port number for the web server to listen on. After calling the member function `start()` to successfully start, call `wait()` to block and wait for user operations on the web interface. You can access `http://localhost:23333` with your browser to interact with the chatbot component on the page, which will call the large model service in the background. `LazyLLM` will display the model’s return results on the page.

!!! Note "Note"

    If there is an error starting up or accessing the web page, please check the error information in the terminal window to see if the port is occupied by another application, or if a proxy is enabled, or if it is blocked by a firewall.

## Using Command Line Interface

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a chatbot by executing:

```bash
lazyllm run chatbot
```

If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a chatbot based on a local model by using:

```bash
lazyllm run chatbot --model=internlm2-chat-7b
```

-----

This concludes the introductory section of `LazyLLM`. The following chapters will explore the powerful features of `LazyLLM` from different aspects.
