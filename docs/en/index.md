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

If you want to use all features of LazyLLM, you can install the full set of dependencies by running the following command:

```bash
pip3 install -r requirements.full.txt
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

`LazyLLM` supports direct installation via `pip`:

```bash
pip3 install lazyllm
```

### Install Dependencies for Different Scenarios

After successfully installing `LazyLLM`, you can  install additional dependencies for specific use cases by using `lazyllm install xxx` in the terminal.

For example:

To install the **minimal dependencies** required for all key features of `LazyLLM`, run:

```bash
lazyllm install standard
```

This not only supports ​**online model fine-tuning and inference**​, but also enables **offline model fine-tuning** (mainly via `LLaMA-Factory`) and **local inference** (mainly via `vLLM`).

To install **all dependencies** of `LazyLLM`, including ​**all core and advanced features**​, run:

```bash
lazyllm install full
```

For more dependency groups for Specific Scenarios, you can install specific sets of dependencies based on your use case:

* ​**alpaca-lora**​: Install dependencies for the **Alpaca-LoRA** fine-tuning framework, suitable for lightweight local model fine-tuning tasks.
* ​**colie**​: Install dependencies for the **Collie** fine-tuning framework, supporting high-performance and distributed training of large models locally.
* ​**llama-factory**​: Install dependencies for the **LLaMA-Factory** fine-tuning framework, compatible with the LLaMA series and other mainstream local models.
* ​**finetune-all**​: Install all fine-tuning frameworks at once, including Alpaca-LoRA, Collie, and LLaMA-Factory — ideal for users needing compatibility with multiple training tools.
* ​**vllm**​: Install dependencies for the **vLLM** local inference framework, offering high-concurrency, low-latency inference performance.
* ​**lmdeploy**​: Install dependencies for the **LMDeploy** inference framework, optimized for deploying large language models in local environments.
* ​**lightllm**​: Install dependencies for the **LightLLM** inference framework, offering lightweight inference capabilities — ideal for resource-constrained environments.
* ​**infinity**​: Install dependencies for the **Infinity** framework, enabling fast local embedding inference, suitable for vector search and RAG use cases.
* ​**deploy-all**​: Install all local inference frameworks, including LightLLM, vLLM, LMDeploy, and Infinity — great for users who need flexibility in local deployment backends.
* ​**multimodal**​: Install dependencies for multimodal features, including speech generation, text-to-image, and other cross-modal capabilities.
* ​**rag-advanced**​: Install advanced RAG system features, including vector database support and embedding model fine-tuning — ideal for building enterprise-grade knowledge-based QA systems.
* ​**agent-advanced**​: Install advanced features for Agent systems, including integration with the **MCP** framework for complex task planning and tool invocation.
* ​**dev**​: Install developer dependencies, including code formatting tools and testing frameworks — recommended for contributing to the project or local development.

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

| Platform                                                                         | API Key Acquisition URL                                                                                                                                                       | Environment Variables to Set                               |
| :------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| [Nova](https://platform.sensenova.cn/)                                           | [API Keys(ak and sk)](https://console.sensecore.cn/help/docs/model-as-a-service/nova/),[API Keys(only api key)](https://console.sensecore.cn/aistudio/management/api-key) | LAZYLLM_SENSENOVA_API_KEY,LAZYLLM_SENSENOVA_SECRET_KEY |
| [OpenAI](https://openai.com/index/openai-api/)                                   | [API Keys](https://platform.openai.com/api-keys)                                                                                                                              | LAZYLLM_OPENAI_API_KEY                                     |
| [Zhipu](https://open.bigmodel.cn/)                                               | [API Keys](https://open.bigmodel.cn/usercenter/apikeys)                                                                                                                       | LAZYLLM_GLM_API_KEY                                        |
| [Kimi](https://platform.moonshot.cn/)                                            | [API Keys](https://platform.moonshot.cn/console/api-keys)                                                                                                                     | LAZYLLM_KIMI_API_KEY                                       |
| [Qwen](https://help.aliyun.com/zh/dashscope/developer-reference/use-qwen-by-api) | [API Keys](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key)                                                                 | LAZYLLM_QWEN_API_KEY                                       |
| [Doubao](https://www.volcengine.com/product/doubao)                              | [API Keys](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)                                                                                                   | LAZYLLM_DOUBAO_API_KEY                                     |

You can use the corresponding platform by setting different environment variables.

!!! Note "Note"
There are two ways to configure the key on the Nova platform. One is to configure both ak (api key) and sk (secret key), that is, you need to configure both the `LAZYLLM_SENSENOVA_API_KEY` and `LAZYLLM_SENSENOVA_SECRET_KEY` variables. The other is to only configure the api key, that is, you only need to configure the `LAZYLLM_SENSENOVA_API_KEY` variable.

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

---

This concludes the introductory section of `LazyLLM`. The following chapters will explore the powerful features of `LazyLLM` from different aspects.

