# 🚀 Getting Started

Welcome to **LazyLLM**! 🎉

LazyLLM is an all-in-one development tool for building and optimizing multi-agent applications. It provides a wide range of tools for every stage of application development, including application setup, data preparation, model deployment, model fine-tuning, evaluation, and more. It assists developers in building AI applications at a very low cost and allows continuous iterative optimization.

## 🛠️ Environment Setup

LazyLLM is developed in Python, so please ensure that your computer has Python 3.10 or higher installed, along with pip as the Python package management tool.

If you have downloaded LazyLLM from GitHub, you need to initialize the LazyLLM runtime environment to ensure that all dependencies are correctly installed. We provide the necessary dependencies for running LazyLLM in `requirements.txt` and `requirements.full.txt`.

### Installing Basic Dependencies

If your computer does not have a GPU and you only wish to build your AI application based on online model services and application APIs, you only need to install the basic packages listed in `requirements.txt`. Navigate to the LazyLLM directory and use the command `pip install -r requirements.txt` to install these dependencies.

> **Note**:
> If you encounter permission issues, you may need to add `sudo` before the command or `--user` after the command to ensure pip has sufficient permissions to install these packages.

## 🚀 Deploying Basic Functionality

Once the basic packages are installed, you can use LazyLLM's basic features to set up services. The following Python script can deploy a service with a simple web interface:

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
t = lazyllm.OnlineChatModule(source="openai", stream=True)
w = lazyllm.WebModule(t)
w.start().wait()
```

This Python script will call OpenAI's model service and start a web service with a multi-turn conversation interface running on port 20570 of your local machine. After the service starts, visit [http://localhost:20570](http://localhost:20570) with your browser. The chatbot component on the page will call the backend large model service, and LazyLLM will print the model's response in the chatbot component.

> **Note**:
> If port 20570 is occupied, LazyLLM will automatically find the next available port for you. Please pay attention to the system log output.

## 🧩 Installing Full Dependencies

For users who wish to perform model training or inference locally, or need to build RAG applications, we provide all the dependencies required for full functionality of LazyLLM in `requirements.full.txt`. Similarly, you can use `pip install -r requirements.full.txt` to install all dependencies. Once installed, LazyLLM can perform fine-tuning, deployment, inference, evaluation, RAG, and other advanced features based on large models. The following Python script can start a large model service and interact with it:

```python
import lazyllm

t = lazyllm.TrainableModule('internlm2-chat-7b')
w = lazyllm.WebModule(t)
w.start().wait()
```

> **Note**:
> If the model data files are not available locally, LazyLLM will automatically download them to `~/.lazyllm/model`.

Happy coding! 🌟