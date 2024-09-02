<div align="center">
  <img src="https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/docs/assets/LazyLLM-logo.png" width="100%"/>
</div>

# LazyLLM: A  Low-code Development Tool For Building Multi-agent LLMs Applications.
[中文](README.CN.md) |  [EN](README.md)

[![CI](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml/badge.svg)](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub star chart](https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=flat-square)](https://star-history.com/#LazyAGI/LazyLLM)
[![](https://dcbadge.vercel.app/api/server/cDSrRycuM6?compact=true&style=flat)](https://discord.gg/cDSrRycuM6)

## What is LazyLLM?

LazyLLM is a low-code development tool for building multi-agent LLMs(large language models) applications. It assists developers in creating complex AI applications at very low costs and enables continuous iterative optimization. LazyLLM offers a convenient workflow for application building and provides numerous standard processes and tools for various stages of the application development process.<br>

The AI application development process based on LazyLLM follows the **prototype building -> data feedback -> iterative optimization** workflow. This means you can quickly build a prototype application using LazyLLM, then analyze bad cases using task-specific data, and subsequently iterate on algorithms and fine-tune models at critical stages of the application to gradually enhance the overall performance.<br>

**Tutorials**： https://docs.lazyllm.ai/ <br>
**Recent Plans**: [v0.2 PRD](https://aicarrier.feishu.cn/wiki/H8bIwORVtiAKv2kwRRJcyfwZnAo?from=from_copylink)<br>
Scan the QR code below with WeChat to join the group chat<br>
<img src="https://github.com/user-attachments/assets/8ad8fd14-b218-48b3-80a4-7334b2a32c5a" width=250/> <br>

## Features

**Convenient AI Application Assembly Process**: Even if you are not familiar with large models, you can still easily assemble AI applications with multiple agents using our built-in data flow and functional modules, just like Lego building.

**One-Click Deployment of Complex Applications**: We offer the capability to deploy all modules with a single click. Specifically, during the POC (Proof of Concept) phase, LazyLLM simplifies the deployment process of multi-agent applications through a lightweight gateway mechanism, solving the problem of sequentially starting each submodule service (such as LLM, Embedding, etc.) and configuring URLs, making the entire process smoother and more efficient. In the application release phase, LazyLLM provides the ability to package images with one click, making it easy to utilize Kubernetes' gateway, load balancing, and fault tolerance capabilities.

**Cross-Platform Compatibility**: Switch IaaS platforms with one click without modifying code, compatible with bare-metal servers, development machines, Slurm clusters, public clouds, etc. This allows developed applications to be seamlessly migrated to other IaaS platforms, greatly reducing the workload of code modification.<br>

**Support for Grid Search Parameter Optimization**: Automatically try different base models, retrieval strategies, and fine-tuning parameters based on user configurations to evaluate and optimize applications. This makes hyperparameter tuning efficient without requiring extensive intrusive modifications to application code, helping users quickly find the best configuration.<br>

**Efficient Model Fine-Tuning**: Support fine-tuning models within applications to continuously improve application performance. Automatically select the best fine-tuning framework and model splitting strategy based on the fine-tuning scenario. This not only simplifies the maintenance of model iterations but also allows algorithm researchers to focus more on algorithm and data iteration, without handling tedious engineering tasks.<br>


## What can you build with Lazyllm

LazyLLM can be used to build common artificial intelligence applications. Here are some examples.

### ChatBots

**This is a simple example of a chat bot.**

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat).start().wait()
```

If you want to use a locally deployed model, please ensure you have installed at least one inference framework (lightllm or vllm), and then use the following code

```python
import lazyllm
# Model will be downloaded automatically if you have an internet connection.
chat = lazyllm.TrainableModule('internlm2-chat-7b')
lazyllm.WebModule(chat, port=23466).start().wait()
```

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a chatbot by executing `lazyllm run chatbot`. If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a chatbot based on a local model by using `lazyllm run chatbot --model=internlm2-chat-7b`.

**This is an advanced bot example with multimodality and intent recognition.**

![Demo Multimodal bot](docs/assets/multimodal-bot.svg)

<details>
<summary>click to look up prompts and imports</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline
from lazyllm.tools import IntentClassifier

painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b')
with IntentClassifier(base) as ic:
    ic.case['Chat', base]
    ic.case['Speech Recognition', TrainableModule('SenseVoiceSmall')]
    ic.case['mage QA', TrainableModule('Mini-InternVL-Chat-2B-V1-5').deploy_method(deploy.LMDeploy)]
    ic.case['Drawing', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['Generate Music', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['Text to Speech', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
```

### RAG

![Demo RAG](docs/assets/demo_rag.svg)

<details>
<summary>click to look up prompts and imports</summary>

```python

import os
import lazyllm
from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'
```
</details>

Here is an online deployment example:

```python
documents = Document(dataset_path="your data path", embed=lazyllm.OnlineEmbeddingModule(), create_ui=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23466).start().wait()
```

Here is an example of a local deployment:

```python
documents = Document(dataset_path='/file/to/yourpath', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'))
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456).start().wait()
```

https://github.com/LazyAGI/LazyLLM/assets/12124621/77267adc-6e40-47b8-96a8-895df165b0ce

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a retrieval-augmented bot by executing `lazyllm run rag --documents=/file/to/yourpath`. If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a retrieval-augmented bot based on a local model by using `lazyllm run rag --documents=/file/to/yourpath --model=internlm2-chat-7b`.

### Stories Creator

<details>
<summary>click to look up prompts and imports</summary>

```python
import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter

toc_prompt="""
You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.

Please generate the corresponding list of nested dictionaries based on the following user input:

Example output:
[
    {
        "title": "# Level 1 Title",
        "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    },
    {
        "title": "## Level 2 Title",
        "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    },
    {
        "title": "### Level 3 Title",
        "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    }
]
User input is as follows:
"""

completion_prompt="""
You are now an intelligent assistant. Your task is to receive a dictionary containing `title` and `describe`, and expand the writing according to the guidance in `describe`.

Input example:
{
    "title": "# Level 1 Title",
    "describe": "This is the description for writing."
}

Output:
This is the expanded content for writing.
Receive as follows:

"""

writer_prompt = {"system": completion_prompt, "user": '{"title": {title}, "describe": {describe}}'}
```
</details>

Here is an online deployment example:

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.OnlineChatModule(stream=False).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(lazyllm.OnlineChatModule(stream=False).prompt(writer_prompt))
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.outline_writer)
lazyllm.WebModule(ppl, port=23466).start().wait()
```

Here is an example of a local deployment:

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.outline_writer)
lazyllm.WebModule(ppl, port=23466).start().wait()
```

### AI Painting Assistant

<details>
<summary>click to look up prompts and imports</summary>

```python
import lazyllm
from lazyllm import pipeline

prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
```
</details>

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```


## What can LazyLLM do

1. **Application Building**: Defines workflows such as pipeline, parallel, diverter, if, switch, and loop. Developers can quickly build multi-agent AI applications based on any functions and modules. Supports one-click deployment for assembled multi-agent applications, and also supports partial or complete updates to the applications.
2. **Platform-independent**: Consistent user experience across different computing platforms. Currently compatible with various platforms such as bare metal, Slurm, SenseCore, etc.
3. **Supports fine-tuning and inference for large models**:
    * Offline (local) model services:
        + Supports fine-tuning frameworks: collie, peft
        + Supports inference frameworks: lightllm, vllm
        + Supports automatically selecting the most suitable framework and model parameters (such as micro-bs, tp, zero, etc.) based on user scenarios..
    * Online services:
        + Supports fine-tuning services: GPT, SenseNova, Tongyi Qianwen
        + Supports inference services: GPT, SenseNova, Kimi, Zhipu, Tongyi Qianwen
        + Supports embedding inference services: OpenAI, SenseNova, GLM, Tongyi Qianwen
    * Support developers to use local services and online services uniformly.
4. **Supports common RAG (Retrieval-Augmented Generation) components**: Document, Parser, Retriever, Reranker, etc.
5. **Supports basic webs**: such as chat interface and document management interface, etc.

## Installation

### from source-code

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

`pip install -r requirements.full.txt` is used when you want to finetune, deploy or build your rag application.

### from pip

Only install lazyllm and necessary dependencies, you can use:
```bash
pip3 install lazyllm
```

Install lazyllm and all dependencies, you can use:
```bash
pip3 install lazyllm
lazyllm install full
```

## Design concept

The design philosophy of LazyLLM stems from a deep understanding of the current limitations of large models in production environments. We recognize that at this stage, large models cannot yet fully solve all practical problems end-to-end. Therefore, the AI application development process based on LazyLLM emphasizes "rapid prototyping, bad-case analysis using scenario-specific data, algorithmic experimentation, and model fine-tuning on key aspects to improve the overall application performance." LazyLLM handles the tedious engineering work involved in this process, offering convenient interfaces that allow users to focus on enhancing algorithmic effectiveness and creating outstanding AI applications.<br>

The goal of LazyLLM is to free algorithm researchers and developers from the complexities of engineering implementations, allowing them to concentrate on what they do best: algorithms and data, and solving real-world problems. Whether you are a beginner or an experienced expert, We hope LazyLLM can provide you with some assistance. For novice developers, LazyLLM thoroughly simplifies the AI application development process. They no longer need to worry about how to schedule tasks on different IaaS platforms, understand the details of API service construction, choose frameworks or split models during fine-tuning, or master any web development knowledge. With pre-built components and simple integration operations, novice developers can easily create tools with production value. For seasoned experts, LazyLLM offers a high degree of flexibility. Each module supports customization and extension, enabling users to seamlessly integrate their own algorithms and state-of-the-art production tools to build more powerful applications.<br>

To prevent you from being bogged down by the implementation details of dependent auxiliary tools, LazyLLM strives to ensure a consistent user experience across similar modules. For instance, we have established a set of Prompt rules that provide a uniform usage method for both online models (such as ChatGPT, SenseNova, Kimi, ChatGLM, etc.) and local models. This consistency allows you to easily and flexibly switch between local and online models in your applications.<br>

Unlike most frameworks on the market, LazyLLM carefully selects and integrates 2-3 tools that we believe are the most advantageous at each stage. This not only simplifies the user’s decision-making process but also ensures that users can build the most productive applications at the lowest cost. We do not pursue the quantity of tools or models, but focus on quality and practical effectiveness, committed to providing the optimal solutions. LazyLLM aims to provide a quick, efficient, and low-threshold path for AI application development, freeing developers' creativity, and promoting the adoption and popularization of AI technology in real-world production.<br>

Finally, LazyLLM is a user-centric tool. If you have any ideas or feedback, feel free to leave us a message. We will do our best to address your concerns and ensure that LazyLLM provides you with the convenience you need.<br>

## Architecture

![Architecture](docs/assets/Architecture.en.png)

## Basic concept

### Component

A Component is the smallest execution unit in LazyLLM; it can be either a function or a bash command. Components have three typical capabilities:
1. Cross-platform execution using a launcher, allowing seamless user experience:
  - EmptyLauncher: Runs locally, supporting development machines, bare metal, etc.
  - RemoteLauncher: Schedules execution on compute nodes, supporting Slurm, SenseCore, etc.
2. Implements a registration mechanism for grouping and quickly locating methods. Supports registration of functions and bash commands. Here is an example:

```python
import lazyllm
lazyllm.component_register.new_group('demo')

@lazyllm.component_register('demo')
def test(input):
    return f'input is {input}'

@lazyllm.component_register.cmd('demo')
def test_cmd(input):
    return f'echo input is {input}'

# >>> lazyllm.demo.test()(1)
# 'input is 1'
# >>> lazyllm.demo.test_cmd(launcher=launchers.slurm)(2)
# Command: srun -p pat_rd -N 1 --job-name=xf488db3 -n1 bash -c 'echo input is 2'
```

### Module

Modules are the top-level components in LazyLLM, equipped with four key capabilities: training, deployment, inference, and evaluation. Each module can choose to implement some or all of these capabilities, and each capability can be composed of one or more components. As shown in the table below, we have built-in some basic modules for everyone to use.

|      |Function | Training/Fine-tuning | Deployment | Inference | Evaluation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ActionModule | Can wrap functions, modules, flows, etc., into a Module | Supports training/fine-tuning of its Submodules through ActionModule | Supports deployment of its Submodules through ActionModule | ✅ | ✅ |
| UrlModule | Wraps any URL into a Module to access external services | ❌ | ❌ | ✅ | ✅ |
| ServerModule | Wraps any function, flow, or Module into an API service | ❌ | ✅ | ✅ | ✅
| TrainableModule | Trainable Module, all supported models are TrainableModules | ✅ | ✅ | ✅ | ✅ |
| WebModule | Launches a multi-round dialogue interface service | ❌ | ✅ | ❌ | ✅ |
| OnlineChatModule | Integrates online model fine-tuning and inference services | ✅ | ✅ | ✅ | ✅ |
| OnlineEmbeddingModule | Integrates online Embedding model inference services | ❌ | ✅ | ✅ | ✅ |


### Flow

Flow in LazyLLM defines the data stream, describing how data is passed from one callable object to another. You can use Flow to intuitively and efficiently organize and manage data flow. Based on various predefined Flows, we can easily build and manage complex applications using Modules, Components, Flows, or any callable objects. The Flows currently implemented in LazyLLM include Pipeline, Parallel, Diverter, Warp, IFS, Loop, etc., which can cover almost all application scenarios. Building applications with Flow offers the following advantages:
1. You can easily combine, add, and replace various modules and components; the design of Flow makes adding new features simple and facilitates collaboration between different modules and even projects.
2. Through a standardized interface and data flow mechanism, Flow reduces the repetitive work developers face when handling data transfer and transformation. Developers can focus more on core business logic, thus improving overall development efficiency.
3. Some Flows support asynchronous processing and parallel execution, significantly enhancing response speed and system performance when dealing with large-scale data or complex tasks.

## RoadMap

We plan to support the following features by the end of this year: <br>

RAG
- [x]  Refactor the RAG module to remove the dependency on llamaindex
- [ ]  Support online parser

One-Click Deployment of Applications
- [ ]  Support one-click generation of Docker, one-click application startup, supporting high concurrency and fault tolerance

Model Services
- [x]  Continue support for pre-training and RLHF
- [ ]  Support multiple inference services during model inference and achieve load balancing
- [x]  Support models for text-to-image and image-text QA, including VQA and SD
- [x]  Support voice models, including TTS and STT

Tools
- [x]  Support Function-Call & Agent
- [ ]  Integrate common search engines
- [ ]  Support common formatters
- [x]  Provide Prompter templates for common scenarios

User Experience Optimization
- [x] Optimize the flow of data in flow, support flexible data flow
