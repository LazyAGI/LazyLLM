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

LazyLLM is a low-code development tool for building **multi-agent** large language model applications. It assists developers in creating complex AI applications at very low costs and enables continuous iterative optimization. LazyLLM offers a convenient workflow for application building and provides numerous standard processes and tools for various stages of the application development process.<br>
The AI application development process based on LazyLLM follows **prototype building -> data feedback -> iterative optimization**, which means you can quickly build a prototype application using LazyLLM, then analyze bad cases using task-specific data, and subsequently iterate on algorithms and fine-tune models at critical stages of the application to gradually improve the overall application performance.<br>
LazyLLM is committed to the unity of agility and efficiency. Developers can efficiently iterate algorithms and then apply the iterated algorithms to industrial production, supporting multiple users, fault tolerance, and high concurrency.
**User Documentation**： https://docs.lazyllm.ai/ <br>
Scan the QR code below with WeChat to join the group chat(left) or learn more by watching a video(right)<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/8ad8fd14-b218-48b3-80a4-7334b2a32c5a" width=250/>
<img src="https://github.com/user-attachments/assets/7a042a97-1339-459e-a451-4bcd6cf64c12" width=250/>
</p>


## Features

**Convenient AI Application Assembly Process**: Even if you are not familiar with large models, you can still easily assemble AI applications with multiple agents using our built-in data flow and functional modules, just like Lego building.

**One-Click Deployment of Complex Applications**: We offer the capability to deploy all modules with a single click. Specifically, during the POC (Proof of Concept) phase, LazyLLM simplifies the deployment process of multi-agent applications through a lightweight gateway mechanism, solving the problem of sequentially starting each submodule service (such as LLM, Embedding, etc.) and configuring URLs, making the entire process smoother and more efficient. In the application release phase, LazyLLM provides the ability to package images with one click, making it easy to utilize Kubernetes' gateway, load balancing, and fault tolerance capabilities.

**Cross-Platform Compatibility**: Switch IaaS platforms with one click without modifying code, compatible with bare-metal servers, development machines, Slurm clusters, public clouds, etc. This allows developed applications to be seamlessly migrated to other IaaS platforms, greatly reducing the workload of code modification.<br>

**Unified User Experience for Different Technical Choices**: We provide a unified user experience for online models from different service providers and locally deployed models, allowing developers to freely switch and upgrade their models for experimentation. In addition, we also unify the user experience for mainstream inference frameworks, fine-tuning frameworks, relational databases, vector databases, and document databases.<br>

**Efficient Model Fine-Tuning**: Support fine-tuning models within applications to continuously improve application performance. Automatically select the best fine-tuning framework and model splitting strategy based on the fine-tuning scenario. This not only simplifies the maintenance of model iterations but also allows algorithm researchers to focus more on algorithm and data iteration, without handling tedious engineering tasks.<br>


## What can you build with Lazyllm

LazyLLM can be used to build common artificial intelligence applications. Here are some examples.

### 3.1 ChatBots

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
    ic.case['Image QA', TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)]
    ic.case['Drawing', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['Generate Music', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['Text to Speech', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
```

### 3.2 Retrieval-Augmented Generation

![Demo RAG](docs/assets/demo_rag.svg)

<details>
<summary>Click to view imports and prompt</summary>

```python

import os
import lazyllm
from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'
```
</details>

This is an online deployment example:

```python
documents = Document(dataset_path="your data path", embed=lazyllm.OnlineEmbeddingModule(), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

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
    ppl.llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456).start().wait()
```

https://github.com/LazyAGI/LazyLLM/assets/12124621/77267adc-6e40-47b8-96a8-895df165b0ce

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a retrieval-augmented bot by executing `lazyllm run rag --documents=/file/to/yourpath`. If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a retrieval-augmented bot based on a local model by using `lazyllm run rag --documents=/file/to/yourpath --model=internlm2-chat-7b`.

### 3.3 More Examples

For more examples, please refer to our official documentation [Usage Examples](https://docs.lazyllm.ai/zh-cn/stable/Cookbook/robot/)
* [Painting Master](https://docs.lazyllm.ai/en/stable/Cookbook/painting_master/)
* [Multimodal Chatbot](https://docs.lazyllm.ai/en/stable/Cookbook/multimodal_robot/)
* [Knowledge Base](https://docs.lazyllm.ai/en/stable/Cookbook/rag/)
* [Search Agent](https://docs.lazyllm.ai/en/latest/Cookbook/bocha_search/)
* [API Interaction Agent](https://docs.lazyllm.ai/en/latest/Cookbook/API_Interaction_Agent_demo/)
* [Tool-Calling Intelligent Agent](https://docs.lazyllm.ai/en/latest/Cookbook/flex_agent/)

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

### pip installation (recommended)

To install only lazyllm and necessary dependencies, you can use:
```bash
pip3 install lazyllm
```

To install lazyllm and all dependencies, you can use:
```bash
pip3 install lazyllm
lazyllm install full
```

### Installation from source

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

### Installation on Windows or macOS

For installation on Windows or macOS, please refer to our [tutorial](https://docs.lazyllm.ai/zh-cn/stable/Home/environment)

## Design Philosophy

The design philosophy of LazyLLM stems from a deep understanding of the current limitations of large models in production environments. We recognize that at this stage, large models cannot yet fully solve all practical problems end-to-end. Therefore, the AI application development process based on LazyLLM emphasizes "rapid prototyping, bad-case analysis using scenario-specific data, algorithmic experimentation, and model fine-tuning on key aspects to improve the overall application performance." LazyLLM handles the tedious engineering work involved in this process, offering convenient interfaces that allow users to focus on enhancing algorithmic effectiveness and creating outstanding AI applications.<br>

The goal of LazyLLM is to free algorithm researchers and developers from the complexities of engineering implementations, allowing them to concentrate on what they do best: algorithms and data, and solving real-world problems. Whether you are a beginner or an experienced expert, We hope LazyLLM can provide you with some assistance. For novice developers, LazyLLM thoroughly simplifies the AI application development process. They no longer need to worry about how to schedule tasks on different IaaS platforms, understand the details of API service construction, choose frameworks or split models during fine-tuning, or master any web development knowledge. With pre-built components and simple integration operations, novice developers can easily create tools with production value. For seasoned experts, LazyLLM offers a high degree of flexibility. Each module supports customization and extension, enabling users to seamlessly integrate their own algorithms and state-of-the-art production tools to build more powerful applications.<br>

To prevent you from being bogged down by the implementation details of dependent auxiliary tools, LazyLLM strives to ensure a consistent user experience across similar modules. For instance, we have established a set of Prompt rules that provide a uniform usage method for both online models (such as ChatGPT, SenseNova, Kimi, ChatGLM, etc.) and local models. This consistency allows you to easily and flexibly switch between local and online models in your applications.<br>

Unlike most frameworks on the market, LazyLLM carefully selects and integrates 2-3 tools that we believe are the most advantageous at each stage. This not only simplifies the user’s decision-making process but also ensures that users can build the most productive applications at the lowest cost. We do not pursue the quantity of tools or models, but focus on quality and practical effectiveness, committed to providing the optimal solutions. LazyLLM aims to provide a quick, efficient, and low-threshold path for AI application development, freeing developers' creativity, and promoting the adoption and popularization of AI technology in real-world production.<br>

Finally, LazyLLM is a user-centric tool. If you have any ideas or feedback, feel free to leave us a message. We will do our best to address your concerns and ensure that LazyLLM provides you with the convenience you need.<br>

## Architecture

![Architecture](docs/assets/Architecture.en.png)

## Basic Concepts

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

## Future Plans

### Timeline
V0.6 Expected to start from September 1st, lasting 3 months, with continuous small version releases in between, such as v0.6.1, v0.6.2
V0.7 Expected to start from December 1st, lasting 3 months, with continuous small version releases in between, such as v0.7.1, v0.7.2
v0.8 Expected to start from March 2026, lasting 3 months, focusing on improving system observability and reducing user debugging costs
v0.9 Expected to start from June 2026, lasting 3 months, focusing on improving the overall system running speed

### Feature Modules
9.2.1 RAG
  - 9.2.1.1 Engineering
    - Integrate LazyRAG capabilities into LazyLLM (V0.6)
    - Extend RAG's macro Q&A capabilities to multiple knowledge bases (V0.6)
    - ✅ RAG modules fully support horizontal scaling, supporting multi-machine deployment of RAG algorithm collaboration (V0.6)
    - Integrate at least 1 open-source knowledge graph framework (V0.6)
    - Support common data splitting strategies, no less than 20 types, covering various document types (V0.6 - v0.7)
  - 9.2.1.2 Data Capabilities
    - Table parsing (V0.6 - 0.7)
    - CAD image parsing (V0.7 -)
  - 9.2.1.3 Algorithm Capabilities
    - Support processing of relatively structured texts like CSV (V0.6)
    - Multi-hop retrieval (links in documents, references, etc.) (V0.6)
    - Information conflict handling (V0.7)
    - AgenticRL & code-writing problem-solving capabilities (V0.7)

9.2.2 Functional Modules
  - ✅ Support memory capabilities (V0.6)
  - Support for distributed Launcher (V0.7)
  - ✅ Database-based Globals support (V0.6)
  - ServerModule can be published as MCP service (v0.7)
  - Integration of online sandbox services (v0.7)

9.2.3 Model Training and Inference
  - ✅ Support OpenAI interface deployment and inference (V0.6)
  - Unify fine-tuning and inference prompts (V0.7)
  - Provide fine-tuning examples in Examples (V0.7)
  - Integrate 2-3 prompt repositories, allowing direct selection of prompts from prompt repositories (V0.6)
  - ✅ Support more intelligent model type judgment and inference framework selection, refactor and simplify auto-finetune framework selection logic (V0.6)
  - Full-chain GRPO support (V0.7)

9.2.4 Documentation
  - Complete API documentation, ensure every public interface has API documentation, with consistent documentation parameters and function parameters, and executable sample code (V0.6)
  - Complete CookBook documentation, increase cases to 50, with comparisons to LangChain/LlamaIndex (code volume, speed, extensibility) (V0.6 - v0.7)
  - ✅ Complete Environment documentation, supplement installation methods on win/linux/macos, supplement package splitting strategies (V0.6)
  - Complete Learn documentation, first teach how to use large models; then teach how to build agents; then teach how to use workflows; finally teach how to build RAG (V0.6)

9.2.5 Quality
  - Reduce CI time to within 10 minutes by mocking most modules (V0.6)
  - Add daily builds, put high-time-consuming/token tasks in daily builds (V0.6)

9.2.6 Development, Deployment and Release
  - Debug optimization (v0.7)
  - Process monitoring [output + performance] (v0.7)
  - Environment isolation and automatic environment setup for dependent training and inference frameworks (V0.6)

9.2.7 Ecosystem
  - ✅ Promote LazyCraft open source (V0.6)
  - Promote LazyRAG open source (V0.7)
  - ✅ Upload code to 2 code hosting websites other than Github and strive for community collaboration (V0.6)
