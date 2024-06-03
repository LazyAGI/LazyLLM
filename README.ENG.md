# LazyLLM: A One-Stop Development Tool for Building and Optimizing Multi-Agent Applications.
[中文](README.md) |  [EN](README.ENG.md)

## What is LazyLLM?

LazyLLM provides a wide array of tools for every stage of the application development process, including application building, data preparation, model deployment, model fine-tuning, and evaluation. It assists developers in creating AI applications at extremely low costs and enables continuous iterative optimization of their performance.

## Features

**Efficient Multi-Agent AI Application Development**: Easily assemble AI applications with multiple agents and deploy all sub-services with a single click. LazyLLM simplifies the deployment process of multi-agent applications with a lightweight gateway mechanism, addressing the traditional need to sequentially start each sub-module service (such as LLM, Embedding, etc.) and configure URLs, making the entire process smoother and more efficient.<br>

**Cross-Platform Compatibility**: Switch IaaS platforms with one click without modifying code, compatible with bare-metal servers, development machines, Slurm clusters, public clouds, etc. This allows developed applications to be seamlessly migrated to other IaaS platforms, greatly reducing the workload of code modification.<br>

**Support for Grid Search Parameter Optimization**: Automatically try different base models, retrieval strategies, and fine-tuning parameters based on user configurations to evaluate and optimize applications. This makes hyperparameter tuning efficient without requiring extensive intrusive modifications to application code, helping users quickly find the best configuration.<br>

**Efficient Model Fine-Tuning**: Support fine-tuning models within applications to continuously improve application performance. Automatically select the best fine-tuning framework and model splitting strategy based on the fine-tuning scenario. This not only simplifies the maintenance of model iterations but also allows algorithm researchers to focus more on algorithm and data iteration, without handling tedious engineering tasks.<br>


## What can you build with Lazyllm

LazyLLM can be used to build common artificial intelligence applications. Here are some examples.

### ChatBots

```python
t = lazyllm.OnlineChatModule('llama2-7b', stream=True)
w = lazyllm.WebModule(t)
w.start()
```

### RAG

### Stories Creator

### AI Agent

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
4. **Supports common RAG (Retrieval-Augmented Generation) components**: Document, Parser, Retriever, Reranker, etc.
5. **Supports basic webs**: such as chat interface and document management interface, etc.

## Installation

Only install lazyllm and necessary dependencies, you can use:
```bash
pip install lazyllm
```

Install lazyllm and all dependencies, you can use:
```bash
pip install lazyllm-full
```

## Design concept

The design philosophy of LazyLLM stems from a deep understanding of the current limitations of large models in production environments. We recognize that at this stage, large models cannot yet fully solve all practical problems end-to-end. Therefore, the AI application development process based on LazyLLM emphasizes "rapid prototyping, bad-case analysis using scenario-specific data, algorithmic experimentation, and model fine-tuning on key aspects to improve the overall application performance." LazyLLM handles the tedious engineering work involved in this process, offering convenient interfaces that allow users to focus on enhancing algorithmic effectiveness and creating outstanding AI applications.<br>

The goal of LazyLLM is to free algorithm researchers and developers from the complexities of engineering implementations, allowing them to concentrate on what they do best: algorithms and data, and solving real-world problems. Whether you are a beginner or an experienced expert, I hope LazyLLM can provide you with some assistance. For novice developers, LazyLLM thoroughly simplifies the AI application development process. They no longer need to worry about how to schedule tasks on different IaaS platforms, understand the details of API service construction, choose frameworks or split models during fine-tuning, or master any web development knowledge. With pre-built components and simple integration operations, novice developers can easily create tools with production value. For seasoned experts, LazyLLM offers a high degree of flexibility. Each module supports customization and extension, enabling users to seamlessly integrate their own algorithms and state-of-the-art production tools to build more powerful applications.<br>

Unlike most frameworks on the market, LazyLLM carefully selects and integrates 2-3 tools that we believe are the most advantageous at each stage. This not only simplifies the user’s decision-making process but also ensures that users can build the most productive applications at the lowest cost. We do not pursue the quantity of tools or models, but focus on quality and practical effectiveness, committed to providing the optimal solutions.<br>

In summary, LazyLLM aims to provide a quick, efficient, and low-threshold path for AI application development, freeing developers' creativity, and promoting the adoption and popularization of AI technology in real-world production.<br>

## Architecture

![Architecture](docs/Architecture.png)

## Basic concept

### Component

A Component is the smallest execution unit in LazyLLM; it can be either a function or a bash command. Components have three typical capabilities:
1. Cross-platform execution using a launcher, allowing seamless user experience:
  - EmptyLauncher: Runs locally, supporting development machines, bare metal, etc.
  - RemoteLauncher: Schedules execution on compute nodes, supporting Slurm, SenseCore, etc.
2. Implements a registration mechanism for grouping and quickly locating methods. Supports registration of functions and bash commands. Here is an example:

```python
import lazyllm
lazyllm.llmregister.make_group('demo')

@lazyllm.llmregister('demo')
def test(input):
    return f'input is {input}'

@lazyllm.llmregister.cmd('demo')
def test_cmd(input):
    return f'echo input is {input}'

# >>> demo.test()(1)
# 'input is 1'
# >>> demo.test_cmd(launcher=launchers.slurm)(2)
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

## Documentation

## Contributing

## Contributors