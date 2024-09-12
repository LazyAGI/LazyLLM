# Painting Master

We will build an AI Painting Master based on the previous section [Build Your First Chatbot](robot.md).

!!! abstract "Through this section, you will learn about the following key points of LazyLLM"

    - How to set prompt for the model;
    - How to assemble applications based on the [Pipeline][lazyllm.flow.Pipeline] control flow;
    - How to use non-LLM models in LazyLLM;

## Design Concept

Firstly, to draw images, we need a model that can draw, here we choose Stable Diffusion3;

Then, since drawing models require some specialized prompt words, and we also want it to support Chinese, we consider introducing a single-turn chatbot to do translation and write drawing prompts;

Finally, we combine the above two modules into a workflow and overlay a user interface on top.

So the design looks like this:

![Painting Master](../assets/2_painting_master1.svg)

## Code Implementation

Let's implement the above design ideas based on LazyLLM.

### Designing Prompt

We design a prompt that specifies its role as a drawing prompt word master, and it can translate and generate or expand prompt words based on user input. Specifically:

[](){#use-prompt}

```python
prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
```

### Setting up the Model

Next, we will use the [Build Your First Chatbot](robot.md) from the previous section, and set the prompt we just wrote for it.

```python
llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
```

At the same time, we also need to introduce the SD3 model:

```python
sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
```

Here, the SD3 model is a non-LLM model, but it is used in the same way as the LLM model—simply specify the model name directly in [TrainableModule][lazyllm.module.TrainableModule].

### Assembling the Application

LazyLLM has many types of control flows, which are generally used to control the flow of data. By using control flows to assemble modules, we construct our Painting Master. Here we choose to use [Pipeline][lazyllm.flow.Pipeline] to implement sequential execution: first the large model generates the prompt words, then feeds the prompt words to the SD3 model to get the image.

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
```

In the above code snippet, we use the `pipeline` context manager to build a LazyLLM control flow.

[](){#use-pipeline}

```python
with pipeline() as ppl:
```

This line of code creates a pipeline instance named `ppl` and enters a context manager.

!!! Note

    - To add a module to `ppl`, you need to add attributes: `ppl.llm = xxx` and `ppl.sd3 = xxx`
    - Modules that are not explicitly added to `ppl` will not go through the control flow;

### Starting the Application

Finally, we wrap the control flow `ppl` into a client and start the deployment (`start()`), and keep the client from closing after deployment (`wait()`).

```python
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## Full Code

<details>
<summary>Click to get import and prompt</summary>

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

The effect is as follows:

![Painting Master](../assets/2_painting_master2.png)
