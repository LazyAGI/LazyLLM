# multi_modal_output_agent

This project demonstrates how to build a multi-modal intelligent Agent using [LazyLLM](https://github.com/LazyAGI/LazyLLM).

## What you will learn from this tutorial:

- How to call the Stable Diffusion model to generate images.
- How to wrap and register a function as a tool;
- How to launch a client service using `WebModule` with just a few lines of code.

## Code Implementation

### Project Dependencies

Make sure you have installed the following dependencies:

```bash
pip install lazyllm
```

### Step-by-step Guide

#### Step 1: Initialize LLM and Image Generation Model

```python
llm = TrainableModule("internlm2-chat-7b").start()
sd3 = TrainableModule("stable-diffusion-3-medium").start()
```

#### Step 2: Define the Prompt Task

Convert Chinese content into English drawing prompts

```python
def convert_to_prompt(content: str) -> str:
    '''
    Convert Chinese description into detailed English prompt words for image generation.

    Args:
        content (str): Chinese text describing the image to be generated.
    '''
    system_prompt = (
        "You are a drawing prompt word master who can convert any Chinese content "
        "entered by the user into English drawing prompt words. "
        "In this task, you need to convert any input content into English drawing prompt words, "
        "and you can enrich and expand the prompt word content."
    )
    full_prompt = f"{system_prompt}\n\nInput:\n{content}"
    llm1 = llm.share()
    return llm1(full_prompt)
```

Use **Stable Diffusion** to generate images

```python
def generate_image(prompt: str) -> str:
    '''
    Generate an image based on the given English prompt and style.

    Args:
        prompt (str): The English prompt describing the image.
    '''
    _sd3 = sd3.share()
    return _sd3(prompt)
```


#### Step 3: Register Tool Functions

Register multiple tool functions using ``@fc_register("tool")``:

```python
@fc_register("tool")
def convert_to_prompt(content: str) -> str:
    '''
    Convert Chinese description into detailed English prompt words for image generation.

    Args:
        content (str): Chinese text describing the image to be generated.
    '''
    system_prompt = (
        "You are a drawing prompt word master who can convert any Chinese content "
        "entered by the user into English drawing prompt words. "
        "In this task, you need to convert any input content into English drawing prompt words, "
        "and you can enrich and expand the prompt word content."
    )
    full_prompt = f"{system_prompt}\n\nInput:\n{content}"
    llm1 = llm.share()
    return llm1(full_prompt)
```

```python
@fc_register("tool")
def generate_image(prompt: str) -> str:
    '''
    Generate an image based on the given English prompt and style.

    Args:
        prompt (str): The English prompt describing the image.
    '''
    _sd3 = sd3.share()
    return _sd3(prompt)
```

#### Step 4: Use the Agent
Let the LLM call the tools we defined:

```python
agent = FunctionCallAgent(llm.share(), tools=["convert_to_prompt", "generate_image"])
```

### Example Output

Example input:

```python
query = "A parrot playing soccer"
```

Is it possible to achieve the above functionality in a simpler way?

Yes! You can use [Pipeline](https://docs.lazyllm.ai/zh-cn/latest/API%20Reference/flow/#lazyllm.flow.Pipeline) control flow to assemble the application with just a few lines of code on the client side:

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```

Example:
![API Agent Demo2](../../assets/agent_sd_demo2.png)