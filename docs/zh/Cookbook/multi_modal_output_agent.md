# 多模态输出Agent：图像与文本

本项目展示了如何使用 [LazyLLM](https://github.com/LazyAGI/LazyLLM) 构建一个使用不直接生成文本的工具来构建**多模态智能体**。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何结合 [TrainableModule][lazyllm.module.TrainableModule] 调用不同模态模型。
    - 如何使用 [ReactAgent][lazyllm.tools.agent.ReactAgent] 自动选择和调用工具来完成复杂任务。

## 代码实现

### 项目依赖

确保你已安装以下依赖：

```bash
pip install lazyllm
```

### 功能简介

* 对用户输入进行翻译并优化，调用sd3模型生成相应图片
* 调用匹配的工具由 `ReactAgent` 进行自动执行。

### 步骤详解

#### Step 1: 初始化图像生成模型
全局缓存 SD3 模型（避免重复加载）
```python

_sd3_model = None
```

#### Step 2: 优化用户问题，构建提示词
```python
@fc_register("tool")
def optimize_prompt(query: str) -> str:
    """
    Translate input to English if needed, then optimize it into a high-quality,
    detailed English prompt specifically for Stable Diffusion 3 Medium (SD3-Medium).
    
    Args:
        query (str): User's input 
    """
    system_prompt = (
        "You are an expert prompt engineer for Stable Diffusion 3 Medium (SD3-Medium). "
        "The user may give input in any language. First, translate it accurately into English if needed. "
        "Then, rewrite it as a single, fluent, natural-language English sentence or paragraph that describes the desired image in rich detail. "
        "Include: main subject, artistic style, lighting, composition, color palette, mood, and key visual elements. "
        "DO NOT use comma-separated tags, keywords, or phrases like 'masterpiece', '4k', 'best quality'. "
        "DO NOT add any prefixes, explanations, or markdown. Output ONLY the final prompt as plain text."
    )
    
    full_prompt = f"{system_prompt}\n\nUser input: {query}"
    llm1 = lazyllm.OnlineChatModule()
    result = llm1(full_prompt)
    return result
```

#### Step 3: 使用 Stable Diffusion 生成图像
```python
@fc_register("tool")
def generate_image(context: str) -> str:
    """
    Generate an image using Stable Diffusion 3 Medium based on an English prompt.
    Args:
        context (str): A detailed, natural-language English prompt optimized for SD3-Medium, describing the image to generate.
    """
    global _sd3_model
    if _sd3_model is None:
        _sd3_model = lazyllm.TrainableModule("stable-diffusion-3-medium").start()
    return _sd3_model(context)
```
#### Step 4: 使用Agent
显式注册工具，并使用`ReactAgent`构建智能体
```python
llm = lazyllm.TrainableModule("Qwen3-32B").start()
tools = ["optimize_prompt",  "generate_image"]
agent = lazyllm.ReactAgent(llm=llm, tools=tools)
print(agent("画个猫"))
```
### 完整代码
```python
import lazyllm
from lazyllm import fc_register
# 全局缓存 SD3 模型（避免重复加载）
_sd3_model = None

@fc_register("tool")
def generate_image(context: str) -> str:
    """
    Generate an image using Stable Diffusion 3 Medium based on an English prompt.
    Args:
        context (str): A detailed, natural-language English prompt optimized for SD3-Medium, describing the image to generate.
    """
    global _sd3_model
    if _sd3_model is None:
        _sd3_model = lazyllm.TrainableModule("stable-diffusion-3-medium").start()
    return _sd3_model(context)
@fc_register("tool")
def optimize_prompt(query: str) -> str:
    """
    Translate input to English if needed, then optimize it into a high-quality,
    detailed English prompt specifically for Stable Diffusion 3 Medium (SD3-Medium).
    
    Args:
        query (str): User's input 
    """
    system_prompt = (
        "You are an expert prompt engineer for Stable Diffusion 3 Medium (SD3-Medium). "
        "The user may give input in any language. First, translate it accurately into English if needed. "
        "Then, rewrite it as a single, fluent, natural-language English sentence or paragraph that describes the desired image in rich detail. "
        "Include: main subject, artistic style, lighting, composition, color palette, mood, and key visual elements. "
        "DO NOT use comma-separated tags, keywords, or phrases like 'masterpiece', '4k', 'best quality'. "
        "DO NOT add any prefixes, explanations, or markdown. Output ONLY the final prompt as plain text."
    )
    # 使用 OnlineChatModule 并强制返回纯文本
    full_prompt = f"{system_prompt}\n\nUser input: {query}"
    llm1 = lazyllm.OnlineChatModule()
    result = llm1(full_prompt)
    print(result)
    return result

llm = lazyllm.TrainableModule("Qwen3-32B").start()
tools = ["optimize_prompt",  "generate_image"]
agent = lazyllm.ReactAgent(llm=llm, tools=tools)
print(agent("画个猫"))
```

同时我们还可以使用`WebModule`来启动web页面
```python
web_module = lazyllm.WebModule(
            agent,
            port=9849
        )
web_module.start().wait()
```
### 示例运行结果
```python
query = "画个猫"
agent(query)
```
<think>
Okay, the user asked to draw a cat. I first used the optimize_prompt tool to translate and enhance the prompt into English. Then, I called generate_image with the optimized prompt. The system returned a base64 encoded image which I saved as a file. Now, I need to present the image to the user. Since the user's original query was in Chinese, I should respond in Chinese. I'll mention that the image has been generated and provide the file path. Let me check if there's any additional step needed, but since the task is complete, I'll just confirm the output.
</think>

Answer: 已根据您的要求生成了一张猫的插画，图片文件路径为：.../xxx.png
![Painting](/home/mnt/zhangkejun/LazyLLM/.temp/base64_to_file_q68njx29.png)
## 结语

该项目展示了如何使用 LazyLLM 构建一个**多模态输出的 Agent**。欢迎继续扩展你的工具集，构建强大的 AI 助手。