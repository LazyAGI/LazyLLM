# 支持的模型

## 常见模型

以下模型是 LazyLLM 中常见支持的模型。您可以用这里提供的名字直接作为 [TrainableModule][lazyllm.module.TrainableModule] 的参数传入，LazyLLM 会自动下载该模型。如果网络不可访问，您也可以直接指定本地模型的绝对路径。

### 大语言模型

- Baichuan-13B-Chat
- Baichuan-7B
- Baichuan2-13B-Chat
- Baichuan2-13B-Chat-4bits
- Baichuan2-7B-Chat
- Baichuan2-7B-Chat-4bits
- Baichuan2-7B-Intermediate-Checkpoints
- chatglm3-6b
- chatglm3-6b-128k
- chatglm3-6b-32k
- CodeLlama-13b-hf
- CodeLlama-34b-hf
- CodeLlama-70b-hf
- CodeLlama-7b-hf
- glm-4-9b-chat
- internlm-20b
- internlm-7b
- internlm-chat-20b
- internlm-chat-20b-4bit
- internlm-chat-7b
- internlm2-1_8b
- internlm2-20b
- internlm2-7b
- internlm2-chat-1_8b
- internlm2-chat-1_8b-sft
- internlm2-chat-20b
- internlm2-chat-20b-4bits
- internlm2-chat-20b-sft
- internlm2-chat-7b
- internlm2-chat-7b-4bits
- internlm2-chat-7b-sft
- internlm2_5-7b-chat
- internlm2-math-20b
- internlm2-math-7b
- Llama-2-13b
- Llama-2-70b
- Llama-2-7b
- Meta-Llama-3-70B
- Meta-Llama-3-8B
- Qwen-14B
- Qwen-1_8B
- Qwen-72B
- Qwen-7B
- Qwen1.5-0.5B-Chat
- Qwen1.5-1.8B
- Qwen1.5-14B
- Qwen1.5-14B-Chat
- Qwen1.5-4B
- Qwen1.5-72B
- Qwen1.5-7B
- Qwen2-72B-Instruct
- Qwen2-72B-Instruct-AWQ
- Qwen2-7B-Instruct

### 多模态模型

- InternVL-Chat-V1-5
- llava-1.5-13b-hf
- llava-1.5-7b-hf
- Mini-InternVL-Chat-2B-V1-5
- bge-large-zh-v1.5
- bge-reranker-large
- bark
- ChatTTS
- musicgen-medium
- musicgen-stereo-small
- stable-diffusion-3-medium
- SenseVoiceSmall

## 其它模型

对于没在上面列表中的 LLM 模型，您也可以尝试使用由 huggingface 或者 modelscope 所提供的模型二级全名，如：`Shanghai_AI_Laboratory/internlm2-chat-1_8b`，将其作为 [TrainableModule][lazyllm.module.TrainableModule] 的参数传入，LazyLLM 也会自动下载该模型。
