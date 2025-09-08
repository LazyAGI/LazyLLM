# TrainableModule Usage Guide

TrainableModule is the core module in LazyLLM, supporting training, deployment, and inference for all types of models (including LLM, Embedding, Rerank, multimodal models, etc.). This document will provide a detailed introduction to various usage methods of TrainableModule.

## Basic Usage

### Creating and Using TrainableModule

Basic creation method

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b')
```

!!! Note

    - In basic usage, if we haven't downloaded the model before, it will automatically download from `huggingface` or `modelscope`. If we don't want automatic model downloading, we can pass the parameter `trust_remote_code=False` to TrainableModule. This approach is usually very effective when we connect to LazyLLM inference services as a client.
    - Models are downloaded to the `~/.lazyllm/` directory by default. If we want to download to other directories, we can use the environment variable `LAZYLLM_MODEL_CACHE_DIR=/path/to` to specify the model download path.
    - Due to China's internet policies, the default model source is `modelscope`. If we want to change the model source, we can use the environment variable `LAZYLLM_MODEL_SOURCE=huggingface` to switch the model source to huggingface.
    - If we have locally downloaded model weights, we can specify with an absolute path: `model = lazyllm.TrainableModule('/path/to/qwen2-1.5b')`
    - If we download models to a unified location, we can configure the environment variable `LAZYLLM_MODEL_PATH=/path/to` to specify the model root directory.

Specify the directory for fine-tuned models

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='my_model')
```

We can use `target_path` to specify the location of fine-tuned models. In scenarios requiring fine-tuning, the fine-tuned model will be saved in `target_path`, and subsequent inference will load the model from `target_path` for inference. If you fine-tune based on a certain model, you can place the fine-tuned model in the `target_path` directory.

### Basic Inference

```python
# Start the model
model.start()

# Perform inference
response = model("hello")
print(response)
```

LazyLLM models still use the functor pattern. Local models need to call `start()` for deployment before they can be called.

## Fine-tuning

### Execute Fine-tuning

```python
import lazyllm

# Use automatic fine-tuning method
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='/path/to/model')
    .finetune_method(lazyllm.finetune.auto)
    .trainset('/path/to/training/data')
    .mode('finetune')

# Use specific fine-tuning method
model = lazyllm.TrainableModule('qwen2-1.5b')
    .finetune_method(lazyllm.finetune.llamafactory, learning_rate=1e-4, num_train_epochs=3)
    .trainset('/path/to/training/data')
    .mode('finetune')

# Execute fine-tuning
model.update()
```

For details on fine-tuning, please refer to the [Fine-tuning Tutorial](../Tutorial/9.md)

### Supported Fine-tuning Methods

- `lazyllm.finetune.auto`: Automatically select fine-tuning method
- `lazyllm.finetune.llamafactory`: Use LLaMA Factory for large model fine-tuning
- `lazyllm.finetune.collie`: Use Collie for fine-tuning. Note that Collie has stopped iteration and will be removed in future versions
- `lazyllm.finetune.flagembedding`: For Embedding model fine-tuning

### Fine-tuning Parameter Configuration

```python
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .finetune_method(lazyllm.finetune.llamafactory,
                     learning_rate=1e-4,      # Learning rate
                     num_train_epochs=3,      # Number of training epochs
                     per_device_train_batch_size=4,  # Batch size
                     max_samples=1000,        # Maximum number of samples
                     val_size=0.1)            # Validation set ratio
```

## Streaming Output

Model inference usually takes a long time. Typically, for a 32B model using A100 GPU for inference, facing a single session, it can generate about 30-35 new characters per second. If we want to generate a 2000-word article, it would take about 1 minute. For users, a 1-minute wait time is often unacceptable. The usual solution is to present the generation to users as it's being generated, which is what we call streaming output.

### Enable Streaming Output

```python
# Enable streaming output when creating
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
```

### Using Streaming Output

```python
import lazyllm
model = lazyllm.StreamCallHelper(model)
for msg in model('hello'):
    print(msg)
```

We use a `StreamCallHelper` to wrap our model, which allows us to iterate over the model's call results to achieve streaming output. When our model is in a flow, we need to wrap the outermost flow, not the model, for example:

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
ppl = lazyllm.pipeline(model)
ppl = lazyllm.StreamCallHelper(ppl)
for msg in ppl('hello'):
    print(msg)
```

### Streaming Output Configuration

We can configure streaming output to add prefixes or suffixes to the streaming output content and implement "colorful" streaming output. The specific configuration is as follows:

```python
# Configure streaming output style
stream_config = {
    'color': 'green',           # Output color
    'prefix': 'AI: ',          # Prefix
    'prefix_color': 'blue',    # Prefix color
    'suffix': 'End\n',            # Suffix
    'suffix_color': 'red'      # Suffix color
}

model = lazyllm.TrainableModule('qwen2-1.5b', stream=stream_config)
```

This way, our streaming output will start with blue "AI:", the streaming output content itself will be green, and end with red "End". If we have multiple large models in one task, this capability will be very useful. We can configure different prefixes and colors for each large model's output to present to users.

## Prompt Settings

### Basic Prompt Settings

```python
# Set simple text prompt
model = lazyllm.TrainableModule('qwen2-1.5b')
    .prompt("You are a helpful AI assistant, please answer questions in concise language.")

# Set conversation history
history = [
    ["User", "Hello"],
    ["Assistant", "Hello! How can I help you?"]
]
model = model.prompt("Continue the conversation", history=history)
```

The conversation history we set is the "system prompt", which takes effect for each user/session in a multi-user/multi-session environment. We can refer to the [Prompt Tutorial](prompt.md) for more information.

### Using Dictionary Format Prompts

```python
# Use dictionary format to set more complex prompts
prompt_config = {
    'system': 'You are a professional {system_role}',
    'user': '{user_input}',
}

model = lazyllm.TrainableModule('qwen2-1.5b').prompt(prompt_config)
model(dict(system_role='Programming Assistant', user_input='Program according to user needs'))
```

### Clear Prompt

```python
# Clear prompt, use empty prompt
model = model.prompt(None)
```

## Output Formatting

### Using Built-in Formatters

```python
# Use JSON formatter
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .formatter(lazyllm.formatter.JsonFormatter('[:][a]'))
```

You can extract JSON from the output through JsonFormatter and get specified elements.

### Custom Formatters

```python
# Use custom function as formatter
def my_formatter(text):
    return f"Processed result: {text.strip()}"

model = lazyllm.TrainableModule('qwen2-1.5b').formatter(my_formatter)

# Use chained formatters
model = model.formatter(lazyllm.formatter.JsonFormatter() | lazyllm.formatter.StrFormatter())
```

## Model Sharing

### Basic Sharing

Although LazyLLM can conveniently deploy models, in actual use, if each module deploys a model separately, it will waste resources. Therefore, we introduced a model sharing mechanism.

```python
# Create base model
base_model = lazyllm.TrainableModule('qwen2-1.5b').start()

# Create shared instances, using the same model but different prompts
chat_model = base_model.share(prompt="You are a chatbot")
code_model = base_model.share(prompt="You are a programming assistant")
```

### Shared Parameter Configuration

```python
# Specify different configurations when sharing
shared_model = base_model.share(
    prompt="New prompt",                    # New prompt
    format=lazyllm.formatter.JsonFormatter(), # New formatter
    stream={'color': 'blue'}                 # New streaming configuration
)
```

### Advantages of Sharing

- **Resource Saving**: Multiple instances share the same model deployment, saving GPU memory
- **Flexible Configuration**: Each shared instance can have different prompts and formatters
- **Performance Optimization**: Avoid repeated model loading

## Connect from URL

### Connect to Existing Services

```python
# Connect to existing HTTP service
model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm, 
).start()

# Connect to the above deployed model, assuming the above URL is http://localhost:8000/generate
remote_model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm,
    url='http://localhost:8000/generate/'
)

# Use remote model for inference
response = remote_model("Hello")
```

Through this method, we can first start inference services, and then multiple different users can use these inference services in multiple different processes.

## Embedding and Rerank Models

### Embedding Models

```python
# Create Embedding model
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5')

# Perform text embedding
embeddings = embedding_model("This is a test text")
print(embeddings)  # Returns vector list

# Fine-tune Embedding model
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5')\
    .finetune_method(lazyllm.finetune.flagembedding)\
    .trainset('/path/to/embedding_data')\
    .mode('finetune')
embedding_model.update()
```

For models not in the LazyLLM official model list, we need to explicitly specify the model type

```python
# Create Embedding model
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5', type='embed')
```

### Rerank Models

```python
# Create Rerank model
rerank_model = lazyllm.TrainableModule('bge-reranker-large')

# Perform reranking
query = "User query"
documents = ["Document 1", "Document 2", "Document 3", "Document 4"]
top_n = 2

results = rerank_model(query, documents=documents, top_n=top_n)
# Returns results in [(index, score), ...] format
print(results)
```

For models not in the LazyLLM official model list, we need to explicitly specify the model type

```python
# Create Embedding model
embedding_model = lazyllm.TrainableModule('bge-reranker-large', type='rerank')
```

Usually, we don't use Rerank models alone, but use them in RAG. For using Rerank models in RAG, please refer to [RAG Best Practices](rag.md)

## Multimodal Models

### Image Generation (SD)

```python
# Create Stable Diffusion model
sd_model = lazyllm.TrainableModule('stable-diffusion-3-medium')

# Generate image
image_prompt = "a beautiful landscape with mountains and lakes"
image_result = sd_model(image_prompt)
# Returns generated image path or data
```

### Speech to Text (STT)

```python
# Create speech recognition model
stt_model = lazyllm.TrainableModule('SenseVoiceSmall')

# Perform speech recognition
audio_file = '/path/to/audio.wav'
text_result = stt_model(audio_file)
print(text_result)
```

### Text to Speech (TTS)

```python
# Create speech synthesis model
tts_model = lazyllm.TrainableModule('ChatTTS')

# Perform speech synthesis
text = "Hello, this is a test"
audio_result = tts_model(text)
# Returns generated audio file path
```

### Vision Language Model (VLM)

```python
# Create vision language model
vlm_model = lazyllm.TrainableModule('internvl-chat-2b-v1-5')\
    .deploy_method(lazyllm.deploy.LMDeploy)

# Perform image Q&A
image_path = '/path/to/image.jpg'
question = "What's in this image?"
response = vlm_model(encode_query_with_filepaths(question, image_path))
print(response)
```

## OpenAI Format Deployment

### Start OpenAI Service Based on vLLM

```python
# Use vLLM to deploy OpenAI format service
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .deploy_method(lazyllm.deploy.vllm, 
                   openai_api=True,  # Enable OpenAI API format
                   port=8000)

# Start service
model.start()

# Service will provide OpenAI-compatible API at http://localhost:8000/v1/
```

### Connect to OpenAI Format Service

```python
# Connect to OpenAI format service
openai_model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm,
    url='http://localhost:8000/v1/'
)

# Use OpenAI format for inference
response = openai_model("Hello")
```

!!! Note

    If the URL ends with `v1/` or `v1/chat/completions`, it will be considered an OpenAI format URL; if it ends with `generate`, it will be considered a VLLM format URL.
