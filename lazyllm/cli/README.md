# LazyLLM Deploy

LazyLLM Deploy is a command-line tool for deploying large language models.

## Installation

```bash
pip install lazyllm
```

## Usage


```bash
lazyllm deploy {model_name} [options]
```

## Examples

```bash
# Start a basic service
lazyllm deploy llama2 --port=8000

# Use tensor parallelism
lazyllm deploy llama2 --tp=2
```

## VLLM Parameter Restriction Policy

For VLLM deployments, we implement a parameter restriction strategy that limits the use of parameters to only those that are explicitly supported. To bypass parameter restrictions and utilize all VLLM-supported parameters, set the following environment variable:

```bash
export LAZYLLM_VLLM_SKIP_CHECK_KW=True
```

### Parameter Restriction Overview

- **Default Behavior**: Only validated and safe parameters are permitted, ensuring deployment stability and compatibility
- **Bypass Restrictions**: After setting the environment variable, all VLLM-supported parameters become available
- **Important Considerations**: When restrictions are bypassed, users are responsible for ensuring parameter correctness

### Usage Examples

```bash
# Using default parameter restrictions
lazyllm deploy llama2 --framework=vllm --tp=2 --max_model_len=4096

# Bypassing parameter restrictions to use all VLLM parameters
export LAZYLLM_VLLM_SKIP_CHECK_KW=True
lazyllm deploy llama2 --framework=vllm --tp=2 --custom_vllm_param=value
```

## Notes

- Ensure all required dependencies are installed before deployment
- Adjust parameter configurations based on available hardware resources
