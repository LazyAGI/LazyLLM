# Build an Intelligent Search Agent with LazyLLM

LazyLLM supports the creation of agents that use language models as reasoning engines to determine which actions to take and how to interact with external tools. These agents can analyze user queries, decide when to use specific tools, and process the results to provide comprehensive responses. This tool-calling capability is fundamental to building intelligent agents that can interact with the real world.

!!! abstract "In this section, you will learn the following key points of LazyLLM"

    - How to create and register custom tools for agents using @fc_register.
    - How to set up a language model as the reasoning engine for agents using [OnlineChatModule][lazyllm.module.OnlineChatModule].
    - How to configure [ReactAgent][lazyllm.tools.agent.ReactAgent] to intelligently decide when to call tools.
    - How to deploy agents with a web interface for interaction using [WebModule][lazyllm.tools.webpages.WebModule].

Let's get started!

## Setup

### Installation

First, install LazyLLM:

```bash
pip install lazyllm
```

### Environment Variables

You'll need to set up your API keys:

```bash
export BOCHA_API_KEY=your_bocha_api_key
export LAZYLLM_DEEPSEEK_API_KEY=your_deepseek_api_key
```
#### Applying for API Keys

Before using this tutorial, you need to apply for the corresponding API keys:

**Bocha API Key Application:**

1. Visit the [Bocha Open Platform](https://open.bochaai.com/overview)
2. Register and log in to your account
3. Create a new API key in the "API KEY Management" page
4. Copy the generated API key and set it in your environment variables

> ❗ Note: To obtain the **free API quota** for Bocha API, please refer to the [Developer Documentation](https://bocha-ai.feishu.cn/wiki/RWdvw557Li3IJekGeLkcDFa3n1f).  
> On the [Bocha API homepage](https://open.bochaai.com/overview), go to **Resource Package Management** and subscribe to the **Free Trial**.

**DeepSeek API Key Application:**

1. Visit the [DeepSeek Platform](https://platform.deepseek.com/)
2. Register and log in to your account
3. Create a new API key in the "API Keys" page
4. Copy the generated API key and set it in your environment variables

### Dependencies

Make sure you have the required dependencies:

```bash
pip install httpx
```

## Define Tools

First, let's define a search tool that the agent can use. In LazyLLM, we use the `@fc_register("tool")` decorator to register functions as tools:

```python
import os
import httpx
import lazyllm
from lazyllm.tools import fc_register

@fc_register("tool")
def bocha_search(query: str) -> str:
    """
    Query information using Bocha Search API
    
    Args:
        query (str): Search query, e.g.: "LazyLLM framework", "Latest AI developments"
    
    Returns:
        str: Search result summary
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv('BOCHA_API_KEY')
        if not api_key:
            return "Error: BOCHA_API_KEY environment variable not set"
        
        # Send request to Bocha API
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": 10
        }
        
        with httpx.Client(timeout=30) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.text
            return f"Search failed: {response.status_code}"
            
    except Exception as e:
        return f"Search error: {str(e)}"
```

The `@fc_register("tool")` decorator automatically makes this function available to LazyLLM agents. The function docstring is important as it helps the agent understand when and how to use the tool.

## Using Language Models

LazyLLM provides easy access to various online language models through `OnlineChatModule`:

```python
# Create LLM
llm = lazyllm.OnlineChatModule(
    source="deepseek",
    timeout=30
)
```

You can also use other providers:

```python
# OpenAI
llm = lazyllm.OnlineChatModule(source="openai")

# Anthropic Claude
llm = lazyllm.OnlineChatModule(source="claude")

# Local models
llm = lazyllm.OnlineChatModule(source="ollama")
```

## Create the Agent

Now let's create a ReactAgent that can use our search tool:

```python
def create_agent():
    """Create a configured search agent"""
    # Create LLM
    llm = lazyllm.OnlineChatModule(
        source="deepseek",
        timeout=30
    )
    
    # Create ReactAgent
    agent = lazyllm.tools.agent.ReactAgent(
        llm=llm,
        tools=["bocha_search"],  # Reference our registered tool
        max_retries=2,
        return_trace=False,
        stream=False
    )
    
    return agent
```

The `ReactAgent` follows the ReAct (Reasoning and Acting) paradigm, which allows the agent to:

- **Think** about what it needs to do
- **Act** by calling tools when needed
- **Observe** the results and continue reasoning

## Run the Agent

Let's test our agent with a simple query:

```python
# Create agent
agent = create_agent()

# Run query
result = agent("Search for the latest information about LazyLLM framework")
print(result)
```

The agent will:

1. Analyze the query
2. Decide to use the `bocha_search` tool
3. Call the tool with the appropriate search terms
4. Process the results and provide a comprehensive answer

## Web Interface

LazyLLM makes it easy to deploy your agent with a web interface:

![Web Interface Demo](../assets/agent-tooluse.png)

```python
def start_web_interface():
    """Start web interface"""
    print("Starting Bocha Search Tool Web Interface...")
    
    try:
        # Check API key
        if not os.getenv('BOCHA_API_KEY'):
            print("Warning: BOCHA_API_KEY environment variable not set")
            print("Please set: export BOCHA_API_KEY=your_api_key")
            return
            
        # Create agent
        agent = create_agent()
        
        # Start web interface
        web_module = lazyllm.WebModule(
            agent,
            port=8848,
            title="Bocha Search Agent"
        )
        
        print(f"Web interface started: http://localhost:8848")
        print("Press Ctrl+C to stop the service")
        
        web_module.start().wait()
        
    except KeyboardInterrupt:
        print("\nStopping service...")
    except Exception as e:
        print(f"Start failed: {e}")
```

## Streaming Responses

To enable streaming responses, modify the agent configuration:

![Streaming Response Demo](../assets/agent-tooluse-stream.png)

```python
agent = lazyllm.tools.agent.ReactAgent(
    llm=llm,
    tools=["bocha_search"],
    max_retries=2,
    return_trace=False,
    stream=True  # Enable streaming output
)
```

## Adding Memory

LazyLLM agents can maintain conversation history through the web interface automatically, or you can implement custom memory functionality to enhance the agent's ability to maintain context across conversations.

Memory features enable agents to:

- Remember previous conversations and user preferences
- Maintain context across multiple interactions
- Provide more personalized responses based on conversation history
- Build up knowledge about ongoing projects or topics

*Custom memory implementation will be added in future updates.*

## Complete Example

Here's the complete working example:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search Tool Agent using LazyLLM Framework
Web interface supported with Bocha Search API
"""

import os
import httpx
import lazyllm
from lazyllm.tools import fc_register


@fc_register("tool")
def bocha_search(query: str) -> str:
    """
    Query information using Bocha Search API
    
    Args:
        query (str): Search query, e.g.: "LazyLLM framework", "Latest AI developments"
    
    Returns:
        str: Search result summary
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv('BOCHA_API_KEY')
        if not api_key:
            return "Error: BOCHA_API_KEY environment variable not set"
        
        # Send request to Bocha API
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": 10
        }
        
        with httpx.Client(timeout=30) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.text
            return f"Search failed: {response.status_code}"
            
    except Exception as e:
        return f"Search error: {str(e)}"


def create_agent():
    """Create a configured search agent"""
    # Create LLM
    llm = lazyllm.OnlineChatModule(
        source="deepseek",
        timeout=30)
    
    # Create ReactAgent
    agent = lazyllm.tools.agent.ReactAgent(
        llm=llm,
        tools=["bocha_search"],
        max_retries=2,
        return_trace=False,
        stream=False
    )
    
    return agent


def start_web_interface():
    """Start web interface"""
    print("Starting Bocha Search Tool Web Interface...")
    
    try:
        # Check API key
        if not os.getenv('BOCHA_API_KEY'):
            print("Warning: BOCHA_API_KEY environment variable not set")
            print("Please set: export BOCHA_API_KEY=your_api_key")
            return
            
        # Create agent
        agent = create_agent()
        
        # Start web interface
        web_module = lazyllm.WebModule(
            agent,
            port=8848,
            title="Bocha Search Agent"
        )
        
        print(f"Web interface started: http://localhost:8848")
        print("Press Ctrl+C to stop the service")
        
        web_module.start().wait()
        
    except KeyboardInterrupt:
        print("\nStopping service...")
    except Exception as e:
        print(f"Start failed: {e}")


if __name__ == "__main__":
    start_web_interface()
```

## Usage Examples

### Command Line Usage

```python
# Create and use agent
agent = create_agent()

# Search for technical information
result = agent("Search for Python async programming best practices")
print(result)

# Search for news
result = agent("Find latest breakthroughs in AI field")
print(result)
```

### Web Interface Usage

1. Run the script: `python tool_agent.py`
2. Open your browser to `http://localhost:8848`
3. Start chatting with the agent through the web interface

## Conclusion

LazyLLM provides a powerful and easy-to-use framework for building intelligent agents. With just a few lines of code, you can:

- Register custom tools
- Create sophisticated agents
- Deploy web interfaces
- Handle streaming and memory

This framework makes it accessible to build production-ready AI agents that can interact with external APIs and provide intelligent responses to user queries.For more information about LazyLLM, check out the [official documentation](docs.lazyllm.ai/) and examples.
