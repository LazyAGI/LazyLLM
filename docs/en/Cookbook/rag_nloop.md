# Build a Question Answering application with chat history

A key challenge in Q&A systems is handling conversational context. This tutorial shows how to add memory to a RAG application so it can handle follow-up questions that depend on chat history.

We'll use the LazyLLM framework to build a conversational RAG system that can handle multi-turn conversations effectively.

## Setup

### Components

We'll build our conversational RAG application using LazyLLM's built-in components:

- **Document**: For loading and managing documents with embeddings
- **Retriever**: For retrieving relevant documents based on similarity
- **Reranker**: For reranking retrieved documents for better relevance
- **OnlineChatModule**: For LLM-based question reformulation and answer generation
- **ReactAgent**: For creating an agent that can iterate over multiple retrieval steps

### Dependencies

```python
import os
import tempfile
from typing import List, Dict, Any
import lazyllm
from lazyllm import pipeline, parallel, bind, fc_register
from lazyllm import Document, Retriever, Reranker, SentenceSplitter, OnlineEmbeddingModule
```

## Chains

A key challenge for conversational RAG is that the user messages and the LLM's responses aren't the only forms of context. User messages can reference previous portions of the conversation, making them challenging to understand in isolation.

For example, the second question below depends on context from the first:

```
Human: What is the standard method for task decomposition?
AI: The standard method for task decomposition is Chain of Thought (CoT) prompting...
Human: What are the common extensions of this method?
```

The second question "What are the common extensions of this method?" is ambiguous without the context of the first question. Our system needs to understand that "this method" refers to "Chain of Thought prompting".

### Stateful management of chat history

To handle this, we need to contextualize the question based on the chat history. Let's build our conversational RAG system:

```python
class ConversationalRAGPipeline:
    """Multi-turn conversational RAG pipeline based on LazyLLM components"""
    
    def __init__(self, docs_path: str = None):
        self.chat_history = []
        self.docs_path = docs_path or self._create_sample_docs()
        self.pipeline = None
        self.init_pipeline()
```

First, we set up our documents and embeddings:

```python
def init_pipeline(self):
    """Initialize RAG pipeline"""
    # Create documents and embeddings
    documents = Document(
        dataset_path=self.docs_path, 
        embed=OnlineEmbeddingModule(
            source="qwen",
            type="embed",
            embed_model_name="text-embedding-v1"
        ),
        manager=False
    )
    documents.create_node_group(
        name="sentences", 
        transform=SentenceSplitter, 
        chunk_size=512, 
        chunk_overlap=100
    )
```

Next, we create a contextualization prompt to reformulate questions based on chat history:

```python
contextualize_prompt = """You are an assistant that helps rewrite questions. Your task is to:
1. Analyze the user's chat history
2. Understand the latest question
3. If the question depends on historical context, rewrite it as a standalone understandable form
4. If the question is already standalone, return the original question
5. Only return the rewritten question, do not add any explanations

Example:
History: "User: Tell me about John\nAssistant: John is an engineer..."
Question: "What is his job?" 
Rewrite as: "What is John's job?"

History: {chat_history}
Question: {question}

Rewritten question:"""

self.contextualizer = lazyllm.OnlineChatModule(
    source="deepseek",
    timeout=30
).prompt(lazyllm.ChatPrompter(contextualize_prompt))
```

Then we build our main RAG pipeline with parallel retrieval and reranking:

```python
with pipeline() as ppl:
    # Parallel retrieval
    with parallel().sum as ppl.prl:
        ppl.prl.retriever1 = Retriever(
            documents, 
            group_name="sentences", 
            similarity="cosine", 
            topk=3
        )
        ppl.prl.retriever2 = Retriever(
            documents, 
            "CoarseChunk", 
            "bm25_chinese", 
            0.003, 
            topk=2
        )
    
    # Reranking
    ppl.reranker = Reranker(
        "ModuleReranker", 
        model=OnlineEmbeddingModule(type="rerank", source="qwen"), 
        topk=3, 
        output_format='content', 
        join=True
    ) | bind(query=ppl.input)
    
    # LLM generates answer
    ppl.llm = lazyllm.OnlineChatModule(
        source="deepseek",
        stream=False,
        timeout=60
    ).prompt(lazyllm.ChatPrompter(rag_prompt, extra_keys=["context_str", "chat_history"]))

self.pipeline = ppl
```

The chat method handles the conversation flow:

```python
def chat(self, question: str) -> str:
    """Process single turn conversation"""
    # 1. Reformulate question
    contextualized_question = self.contextualize_question(question)
    
    # 2. Build historical conversation text
    history_text = self._format_chat_history(self.chat_history, max_turns=4)
    
    # 3. Generate answer through RAG pipeline
    response = self.pipeline(
        contextualized_question,
        chat_history=history_text
    )
    
    # 4. Update chat history
    self.chat_history.append({"role": "user", "content": question})
    self.chat_history.append({"role": "assistant", "content": response})
    
    return response
```

Let's test our conversational RAG:

```python
rag = ConversationalRAGPipeline()

# First question
response1 = rag.chat("What is the standard method for task decomposition?")
print(f"Response 1: {response1}")

# Follow-up question that depends on context
response2 = rag.chat("What are the common extensions of this method?")
print(f"Response 2: {response2}")
```

Note that the question generated by the model in the second question incorporates the conversational context, transforming "this method" into "Chain of Thought prompting extensions".

## Agents

Agents leverage the reasoning capabilities of LLMs to make decisions during execution. Using agents allows you to offload additional discretion over the retrieval process. Although their behavior is less predictable than the above "chain", they are able to execute multiple retrieval steps in service of a query, or iterate on a single search.

Below we assemble a minimal RAG agent using LazyLLM's ReactAgent:

```python
@fc_register("tool")
def conversational_rag_chat(question: str) -> str:
    """
    Multi-turn conversational RAG tool
    
    Args:
        question (str): User question
    
    Returns:
        str: RAG system's answer
    """
    global rag_system
    if rag_system is None:
        rag_system = ConversationalRAGPipeline()
    
    return rag_system.chat(question)

def create_rag_agent():
    """Create RAG agent"""
    agent = lazyllm.ReactAgent(
        llm=lazyllm.OnlineChatModule(source="deepseek", timeout=60),
        tools=["conversational_rag_chat"],
        max_retries=3,
        return_trace=False,
        stream=False
    )
    
    return agent
```

The key difference from our earlier implementation is that instead of a final generation step that ends the run, here the tool invocation can loop back to gather more information. The agent can then either answer the question using the retrieved context, or generate another tool call to obtain more information.

Let's test this out with a question that would typically require an iterative sequence of retrieval steps:

```python
agent = create_rag_agent()

response = agent(
    "What is the standard method for Task Decomposition? "
    "Once you get the answer, look up common extensions of that method."
)
print(response)
```

Note that the agent:
1. Generates a query to search for a standard method for task decomposition
2. Receiving the answer, generates a second query to search for common extensions of it
3. Having received all necessary context, answers the question

## Running the Application

To start the web interface:

```bash
python agent_n_rag.py
```

Then visit `http://localhost:8849` to interact with the conversational RAG system through a web interface.

## Next steps

We've covered the steps to build a basic conversational Q&A application:

* We used chains to build a predictable application that contextualizes questions based on chat history
* We used agents to build an application that can iterate on a sequence of queries

To explore different types of retrievers and retrieval strategies, visit LazyLLM's documentation on retrieval components.

For more advanced agent architectures, check out LazyLLM's agent documentation and examples.
