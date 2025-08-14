# LazyLLM 10Q Analysis System Tutorial

This tutorial demonstrates how to use the LazyLLM framework to build a complex 10Q financial report analysis system that answers complex financial queries through sub-question decomposition techniques.

## Overview

This system can achieve:

- Decompose complex questions into multiple simple sub-questions
- Query different data sources
- Synthesize answers from multiple sub-questions to form complete responses
- Provide professional financial analysis

## System Architecture

### Core Components

1. **SubQuestionGenerator**: Sub-question generator
2. **SubQuestionQueryEngine**: Sub-question query engine
3. **QueryEngineTool**: Query engine tool wrapper
4. **RAG Pipeline**: Retrieval-Augmented Generation pipeline

### Workflow

```
Complex Question → Sub-question Decomposition → Multi-source Query → Answer Synthesis → Final Response
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install lazyllm
```

### 2. Prepare Data

Download Uber's 10Q report files to the `data/10q/` directory:

```bash
mkdir -p data/10q/
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'
```

## Core Code Analysis

### 1. Sub-Question Generator (SubQuestionGenerator)

```python
class SubQuestionGenerator(ModuleBase):
    """Sub-question generator"""
    
    def __init__(self, llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.llm = llm
        
        # Sub-question generation prompt template
        self.sub_question_prompt = """You are a professional financial analyst. Given a complex question and available data sources, you need to decompose the complex question into multiple simple sub-questions.

Available data sources:
{data_sources}

Complex question: {query}

Please decompose this question into 2-4 sub-questions. Each sub-question should:
1. Target a specific data source
2. Be simple and clear, easy to answer
3. Help answer the original question

Please return in JSON format as follows:
{{
    "sub_questions": [
        {{
            "question": "Sub-question 1",
            "data_source": "Corresponding data source name",
            "reasoning": "Why this sub-question is needed"
        }},
        {{
            "question": "Sub-question 2", 
            "data_source": "Corresponding data source name",
            "reasoning": "Why this sub-question is needed"
        }}
    ]
}}

Return only JSON format, no other content."""
```

**Function Description**:
- Receives complex questions and available data sources
- Uses LLM to decompose complex questions into 2-4 simple sub-questions
- Each sub-question is associated with a specific data source
- Returns structured JSON format

### 2. Sub-Question Query Engine (SubQuestionQueryEngine)

```python
class SubQuestionQueryEngine(ModuleBase):
    """Sub-question query engine"""
    
    def __init__(self, query_engine: Dict[str, Any], llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.query_engine = query_engine
        self.llm = llm
        self.sub_question_generator = SubQuestionGenerator(llm)
        
        self.synthesis_prompt = """You are a professional financial analyst. Based on the following sub-question answers, please synthesize a complete response to answer the original question.

Original question: {original_query}

Sub-questions and their answers:
{sub_answers}

Please provide a comprehensive, accurate, and logically clear response that integrates all relevant information. The response should:
1. Directly answer the original question
2. Be based on the provided sub-question answers
3. Maintain objectivity and professionalism
4. Include specific numbers if data is available

Response:"""
```

**Function Description**:
- Coordinates the entire sub-question query process
- Generates sub-questions and executes queries
- Synthesizes answers from multiple sub-questions
- Provides final comprehensive response

### 3. RAG Pipeline Construction

```python
def create_10q_analysis_system():
    """Create 10Q analysis system"""
    
    # Configure LLM
    try:
        llm = OnlineChatModule()
    except:
        llm = TrainableModule("internlm2-chat-7b")
    
    # Document loading and indexing
    m = TrainableModule("bge-m3").start()
    documents = Document(
        dataset_path=data_path,
        embed=m,
        manager=False
    )
    
    # Create node group
    documents.create_node_group(
        name="sentences",
        transform=SentenceSplitter,
        chunk_size=1024,
        chunk_overlap=128
    )
    
    # Retriever
    retriever = Retriever(
        documents,
        group_name="sentences",
        similarity="cosine",
        topk=6
    ).start()
    
    # Reranker
    reranker = Reranker(
        "ModuleReranker",
        model="bge-reranker-large",
        topk=3,
        output_format='content',
        join=True
    ).start()
```

**Function Description**:
- Uses BGE-M3 model for document embedding
- Uses SentenceSplitter for document chunking
- Configures cosine similarity retriever
- Uses BGE-Reranker for result reranking

## Complete Code Display

```python
import os
from lazyllm import Document, Retriever, Reranker, SentenceSplitter
from lazyllm.module import ModuleBase, OnlineChatModule, TrainableModule
from lazyllm.components import ChatPrompter
from typing import List, Dict, Any
import json

class SubQuestionGenerator(ModuleBase):
    """Sub-question generator"""

    def __init__(self, llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.llm = llm

        # Sub-question generation prompt template
        self.sub_question_prompt = """You are a professional financial analyst. Given a complex question and available data sources, you need to decompose the complex question into multiple simple sub-questions.

Available data sources:
{data_sources}

Complex question: {query}

Please decompose this question into 2-4 sub-questions. Each sub-question should:
1. Target a specific data source
2. Be simple and clear, easy to answer
3. Help answer the original question

Please return in JSON format as follows:
{{
    "sub_questions": [
        {{
            "question": "Sub-question 1",
            "data_source": "Corresponding data source name",
            "reasoning": "Why this sub-question is needed"
        }},
        {{
            "question": "Sub-question 2", 
            "data_source": "Corresponding data source name",
            "reasoning": "Why this sub-question is needed"
        }}
    ]
}}

Return only JSON format, no other content."""

    def forward(self, query: str, data_sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Generate sub-questions"""
        data_sources_str = "\n".join([
            f"- {ds['name']}: {ds['description']}" 
            for ds in data_sources
        ])

        prompt = self.sub_question_prompt.format(
            query=query,
            data_sources=data_sources_str
        )

        prompter = ChatPrompter(instruction=prompt)
        response = self.llm.prompt(prompter)(query)

        try:
            result = json.loads(response)
            return result.get("sub_questions", [])
        except json.JSONDecodeError:
            print("ERROR")
            return [{"question": query, "data_source": data_sources[0]["name"], "reasoning": "Original question"}]

class SubQuestionQueryEngine(ModuleBase):
    """Sub-question query engine"""

    def __init__(self, query_engine: Dict[str, Any], llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.query_engine = query_engine
        self.llm = llm
        self.sub_question_generator = SubQuestionGenerator(llm)

        self.synthesis_prompt = """You are a professional financial analyst. Based on the following sub-question answers, please synthesize a complete response to answer the original question.

Original question: {original_query}

Sub-questions and their answers:
{sub_answers}

Please provide a comprehensive, accurate, and logically clear response that integrates all relevant information. The response should:
1. Directly answer the original question
2. Be based on the provided sub-question answers
3. Maintain objectivity and professionalism
4. Include specific numbers if data is available

Response:"""

    def forward(self, query: str) -> str:
        """Execute sub-question query"""
        data_sources = [
            {"name": self.query_engine["name"], "description": self.query_engine["description"]}
        ]

        sub_questions = self.sub_question_generator(query, data_sources)

        if self._return_trace:
            print(f"Generated {len(sub_questions)} sub-questions:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"[{sq['data_source']}] Q: {sq['question']}")

        sub_answers = []
        for sub_q in sub_questions:
            try:
                answer = self.query_engine["engine"](sub_q["question"])
            except Exception as e:
                answer = f"Query failed: {e}"
            sub_answers.append({
                "question": sub_q["question"],
                "data_source": sub_q["data_source"],
                "answer": answer
            })

            if self._return_trace:
                print(f"[{sub_q['data_source']}] A: {answer}")

        if not sub_answers:
            return "Unable to find relevant information to answer this question."

        sub_answers_str = "\n".join([
            f"Question: {sa['question']}\nData Source: {sa['data_source']}\nAnswer: {sa['answer']}\n"
            for sa in sub_answers
        ])

        synthesis_prompter = ChatPrompter(
            instruction=self.synthesis_prompt.format(
                original_query=query,
                sub_answers=sub_answers_str
            )
        )

        final_answer = self.llm.prompt(synthesis_prompter)(query)
        return final_answer

class QueryEngineTool(ModuleBase):
    """Query engine tool wrapper"""

    def __init__(self, name: str, description: str, engine, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.name = name
        self.description = description
        self.engine = engine

    def forward(self, query: str) -> str:
        return self.engine(query)

def create_10q_analysis_system():
    """Create 10Q analysis system"""

    # Configure LLM
    try:
        llm = OnlineChatModule()
    except:
        llm = TrainableModule("internlm2-chat-7b")

    data_path = "/home/mnt/yehongfei/Code/Test/data/10q"

    required_files = [
        "uber_10q_march_2022.pdf",
        "uber_10q_june_2022.pdf", 
        "uber_10q_sept_2022.pdf"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Missing the following data files: {missing_files}")
        print("Please download Uber's 10Q report files to the data/10q/ directory")
        return None

    print("Loading and indexing documents...")

    m = TrainableModule("bge-m3").start()
    documents = Document(
        dataset_path=data_path,
        embed=m,
        manager=False
    )

    documents.create_node_group(
        name="sentences",
        transform=SentenceSplitter,
        chunk_size=1024,
        chunk_overlap=128
    )

    retriever = Retriever(
        documents,
        group_name="sentences",
        similarity="cosine",
        topk=6
    ).start()

    reranker = Reranker(
        "ModuleReranker",
        model="bge-reranker-large",
        topk=3,
        output_format='content',
        join=True
    ).start()

    def query_func(query_str):
        try:
            nodes = retriever(query_str)
            if not nodes:
                return "No relevant information found."

            try:
                rerank_nodes = reranker.forward(nodes, query_str)
            except Exception as e:
                rerank_nodes = nodes

            if hasattr(rerank_nodes, '__iter__') and not isinstance(rerank_nodes, str):
                context_str = "\n".join([node.get_text() for node in rerank_nodes])
            else:
                context_str = str(rerank_nodes)

            answer_prompt = """You are a professional financial analyst. Based on the following context information, please answer the user's question.

Context information:
{context_str}

User question: {query}

Please provide accurate and professional answers. If there is no relevant information in the context, please clearly state so."""

            prompt = answer_prompt.format(context_str=context_str, query=query_str)
            prompter = ChatPrompter(instruction=prompt)
            try:
                return llm.prompt(prompter)(query_str)
            except Exception as e:
                return f"LLM answer generation error: {e}"

        except Exception as e:
            return f"Query processing error: {e}"

    # Unified query engine tool
    query_engine_tool = QueryEngineTool(
        name="10Q Knowledge Base",
        description="Provides all information about Uber's 10Q reports for each quarter of 2022",
        engine=query_func
    )

    query_engine = {
        "name": "10Q Knowledge Base",
        "description": "Provides all information about Uber's 10Q reports for each quarter of 2022",
        "engine": query_engine_tool
    }

    # Create sub-question query engine
    sub_question_engine = SubQuestionQueryEngine(query_engine, llm, return_trace=True)

    return sub_question_engine

def main():
    """Main function"""
    print("=== LazyLLM 10Q Analysis System ===")
    print()

    analysis_system = create_10q_analysis_system()

    if analysis_system is None:
        return

    print("System initialization complete!")
    print("You can start asking questions. Enter 'quit' to exit.")
    print()

    example_questions = [
        "What percentage did Uber's revenue increase from April-June to July-September 2022?"
    ]

    print("Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    print()

    while True:
        try:
            query = input("Please enter your question: ").strip()

            if query.lower() in ['quit', 'exit']:
                break

            if not query:
                continue

            print("\nAnalyzing...")
            print("-" * 50)

            response = analysis_system(query)

            print("\nFinal Answer:")
            print(response)
            print("-" * 50)
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error occurred during query: {e}")
            print()

if __name__ == "__main__":
    main()
```

## Usage Example

```bash
Please enter your question: What percentage did Uber's revenue increase from April-June to July-September 2022?

Analyzing...
--------------------------------------------------
Generated 2 sub-questions:
[10Q Knowledge Base] Q: What was Uber's revenue in July-September 2022?
[10Q Knowledge Base] Q: What was Uber's revenue in April-June 2022?

[10Q Knowledge Base] A: Uber's revenue in July-September 2022 was $8,343 million.
[10Q Knowledge Base] A: Uber's revenue in April-June 2022 was $8,073 million.

Final Answer:
Uber's revenue in July-September 2022 was $8,343 million, while revenue in April-June 2022 was $8,073 million. To calculate the percentage increase between these two periods, we can use the following formula:

\[ \text{Percentage Increase} = \left( \frac{\text{July-Sep Revenue} - \text{Apr-Jun Revenue}}{\text{Apr-Jun Revenue}} \right) \times 100\% \]

Substituting the given data into the formula:

\[ \text{Percentage Increase} = \left( \frac{8,343 - 8,073}{8,073} \right) \times 100\% \]

\[ \text{Percentage Increase} = \left( \frac{270}{8,073} \right) \times 100\% \]

\[ \text{Percentage Increase} \approx 3.35\% \]

Therefore, Uber's revenue in July-September 2022 increased by approximately 3.35% compared to April-June 2022.
```

## Summary

This tutorial demonstrates how to use the LazyLLM framework to build a professional 10Q analysis system. Through sub-question decomposition techniques, the system can handle complex financial queries and provide accurate, professional analysis results.

The core advantages of the system include:
- Intelligent question decomposition capabilities
- Modular architecture design
- Professional financial analysis capabilities
- Good scalability 