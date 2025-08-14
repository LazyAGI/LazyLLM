# LazyLLM 10Q分析系统教程

本教程将展示如何使用LazyLLM框架构建一个复杂的10Q财务报告分析系统，通过子问题分解技术来回答复杂的金融查询。

## 概述

本系统能够实现：

- 将复杂问题分解为多个简单子问题
- 针对不同数据源进行查询
- 合成多个子问题的答案形成完整回答
- 提供专业的金融分析

## 系统架构

### 核心组件

1. **SubQuestionGenerator**: 子问题生成器
2. **SubQuestionQueryEngine**: 子问题查询引擎
3. **QueryEngineTool**: 查询引擎工具包装器
4. **RAG Pipeline**: 检索增强生成管道

### 工作流程

```
复杂问题 → 子问题分解 → 多数据源查询 → 答案合成 → 最终回答
```

## 环境准备

### 1. 安装依赖

```bash
pip install lazyllm
```

### 2. 准备数据

下载Uber的10Q报告文件到`data/10q/`目录：

```bash
mkdir -p data/10q/
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'
```

## 核心代码解析

### 1. 子问题生成器 (SubQuestionGenerator)

```python
class SubQuestionGenerator(ModuleBase):
    """子问题生成器"""
    
    def __init__(self, llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.llm = llm
        
        # 子问题生成提示模板
        self.sub_question_prompt = """你是一个专业的金融分析师。给定一个复杂的问题和可用的数据源，你需要将复杂问题分解为多个简单的子问题。

可用数据源：
{data_sources}

复杂问题：{query}

请将这个问题分解为2-4个子问题，每个子问题应该：
1. 针对特定的数据源
2. 简单明确，易于回答
3. 能够帮助回答原始问题

请以JSON格式返回，格式如下：
{{
    "sub_questions": [
        {{
            "question": "子问题1",
            "data_source": "对应的数据源名称",
            "reasoning": "为什么需要这个子问题"
        }},
        {{
            "question": "子问题2", 
            "data_source": "对应的数据源名称",
            "reasoning": "为什么需要这个子问题"
        }}
    ]
}}

只返回JSON格式，不要其他内容。"""
```

**功能说明**：
- 接收复杂问题和可用数据源
- 使用LLM将复杂问题分解为2-4个简单子问题
- 每个子问题都关联到特定的数据源
- 返回结构化的JSON格式

### 2. 子问题查询引擎 (SubQuestionQueryEngine)

```python
class SubQuestionQueryEngine(ModuleBase):
    """子问题查询引擎"""
    
    def __init__(self, query_engine: Dict[str, Any], llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.query_engine = query_engine
        self.llm = llm
        self.sub_question_generator = SubQuestionGenerator(llm)
        
        self.synthesis_prompt = """你是一个专业的金融分析师。基于以下子问题的答案，请合成一个完整的回答来回答原始问题。

原始问题：{original_query}

子问题及其答案：
{sub_answers}

请提供一个全面、准确、逻辑清晰的回答，整合所有相关信息。回答应该：
1. 直接回答原始问题
2. 基于提供的子问题答案
3. 保持客观和专业性
4. 如果有数据，请包含具体数字

回答："""
```

**功能说明**：
- 协调整个子问题查询流程
- 生成子问题并执行查询
- 合成多个子问题的答案
- 提供最终的综合回答

### 3. RAG管道构建

```python
def create_10q_analysis_system():
    """创建10Q分析系统"""
    
    # 配置LLM
    try:
        llm = OnlineChatModule()
    except:
        llm = TrainableModule("internlm2-chat-7b")
    
    # 文档加载和索引
    m = TrainableModule("bge-m3").start()
    documents = Document(
        dataset_path=data_path,
        embed=m,
        manager=False
    )
    
    # 创建节点组
    documents.create_node_group(
        name="sentences",
        transform=SentenceSplitter,
        chunk_size=1024,
        chunk_overlap=128
    )
    
    # 检索器
    retriever = Retriever(
        documents,
        group_name="sentences",
        similarity="cosine",
        topk=6
    ).start()
    
    # 重排序器
    reranker = Reranker(
        "ModuleReranker",
        model="bge-reranker-large",
        topk=3,
        output_format='content',
        join=True
    ).start()
```

**功能说明**：
- 使用BGE-M3模型进行文档嵌入
- 使用SentenceSplitter进行文档分块
- 配置余弦相似度检索器
- 使用BGE-Reranker进行结果重排序

## 完整代码展示

```python
import os
from lazyllm import Document, Retriever, Reranker, SentenceSplitter
from lazyllm.module import ModuleBase, OnlineChatModule, TrainableModule
from lazyllm.components import ChatPrompter
from typing import List, Dict, Any
import json

class SubQuestionGenerator(ModuleBase):
    """子问题生成器"""

    def __init__(self, llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.llm = llm

        # 子问题生成提示模板
        self.sub_question_prompt = """你是一个专业的金融分析师。给定一个复杂的问题和可用的数据源，你需要将复杂问题分解为多个简单的子问题。

可用数据源：
{data_sources}

复杂问题：{query}

请将这个问题分解为2-4个子问题，每个子问题应该：
1. 针对特定的数据源
2. 简单明确，易于回答
3. 能够帮助回答原始问题

请以JSON格式返回，格式如下：
{{
    "sub_questions": [
        {{
            "question": "子问题1",
            "data_source": "对应的数据源名称",
            "reasoning": "为什么需要这个子问题"
        }},
        {{
            "question": "子问题2", 
            "data_source": "对应的数据源名称",
            "reasoning": "为什么需要这个子问题"
        }}
    ]
}}

只返回JSON格式，不要其他内容。"""

    def forward(self, query: str, data_sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """生成子问题"""
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
            return [{"question": query, "data_source": data_sources[0]["name"], "reasoning": "原始问题"}]

class SubQuestionQueryEngine(ModuleBase):
    """子问题查询引擎"""

    def __init__(self, query_engine: Dict[str, Any], llm: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.query_engine = query_engine
        self.llm = llm
        self.sub_question_generator = SubQuestionGenerator(llm)

        self.synthesis_prompt = """你是一个专业的金融分析师。基于以下子问题的答案，请合成一个完整的回答来回答原始问题。

原始问题：{original_query}

子问题及其答案：
{sub_answers}

请提供一个全面、准确、逻辑清晰的回答，整合所有相关信息。回答应该：
1. 直接回答原始问题
2. 基于提供的子问题答案
3. 保持客观和专业性
4. 如果有数据，请包含具体数字

回答："""

    def forward(self, query: str) -> str:
        """执行子问题查询"""
        data_sources = [
            {"name": self.query_engine["name"], "description": self.query_engine["description"]}
        ]

        sub_questions = self.sub_question_generator(query, data_sources)

        if self._return_trace:
            print(f"生成了 {len(sub_questions)} 个子问题:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"[{sq['data_source']}] Q: {sq['question']}")

        sub_answers = []
        for sub_q in sub_questions:
            try:
                answer = self.query_engine["engine"](sub_q["question"])
            except Exception as e:
                answer = f"查询失败: {e}"
            sub_answers.append({
                "question": sub_q["question"],
                "data_source": sub_q["data_source"],
                "answer": answer
            })

            if self._return_trace:
                print(f"[{sub_q['data_source']}] A: {answer}")

        if not sub_answers:
            return "无法找到相关信息来回答这个问题。"

        sub_answers_str = "\n".join([
            f"问题：{sa['question']}\n数据源：{sa['data_source']}\n答案：{sa['answer']}\n"
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
    """查询引擎工具包装器"""

    def __init__(self, name: str, description: str, engine, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.name = name
        self.description = description
        self.engine = engine

    def forward(self, query: str) -> str:
        return self.engine(query)

def create_10q_analysis_system():
    """创建10Q分析系统"""

    # 配置LLM
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
        print(f"缺少以下数据文件: {missing_files}")
        print("请下载Uber的10Q报告文件到data/10q/目录")
        return None

    print("正在加载和索引文档...")

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
                return "未找到相关信息。"

            try:
                rerank_nodes = reranker.forward(nodes, query_str)
            except Exception as e:
                rerank_nodes = nodes

            if hasattr(rerank_nodes, '__iter__') and not isinstance(rerank_nodes, str):
                context_str = "\n".join([node.get_text() for node in rerank_nodes])
            else:
                context_str = str(rerank_nodes)

            answer_prompt = """你是一个专业的金融分析师。基于以下上下文信息，请回答用户的问题。

上下文信息：
{context_str}

用户问题：{query}

请提供准确、专业的回答，如果上下文中没有相关信息，请明确说明。"""

            prompt = answer_prompt.format(context_str=context_str, query=query_str)
            prompter = ChatPrompter(instruction=prompt)
            try:
                return llm.prompt(prompter)(query_str)
            except Exception as e:
                return f"LLM生成答案错误: {e}"

        except Exception as e:
            return f"查询处理错误: {e}"

    # 统一的查询引擎工具
    query_engine_tool = QueryEngineTool(
        name="10Q知识库",
        description="提供关于Uber 2022年各季度10Q报告的所有信息",
        engine=query_func
    )

    query_engine = {
        "name": "10Q知识库",
        "description": "提供关于Uber 2022年各季度10Q报告的所有信息",
        "engine": query_engine_tool
    }

    # 创建子问题查询引擎
    sub_question_engine = SubQuestionQueryEngine(query_engine, llm, return_trace=True)

    return sub_question_engine

def main():
    """主函数"""
    print("=== LazyLLM 10Q分析系统 ===")
    print()

    analysis_system = create_10q_analysis_system()

    if analysis_system is None:
        return

    print("系统初始化完成！")
    print("可以开始提问了。输入 'quit' 退出。")
    print()

    example_questions = [
        "Uber在2022年7月-9月的收入比4月-6月增长了多少百分比？"
    ]

    print("示例问题：")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    print()

    while True:
        try:
            query = input("请输入您的问题: ").strip()

            if query.lower() in ['quit', 'exit', '退出']:
                break

            if not query:
                continue

            print("\n正在分析...")
            print("-" * 50)

            response = analysis_system(query)

            print("\n最终答案:")
            print(response)
            print("-" * 50)
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"查询过程中出现错误: {e}")
            print()

if __name__ == "__main__":
    main()
```

## 使用示例

```bash
请输入您的问题: Uber在2022年7-9月的收入比4-6月的收入增长了多少百分比?

正在分析...
--------------------------------------------------
生成了 2 个子问题:
[10Q知识库] Q: Uber在2022年7-9月的收入是多少？
[10Q知识库] Q: Uber在2022年4-6月的收入是多少？

[10Q知识库] A: Uber在2022年7-9月的收入是8,343百万美元。
[10Q知识库] A: Uber在2022年4-6月的收入是8,073百万美元。

最终答案:
Uber在2022年7-9月的收入为8,343百万美元，而2022年4-6月的收入为8,073百万美元。为了计算这两个时期的收入增长百分比，我们可以使用以下公式：

\[ \text{增长百分比} = \left( \frac{\text{7-9月收入} - \text{4-6月收入}}{\text{4-6月收入}} \right) \times 100\% \]

将给定的数据代入公式中：

\[ \text{增长百分比} = \left( \frac{8,343 - 8,073}{8,073} \right) \times 100\% \]

\[ \text{增长百分比} = \left( \frac{270}{8,073} \right) \times 100\% \]

\[ \text{增长百分比} \approx 3.35\% \]

因此，Uber在2022年7-9月的收入比4-6月的收入增长了大约3.35%。
```

## 总结

本教程展示了如何使用LazyLLM框架构建一个专业的10Q分析系统。通过子问题分解技术，系统能够处理复杂的金融查询，并提供准确、专业的分析结果。

系统的核心优势在于：
- 智能的问题分解能力
- 模块化的架构设计
- 专业的金融分析能力
- 良好的可扩展性 