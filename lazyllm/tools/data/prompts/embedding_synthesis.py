from typing import List, Optional
from .base_prompt import PromptABC


class EmbeddingQueryGeneratorPrompt(PromptABC):

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一位经验丰富的查询构建专家。你的职责是基于提供的文档片段，构建高质量的检索查询。

标准：
1. 构建的查询应当自然流畅、形式多样，贴近实际用户的检索行为
2. 每个查询都能从给定片段中获得答案
3. 禁止直接照搬片段中的原句
4. 构建涵盖不同类别的查询（事实类、语义类、推理类）

请使用JSON格式输出结果。'''
        else:
            return '''You are an experienced query construction specialist. Your responsibility is to build \
high-quality search queries based on the provided document segment.

Standards:
1. Constructed queries should be natural, fluent, and diverse, mirroring actual user search behaviors
2. Each query should be answerable from the given segment
3. Do not directly copy sentences from the segment
4. Build queries covering different categories (factual, semantic, inferential)

Output results in JSON format.'''

    def build_prompt(
            self,
            passage: str,
            num_queries: int = 3,
            query_types: Optional[List[str]] = None
    ) -> str:
        query_types = query_types or ['factual', 'semantic', 'inferential']
        types_str = ', '.join(query_types)

        if self.lang == 'zh':
            return f'''请基于以下文档片段构建 {num_queries} 个检索查询。

文档片段：
{passage}

标准：
- 构建 {num_queries} 个互不相同的查询
- 查询类别应包含：{types_str}
- 所有查询都能从上述片段中获取答案

请按照以下JSON格式输出：
```json
[
    {{"query": "查询内容1", "type": "factual"}},
    {{"query": "查询内容2", "type": "semantic"}},
    ...
]
```'''
        else:
            return f'''Build {num_queries} search queries based on the following document segment.

Document Segment:
{passage}

Standards:
- Construct {num_queries} distinct queries
- Query categories should include: {types_str}
- All queries should be answerable from the segment above

Output in the following JSON format:
```json
[
    {{"query": "query content 1", "type": "factual"}},
    {{"query": "query content 2", "type": "semantic"}},
    ...
]
```'''


class EmbeddingQueryAugmentPrompt(PromptABC):

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一位查询转换专家。你的职责是将提供的查询转换为语义一致但表述不同的形式。

标准：
1. 维持原始查询的语义内容不变
2. 采用不同的用词和句型结构
3. 构建自然流畅的表述
4. 避免仅进行简单的同义词替换

请使用JSON格式输出结果。'''
        else:
            return '''You are a query transformation specialist. Your responsibility is to convert the provided \
query into semantically equivalent but differently worded forms.

Standards:
1. Maintain the original query's semantic content
2. Employ different word choices and sentence patterns
3. Build natural and fluent expressions
4. Avoid mere synonym replacement

Output results in JSON format.'''

    def build_prompt(self, query: str, num_rewrites: int = 2) -> str:
        if self.lang == 'zh':
            return f'''请将以下查询转换为 {num_rewrites} 个不同的表述形式，确保语义保持一致。

原始查询：{query}

请按照以下JSON格式输出：
```json
[
    "改写后的查询1",
    "改写后的查询2",
    ...
]
```'''
        else:
            return f'''Transform the following query into {num_rewrites} different wordings \
while maintaining the same meaning.

Original query: {query}

Output in the following JSON format:
```json
[
    "rewritten query 1",
    "rewritten query 2",
    ...
]
```'''


class EmbeddingPassageEnhancePrompt(PromptABC):

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一位文档优化专家。你的职责是对提供的文档片段进行增强处理，提升其在检索系统中的可发现性。

优化方法：
1. 识别并提取关键实体和核心概念
2. 补充摘要信息或重要要点
3. 添加相关的同义表达或别名形式

请使用JSON格式输出结果。'''
        else:
            return '''You are a document optimization specialist. Your responsibility is to enhance the provided \
document segment to improve its discoverability in retrieval systems.

Optimization approaches:
1. Identify and extract key entities and core concepts
2. Supplement summary information or key points
3. Add related synonymous expressions or alias forms

Output results in JSON format.'''

    def build_prompt(self, passage: str) -> str:
        if self.lang == 'zh':
            return f'''请对以下文档片段进行优化处理。

原始片段：
{passage}

请按照以下JSON格式输出：
```json
{{
    "enhanced_passage": "增强后的段落",
    "key_entities": ["实体1", "实体2"],
    "summary": "段落摘要",
    "keywords": ["关键词1", "关键词2"]
}}
```'''
        else:
            return f'''Optimize the following document segment.

Original segment:
{passage}

Output in the following JSON format:
```json
{{
    "enhanced_passage": "enhanced passage",
    "key_entities": ["entity1", "entity2"],
    "summary": "passage summary",
    "keywords": ["keyword1", "keyword2"]
}}
```'''


__all__ = [
    'EmbeddingQueryGeneratorPrompt',
    'EmbeddingQueryAugmentPrompt',
    'EmbeddingPassageEnhancePrompt',
]
