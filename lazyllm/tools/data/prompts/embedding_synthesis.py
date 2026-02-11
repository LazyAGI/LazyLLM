'''
Prompts for Embedding Synthesis Operators

This module contains prompt templates for generating embedding training data.
该模块包含用于生成 Embedding 训练数据的提示词模板。
'''
from typing import List, Optional
from .base_prompt import PromptABC


class EmbeddingQueryGeneratorPrompt(PromptABC):
    '''
    Prompt template for generating queries from passages.
    用于从段落生成查询的提示词模板。
    '''

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一个专业的查询生成专家。你的任务是根据给定的文档段落，生成高质量的搜索查询。

要求：
1. 生成的查询应该自然、多样，符合真实用户的搜索习惯
2. 查询应该可以通过给定段落来回答
3. 避免直接复制段落中的句子
4. 生成不同类型的查询（事实型、语义型、推理型）

请以JSON格式返回结果。'''
        else:
            return '''You are a professional query generation expert. Your task is to generate high-quality \
search queries based on the given document passage.

Requirements:
1. Generated queries should be natural, diverse, and reflect real user search patterns
2. Queries should be answerable by the given passage
3. Avoid directly copying sentences from the passage
4. Generate different types of queries (factual, semantic, inferential)

Return results in JSON format.'''

    def build_prompt(
            self,
            passage: str,
            num_queries: int = 3,
            query_types: Optional[List[str]] = None
    ) -> str:
        query_types = query_types or ['factual', 'semantic', 'inferential']
        types_str = ', '.join(query_types)

        if self.lang == 'zh':
            return f'''请根据以下文档段落生成 {num_queries} 个搜索查询。

文档段落：
{passage}

要求：
- 生成 {num_queries} 个不同的查询
- 查询类型包括：{types_str}
- 每个查询都应该可以通过上述段落找到答案

请以以下JSON格式返回：
```json
[
    {{"query": "查询内容1", "type": "factual"}},
    {{"query": "查询内容2", "type": "semantic"}},
    ...
]
```'''
        else:
            return f'''Generate {num_queries} search queries based on the following document passage.

Document Passage:
{passage}

Requirements:
- Generate {num_queries} different queries
- Query types include: {types_str}
- Each query should be answerable by the passage above

Return in the following JSON format:
```json
[
    {{"query": "query content 1", "type": "factual"}},
    {{"query": "query content 2", "type": "semantic"}},
    ...
]
```'''


class EmbeddingQueryAugmentPrompt(PromptABC):
    '''
    Prompt template for augmenting/rewriting queries.
    用于增强/改写查询的提示词模板。
    '''

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一个查询改写专家。你的任务是将给定的查询改写成语义相同但表达不同的形式。

要求：
1. 保持原始查询的语义不变
2. 使用不同的词汇和句式
3. 生成自然流畅的表达
4. 避免简单的同义词替换

请以JSON格式返回结果。'''
        else:
            return '''You are a query rewriting expert. Your task is to rewrite the given query into \
semantically equivalent but differently expressed forms.

Requirements:
1. Preserve the original query's meaning
2. Use different vocabulary and sentence structures
3. Generate natural and fluent expressions
4. Avoid simple synonym substitution

Return results in JSON format.'''

    def build_prompt(self, query: str, num_rewrites: int = 2) -> str:
        if self.lang == 'zh':
            return f'''请将以下查询改写成 {num_rewrites} 个不同的表达方式，保持语义相同。

原始查询：{query}

请以以下JSON格式返回：
```json
[
    "改写后的查询1",
    "改写后的查询2",
    ...
]
```'''
        else:
            return f'''Rewrite the following query into {num_rewrites} different expressions \
while preserving the same meaning.

Original query: {query}

Return in the following JSON format:
```json
[
    "rewritten query 1",
    "rewritten query 2",
    ...
]
```'''


class EmbeddingPassageEnhancePrompt(PromptABC):
    '''
    Prompt template for enhancing passages for better retrieval.
    用于增强段落以提升检索效果的提示词模板。
    '''

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一个文档处理专家。你的任务是对给定的文档段落进行增强，使其更易于被检索系统找到。

增强方式：
1. 提取关键实体和概念
2. 添加摘要或关键信息
3. 补充相关的同义词或别名

请以JSON格式返回结果。'''
        else:
            return '''You are a document processing expert. Your task is to enhance the given document passage \
to make it more discoverable by retrieval systems.

Enhancement methods:
1. Extract key entities and concepts
2. Add summaries or key information
3. Include related synonyms or aliases

Return results in JSON format.'''

    def build_prompt(self, passage: str) -> str:
        if self.lang == 'zh':
            return f'''请对以下文档段落进行增强处理。

原始段落：
{passage}

请以以下JSON格式返回：
```json
{{
    "enhanced_passage": "增强后的段落",
    "key_entities": ["实体1", "实体2"],
    "summary": "段落摘要",
    "keywords": ["关键词1", "关键词2"]
}}
```'''
        else:
            return f'''Enhance the following document passage.

Original passage:
{passage}

Return in the following JSON format:
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
