'''
Prompts for Reranker Synthesis Operators

This module contains prompt templates for generating reranker training data.
该模块包含用于生成 Reranker 训练数据的提示词模板。
'''
from typing import List, Optional
from .base_prompt import PromptABC


class RerankerQueryGeneratorPrompt(PromptABC):
    '''
    Prompt template for generating queries from passages for reranker training.
    用于从段落生成查询的提示词模板（用于 Reranker 训练）。
    '''

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一个专业的查询生成专家，专门为文档重排序模型生成训练数据。\
你的任务是根据给定的文档段落，生成不同难度级别的搜索查询。

要求：
1. 生成的查询应该自然、多样，符合真实用户的搜索习惯
2. 查询应该可以通过给定段落来回答
3. 生成不同难度级别的查询：
   - 简单(easy): 直接从文本中可以找到答案的查询
   - 中等(medium): 需要理解文本含义才能回答的查询
   - 困难(hard): 需要推理或综合多个信息点的查询
4. 避免直接复制段落中的句子

请以JSON格式返回结果。'''
        else:
            return '''You are a professional query generation expert, specializing in generating training data \
for document reranking models. Your task is to generate search queries of varying difficulty levels based \
on the given document passage.

Requirements:
1. Generated queries should be natural, diverse, and reflect real user search patterns
2. Queries should be answerable by the given passage
3. Generate queries at different difficulty levels:
   - Easy: Queries with answers directly found in the text
   - Medium: Queries requiring understanding of the text meaning
   - Hard: Queries requiring inference or synthesis of multiple information points
4. Avoid directly copying sentences from the passage

Return results in JSON format.'''

    def build_prompt(
            self,
            passage: str,
            num_queries: int = 3,
            difficulty_levels: Optional[List[str]] = None
    ) -> str:
        difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard']
        levels_str = ', '.join(difficulty_levels)

        if self.lang == 'zh':
            return f'''请根据以下文档段落生成 {num_queries} 个搜索查询，覆盖不同难度级别。

文档段落：
{passage}

要求：
- 生成 {num_queries} 个不同的查询
- 难度级别包括：{levels_str}
- 每个查询都应该可以通过上述段落找到答案
- 这些查询将用于训练重排序模型，需要有区分度

请以以下JSON格式返回：
```json
[
    {{"query": "简单查询内容", "difficulty": "easy"}},
    {{"query": "中等查询内容", "difficulty": "medium"}},
    {{"query": "困难查询内容", "difficulty": "hard"}}
]
```'''
        else:
            return f'''Generate {num_queries} search queries based on the following document passage, \
covering different difficulty levels.

Document Passage:
{passage}

Requirements:
- Generate {num_queries} different queries
- Difficulty levels include: {levels_str}
- Each query should be answerable by the passage above
- These queries will be used for reranker training and need to be discriminative

Return in the following JSON format:
```json
[
    {{"query": "easy query content", "difficulty": "easy"}},
    {{"query": "medium query content", "difficulty": "medium"}},
    {{"query": "hard query content", "difficulty": "hard"}}
]
```'''


class RerankerNegativeGeneratorPrompt(PromptABC):
    '''
    Prompt template for generating confusing negative samples.
    用于生成易混淆负样本的提示词模板。
    '''

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一个专业的负样本生成专家，专门为重排序模型生成困难负样本。\
你的任务是根据给定的查询和正确答案段落，生成看起来相关但实际上不能回答查询的段落。

要求：
1. 生成的负样本应该与查询在主题上相关
2. 负样本应该包含一些与正确答案相似的关键词
3. 但负样本不能真正回答查询的问题
4. 这些负样本用于训练模型区分"相关但不正确"和"正确答案"

请以JSON格式返回结果。'''
        else:
            return '''You are a professional negative sample generation expert, specializing in generating \
hard negatives for reranking models. Your task is to generate passages that look relevant but cannot \
actually answer the query.

Requirements:
1. Generated negatives should be topically related to the query
2. Negatives should contain some keywords similar to the correct answer
3. But negatives should not actually answer the query
4. These negatives are used to train models to distinguish "relevant but incorrect" from "correct answer"

Return results in JSON format.'''

    def build_prompt(
            self,
            query: str,
            positive_passage: str,
            num_negatives: int = 3
    ) -> str:
        if self.lang == 'zh':
            return f'''请根据以下查询和正确答案段落，生成 {num_negatives} 个困难负样本。

查询：{query}

正确答案段落：
{positive_passage}

要求：
- 生成 {num_negatives} 个负样本段落
- 每个负样本应该与查询主题相关，但不能回答查询
- 负样本应该具有迷惑性，包含一些相似的关键词

请以以下JSON格式返回：
```json
[
    {{"negative": "负样本段落1", "reason": "为什么这是负样本"}},
    {{"negative": "负样本段落2", "reason": "为什么这是负样本"}},
    ...
]
```'''
        else:
            return f'''Generate {num_negatives} hard negative samples based on the following query \
and correct passage.

Query: {query}

Correct Passage:
{positive_passage}

Requirements:
- Generate {num_negatives} negative passages
- Each negative should be topically related to the query but cannot answer it
- Negatives should be confusing, containing similar keywords

Return in the following JSON format:
```json
[
    {{"negative": "negative passage 1", "reason": "why this is negative"}},
    {{"negative": "negative passage 2", "reason": "why this is negative"}},
    ...
]
```'''


__all__ = [
    'RerankerQueryGeneratorPrompt',
    'RerankerNegativeGeneratorPrompt',
]
