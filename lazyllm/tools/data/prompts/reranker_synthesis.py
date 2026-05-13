from typing import List, Optional
from .base_prompt import PromptABC


class RerankerQueryGeneratorPrompt(PromptABC):

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一位经验丰富的查询构建专家，专注于为重排序模型制作训练样本。\
你的职责是基于提供的文档片段，构建具有不同复杂程度的检索查询。

标准：
1. 构建的查询应当自然流畅、形式多样，贴近实际用户的检索行为
2. 每个查询都能从给定段落中获得答案
3. 构建涵盖不同复杂程度的查询：
   - 简单(easy): 答案可直接在文本中定位的查询
   - 中等(medium): 需要理解文本深层含义才能解答的查询
   - 困难(hard): 需要逻辑推理或整合多个信息要素的查询
4. 禁止直接照搬段落中的原句

请使用JSON格式输出结果。'''
        else:
            return '''You are an experienced query construction specialist, focused on creating training samples \
for reranking models. Your responsibility is to build search queries with varying complexity levels based \
on the provided document segment.

Standards:
1. Constructed queries should be natural, fluent, and diverse, mirroring actual user search behaviors
2. Each query should be answerable from the given passage
3. Build queries covering different complexity levels:
   - Easy: Queries where answers can be directly located in the text
   - Medium: Queries requiring comprehension of the text's deeper meaning
   - Hard: Queries requiring logical reasoning or integration of multiple information elements
4. Do not directly copy sentences from the passage

Output results in JSON format.'''

    def build_prompt(
            self,
            passage: str,
            num_queries: int = 3,
            difficulty_levels: Optional[List[str]] = None
    ) -> str:
        difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard']
        levels_str = ', '.join(difficulty_levels)

        if self.lang == 'zh':
            return f'''请基于以下文档片段构建 {num_queries} 个检索查询，确保涵盖不同复杂程度。

文档片段：
{passage}

标准：
- 构建 {num_queries} 个互不相同的查询
- 复杂程度应包含：{levels_str}
- 所有查询都能从上述片段中获取答案
- 这些查询将用于重排序模型训练，需具备良好的区分能力

请按照以下JSON格式输出：
```json
[
    {{"query": "简单查询内容", "difficulty": "easy"}},
    {{"query": "中等查询内容", "difficulty": "medium"}},
    {{"query": "困难查询内容", "difficulty": "hard"}}
]
```'''
        else:
            return f'''Build {num_queries} search queries based on the following document segment, \
ensuring coverage of different complexity levels.

Document Segment:
{passage}

Standards:
- Construct {num_queries} distinct queries
- Complexity levels should include: {levels_str}
- All queries should be answerable from the segment above
- These queries will be used for reranker model training and need strong discriminative power

Output in the following JSON format:
```json
[
    {{"query": "easy query content", "difficulty": "easy"}},
    {{"query": "medium query content", "difficulty": "medium"}},
    {{"query": "hard query content", "difficulty": "hard"}}
]
```'''


class RerankerNegativeGeneratorPrompt(PromptABC):

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == 'zh':
            return '''你是一位资深的负样本构建专家，专注于为重排序模型制作具有挑战性的负样本。\
你的职责是基于提供的查询和正确回答段落，构建看似相关但实际无法解答查询的段落。

标准：
1. 构建的负样本应当与查询在主题层面相关
2. 负样本应当包含与正确答案相近的部分关键词
3. 但负样本无法真正解决查询提出的问题
4. 这些负样本用于帮助模型学会区分"相关但错误"与"正确回答"

请使用JSON格式输出结果。'''
        else:
            return '''You are an experienced negative sample construction specialist, focused on creating \
challenging negatives for reranking models. Your responsibility is to build passages that appear relevant \
but cannot actually answer the query.

Standards:
1. Constructed negatives should be topically related to the query
2. Negatives should contain some keywords similar to the correct answer
3. But negatives should not actually solve the query's question
4. These negatives are used to help models learn to distinguish "relevant but incorrect" from "correct answer"

Output results in JSON format.'''

    def build_prompt(
            self,
            query: str,
            positive_passage: str,
            num_negatives: int = 3
    ) -> str:
        if self.lang == 'zh':
            return f'''请基于以下查询和正确回答段落，构建 {num_negatives} 个具有挑战性的负样本。

查询：{query}

正确回答段落：
{positive_passage}

标准：
- 构建 {num_negatives} 个负样本段落
- 每个负样本应当与查询主题相关，但无法解答查询
- 负样本应当具备迷惑性，包含部分相似的关键词

请按照以下JSON格式输出：
```json
[
    {{"negative": "负样本段落1", "reason": "为什么这是负样本"}},
    {{"negative": "负样本段落2", "reason": "为什么这是负样本"}},
    ...
]
```'''
        else:
            return f'''Build {num_negatives} challenging negative samples based on the following query \
and correct passage.

Query: {query}

Correct Passage:
{positive_passage}

Standards:
- Construct {num_negatives} negative passages
- Each negative should be topically related to the query but cannot answer it
- Negatives should be confusing, containing similar keywords

Output in the following JSON format:
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
