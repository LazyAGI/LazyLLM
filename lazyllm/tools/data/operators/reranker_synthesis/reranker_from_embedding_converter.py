import json
import random
from pathlib import Path
from typing import Optional

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


@data_register('data.reranker', rewrite_func='forward', _concurrency_mode='process')
def validate_reranker_embedding_data(
    data: dict,
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    input_neg_key: str = 'neg',
) -> dict:
    query = data.get(input_query_key, '')
    pos = data.get(input_pos_key, [])

    if not query:
        return {**data, '_is_valid': False, '_error': 'Empty query'}

    # Ensure pos is a list
    if not isinstance(pos, list):
        pos = [pos] if pos else []

    if not pos:
        return {**data, '_is_valid': False, '_error': 'No positive samples'}

    # Ensure neg is a list
    neg = data.get(input_neg_key, [])
    if not isinstance(neg, list):
        neg = [neg] if neg else []

    return {
        **data,
        '_is_valid': True,
        '_query': query,
        '_pos': pos,
        '_neg': neg
    }


class RerankerAdjustNegatives(reranker):
    def __init__(self, adjust_neg_count: int = 7, seed: int = 42, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.adjust_neg_count = adjust_neg_count
        self.seed = seed

    def forward(self, data: dict, **kwargs) -> dict:
        if not data.get('_is_valid'):
            return data

        neg = data.get('_neg', [])

        if len(neg) > self.adjust_neg_count:
            # Truncate to target count
            neg = neg[:self.adjust_neg_count]
        elif len(neg) < self.adjust_neg_count and neg:
            # Pad with duplicates if needed (when we have some negatives)
            local_random = random.Random(f'{self.seed}_{data["_query"]}')
            while len(neg) < self.adjust_neg_count:
                neg.append(local_random.choice(neg))

        return {**data, '_neg': neg}


class RerankerBuildFormat(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict, **kwargs) -> dict:
        if not data.get('_is_valid'):
            return {}

        # Build reranker format (no prompt/instruction)
        reranker_item = {
            'query': data['_query'],
            'pos': data['_pos'],
            'neg': data['_neg'],
        }

        return reranker_item
