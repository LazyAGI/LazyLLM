import json
import random
import os
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


class RerankerValidateData(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(
        self,
        data: dict,
        input_query_key: str = "query",
        input_pos_key: str = "pos",
        input_neg_key: str = "neg",
        **kwargs
    ) -> dict:
        query = data.get(input_query_key, "")
        pos = data.get(input_pos_key, [])

        if not query:
            return {**data, '_is_valid': False, '_error': 'Missing query'}
        
        if not pos:
            return {**data, '_is_valid': False, '_error': 'Missing positive samples'}

        # Ensure pos and neg are lists
        if not isinstance(pos, list):
            pos = [pos]
        
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


class RerankerFormatFlagReranker(reranker):
    def __init__(self, train_group_size: int = 8, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.train_group_size = train_group_size

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        # Ensure neg has exactly train_group_size - 1 samples
        num_neg_needed = self.train_group_size - 1
        if len(neg) < num_neg_needed:
            # Pad with duplicates if needed
            neg = (neg * (num_neg_needed // len(neg) + 1))[:num_neg_needed] if neg else []
        else:
            neg = neg[:num_neg_needed]

        return [{
            "query": query,
            "pos": pos,
            "neg": neg,
        }]


class RerankerFormatCrossEncoder(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        results = []

        # Positive samples with label 1
        for p in pos:
            results.append({"query": query, "document": p, "label": 1})

        # Negative samples with label 0
        for n in neg:
            results.append({"query": query, "document": n, "label": 0})

        return results


class RerankerFormatPairwise(reranker):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict, **kwargs) -> List[dict]:
        if not data.get('_is_valid'):
            return []

        query = data['_query']
        pos = data['_pos']
        neg = data['_neg']

        results = []

        # Create pairwise comparisons
        for p in pos:
            for n in neg:
                results.append({"query": query, "doc_pos": p, "doc_neg": n})

        return results


class RerankerSaveToFile(reranker):
    def __init__(self, output_file: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_file = output_file

    def forward(self, data: dict, **kwargs) -> dict:
        if not self.output_file:
            return data

        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            LOG.warning(f"Failed to save to file: {e}")

        return data


class RerankerTrainTestSplitter(reranker):
    def __init__(
            self,
            test_size: float = 0.1,
            seed: int = 42,
            train_output_file: Optional[str] = None,
            test_output_file: Optional[str] = None,
            **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f"Initializing {self.__class__.__name__} with test_size: {test_size}")

    def forward_batch_input(self, data: List[dict]) -> List[dict]:
        assert isinstance(data, list), "Input data must be a list"
        records = list(data)

        LOG.info(f"Splitting {len(records)} samples with test_size={self.test_size}")

        # Shuffle and split
        random.seed(self.seed)
        shuffled = records.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - self.test_size))
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]

        # Add split labels
        for item in train_data:
            item['split'] = 'train'
        for item in test_data:
            item['split'] = 'test'

        LOG.info(f"Split completed: {len(train_data)} train, {len(test_data)} test")

        # Save to files if specified
        if self.train_output_file:
            output_path = Path(self.train_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in train_data:
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved train data to {output_path}")

        if self.test_output_file:
            output_path = Path(self.test_output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    # For eval data, rename pos to corpus for compatibility
                    item_copy = {
                        'query': item.get('query', ''),
                        'corpus': item.get('pos', []),
                        'neg': item.get('neg', [])
                    }
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved test data to {output_path}")

        return train_data + test_data
