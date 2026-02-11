import json
import random
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# Get or create embedding group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingFormatFlagEmbedding(embedding):
    def __init__(self, instruction: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.instruction = instruction

    def forward(self, data: dict) -> dict:
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        if not isinstance(pos, list):
            pos = [pos]
        if not isinstance(neg, list):
            neg = [neg] if neg else []

        result = {
            "query": query,
            "pos": pos,
            "neg": neg,
        }
        if self.instruction:
            result["prompt"] = self.instruction

        return result


class EmbeddingFormatSentenceTransformers(embedding):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg] if neg else []

        # Create anchor-positive-negative triplets
        results = []
        for p in pos_list:
            for n in neg_list:
                results.append({
                    "anchor": query,
                    "positive": p,
                    "negative": n,
                })

        return results


class EmbeddingFormatTriplet(embedding):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)

    def forward(self, data: dict) -> List[dict]:
        query = data.get('query', '')
        pos = data.get('pos', [])
        neg = data.get('neg', [])

        if not query or not pos:
            return []

        # Ensure pos and neg are lists
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg] if neg else []

        # Create query-positive-negative triplets
        results = []
        for p in pos_list:
            for n in neg_list:
                results.append({
                    "query": query,
                    "positive": p,
                    "negative": n,
                })

        return results


class EmbeddingTrainTestSplitter(embedding):
    def __init__(
        self,
        test_size: float = 0.1,
        seed: int = 42,
        stratify_key: Optional[str] = None,
        train_output_file: Optional[str] = None,
        test_output_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.stratify_key = stratify_key
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f"Initializing {self.__class__.__name__} with test_size: {test_size}")

    def forward_batch_input(
        self,
        inputs: List[dict],
        **kwargs
    ) -> List[dict]:
        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Splitting {len(inputs)} samples with test_size={self.test_size}")

        # Shuffle and split
        random.seed(self.seed)
        shuffled = inputs.copy()
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
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved test data to {output_path}")

        return train_data + test_data
