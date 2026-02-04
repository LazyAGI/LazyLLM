"""
Embedding Data Formatter Operator

This operator formats embedding training data into standard training formats.
该算子将 Embedding 训练数据格式化为标准的训练格式。

原始算法参考：标准的数据格式化方法，参考各主流 Embedding 训练框架的数据格式：
1. FlagEmbedding/BGE: {"query": str, "pos": [str], "neg": [str], "prompt": str}
2. Sentence-Transformers: {"anchor": str, "positive": str, "negative": str}
3. Simple Triplet: {"query": str, "positive": str, "negative": str}
"""
import json
import random
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 复用已存在的 embedding 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingDataFormatter(embedding):
    """
    Format embedding training data into standard formats.
    将 Embedding 训练数据格式化为标准格式。

    Supported output formats:
    - flagembedding: Format for FlagEmbedding/BGE training
      {"query": str, "pos": [str], "neg": [str], "prompt": str}
    - sentence_transformers: Format for Sentence-Transformers training
      {"anchor": str, "positive": str, "negative": str}
    - triplet: Simple triplet format
      {"query": str, "positive": str, "negative": str}

    Args:
        output_format: Target output format (default: "flagembedding")
        instruction: Instruction/prompt to prepend to queries (optional)
        output_file: Path to save formatted data (optional)
        _concurrency_mode: Concurrency mode ('process', 'thread', 'single')
        _save_data: Whether to save intermediate data
    """

    def __init__(
            self,
            output_format: str = "flagembedding",
            instruction: Optional[str] = None,
            output_file: Optional[str] = None,
            _concurrency_mode: str = 'single',
            _save_data: bool = True,
            **kwargs
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, _save_data=_save_data, **kwargs)
        self.output_format = output_format
        self.instruction = instruction
        self.output_file = output_file
        LOG.info(f"Initializing {self.__class__.__name__} with format: {output_format}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingDataFormatter 算子用于将训练数据格式化为标准格式。\n\n"
                "支持的输出格式：\n"
                "- flagembedding: FlagEmbedding/BGE 训练格式\n"
                "- sentence_transformers: Sentence-Transformers 格式\n"
                "- triplet: 简单三元组格式\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- input_neg_key: 负样本字段名（默认：'neg'）\n\n"
                "输出：格式化后的训练数据列表"
            )
        else:
            return (
                "EmbeddingDataFormatter formats training data into standard formats.\n\n"
                "Supported formats:\n"
                "- flagembedding: FlagEmbedding/BGE training format\n"
                "- sentence_transformers: Sentence-Transformers format\n"
                "- triplet: Simple triplet format\n\n"
                "Input:\n"
                "- input_query_key: Query field name (default: 'query')\n"
                "- input_pos_key: Positive samples field name (default: 'pos')\n"
                "- input_neg_key: Negative samples field name (default: 'neg')"
            )

    def _format_flagembedding(
            self,
            query: str,
            pos: List[str],
            neg: List[str],
    ) -> dict:
        """Format to FlagEmbedding training format."""
        result = {
            "query": query,
            "pos": pos if isinstance(pos, list) else [pos],
            "neg": neg if isinstance(neg, list) else [neg],
        }
        if self.instruction:
            result["prompt"] = self.instruction
        return result

    def _format_sentence_transformers(
            self,
            query: str,
            pos: List[str],
            neg: List[str],
    ) -> List[dict]:
        """Format to Sentence-Transformers training format (multiple rows)."""
        results = []
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg]

        # Create anchor-positive-negative triplets
        for p in pos_list:
            for n in neg_list:
                results.append({
                    "anchor": query,
                    "positive": p,
                    "negative": n,
                })
        return results

    def _format_triplet(
            self,
            query: str,
            pos: List[str],
            neg: List[str],
    ) -> List[dict]:
        """Format to simple triplet format."""
        results = []
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg]

        for p in pos_list:
            for n in neg_list:
                results.append({
                    "query": query,
                    "positive": p,
                    "negative": n,
                })
        return results

    def forward_batch_input(
            self,
            inputs: List[dict],
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            input_neg_key: str = "neg",
            **kwargs
    ) -> List[dict]:
        """
        Format the training data.

        Args:
            inputs: List of dict
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field
            input_neg_key: Key for negative samples field

        Returns:
            List of dict in the specified output format
        """
        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Formatting {len(inputs)} samples to {self.output_format} format...")

        results = []
        for item in inputs:
            query = item.get(input_query_key, "")
            pos = item.get(input_pos_key, [])
            neg = item.get(input_neg_key, [])

            if not query or not pos:
                LOG.warning(f"Skipping item with missing query or pos: {item}")
                continue

            # Ensure pos and neg are lists
            if not isinstance(pos, list):
                pos = [pos]
            if not isinstance(neg, list):
                neg = [neg] if neg else []

            # Format based on output format
            if self.output_format == "flagembedding":
                formatted = self._format_flagembedding(query, pos, neg)
                results.append(formatted)
            elif self.output_format == "sentence_transformers":
                formatted = self._format_sentence_transformers(query, pos, neg)
                results.extend(formatted)
            elif self.output_format == "triplet":
                formatted = self._format_triplet(query, pos, neg)
                results.extend(formatted)
            else:
                raise ValueError(f"Unknown output format: {self.output_format}")

        LOG.info(f"Formatted {len(results)} training samples.")

        # Save to file if specified
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved formatted data to {output_path}")

        return results


class EmbeddingTrainTestSplitter(embedding):
    """
    Split embedding training data into train/test sets.
    将 Embedding 训练数据分割为训练集和测试集。

    Args:
        test_size: Proportion of data for test set (default: 0.1)
        seed: Random seed for reproducibility
        stratify_key: Key for stratified splitting (optional)
        train_output_file: Path to save train data (optional)
        test_output_file: Path to save test data (optional)
        _concurrency_mode: Concurrency mode ('process', 'thread', 'single')
        _save_data: Whether to save intermediate data
    """

    def __init__(
            self,
            test_size: float = 0.1,
            seed: int = 42,
            stratify_key: Optional[str] = None,
            train_output_file: Optional[str] = None,
            test_output_file: Optional[str] = None,
            _concurrency_mode: str = 'single',
            _save_data: bool = True,
            **kwargs
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, _save_data=_save_data, **kwargs)
        self.test_size = test_size
        self.seed = seed
        self.stratify_key = stratify_key
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file
        LOG.info(f"Initializing {self.__class__.__name__} with test_size: {test_size}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingTrainTestSplitter 算子用于分割训练集和测试集。\n\n"
                "功能特点：\n"
                "- 支持随机分割和分层分割\n"
                "- 可配置测试集比例\n"
                "- 支持直接输出到文件\n\n"
                "输出：包含 'split' 字段标记的数据列表（'train' 或 'test'）"
            )
        else:
            return (
                "EmbeddingTrainTestSplitter splits data into train/test sets.\n\n"
                "Features:\n"
                "- Supports random and stratified splitting\n"
                "- Configurable test set proportion\n"
                "- Direct output to files\n\n"
                "Output: Data list with 'split' field ('train' or 'test')"
            )

    def forward_batch_input(
            self,
            inputs: List[dict],
            **kwargs
    ) -> List[dict]:
        """
        Split the data into train and test sets.

        Args:
            inputs: List of dict

        Returns:
            List of dict with 'split' field indicating train/test
        """
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
                    item_copy = {k: v for k, v in item.items() if k != 'split'}
                    f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
            LOG.info(f"Saved test data to {output_path}")

        return train_data + test_data
