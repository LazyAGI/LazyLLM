"""
Reranker From Embedding Converter Operator

This operator converts embedding training data to reranker format.
该算子将 Embedding 训练数据转换为 Reranker 格式。
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List
from lazyllm import LOG
from ...base_data import DataOperatorRegistry


@DataOperatorRegistry.register(one_item=False, tag='reranker_synthesis')
class RerankerFromEmbeddingConverter:
    """
    Convert embedding training data to reranker format.
    将 Embedding 训练数据转换为 Reranker 格式。

    The main difference between embedding and reranker training data:
    - Embedding: {"query": str, "pos": [str], "neg": [str], "prompt": str}
    - Reranker: {"query": str, "pos": [str], "neg": [str]} (no prompt/instruction)

    This operator:
    1. Removes the prompt/instruction field
    2. Optionally re-mines negatives for better reranker training
    3. Adjusts negative count to match train_group_size

    Args:
        remove_instruction: Whether to remove instruction/prompt fields (default: True)
        adjust_neg_count: Target number of negatives per sample (default: 7)
        output_file: Path to save converted data (optional)
    """

    def __init__(
            self,
            remove_instruction: bool = True,
            adjust_neg_count: int = 7,
            output_file: Optional[str] = None,
    ):
        self.remove_instruction = remove_instruction
        self.adjust_neg_count = adjust_neg_count
        self.output_file = output_file
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerFromEmbeddingConverter 算子用于将 Embedding 训练数据转换为 Reranker 格式。\n\n"
                "核心功能：\n"
                "- 移除 prompt/instruction 字段\n"
                "- 调整负样本数量以匹配 train_group_size\n"
                "- 保持 query-pos-neg 格式\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- input_neg_key: 负样本字段名（默认：'neg'）\n\n"
                "输出：Reranker 格式的训练数据列表"
            )
        else:
            return (
                "RerankerFromEmbeddingConverter converts embedding training data to reranker format.\n\n"
                "Features:\n"
                "- Removes prompt/instruction fields\n"
                "- Adjusts negative count to match train_group_size\n"
                "- Maintains query-pos-neg format"
            )

    def _load_embedding_data(self, file_path: str) -> List[dict]:
        """Load embedding training data from file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Support both JSON array and JSONL formats
            if content.startswith('['):
                data = json.loads(content)
            else:
                f.seek(0)
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data

    def __call__(
            self,
            data,
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            input_neg_key: str = "neg",
    ):
        """
        Convert embedding data to reranker format.

        Args:
            data: List of dict, pandas DataFrame, or path to embedding data file
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field
            input_neg_key: Key for negative samples field

        Returns:
            List of dict in reranker format
        """
        # Load data if file path is provided
        if isinstance(data, str):
            LOG.info(f"Loading embedding data from {data}...")
            data = self._load_embedding_data(data)

        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        LOG.info(f"Converting {len(dataframe)} embedding samples to reranker format...")

        # Collect all negatives for padding if needed
        all_negatives = []
        for neg_list in dataframe[input_neg_key].tolist():
            if isinstance(neg_list, list):
                all_negatives.extend(neg_list)
            elif neg_list:
                all_negatives.append(neg_list)

        results = []
        for _, row in dataframe.iterrows():
            query = row.get(input_query_key, "")
            pos = row.get(input_pos_key, [])
            neg = row.get(input_neg_key, [])

            if not query:
                continue

            # Ensure pos is a list
            if not isinstance(pos, list):
                pos = [pos] if pos else []

            # Ensure neg is a list and adjust count
            if not isinstance(neg, list):
                neg = [neg] if neg else []

            # Adjust negative count
            if len(neg) < self.adjust_neg_count and all_negatives:
                # Pad with random negatives from corpus
                import random
                additional = random.sample(
                    all_negatives,
                    min(self.adjust_neg_count - len(neg), len(all_negatives))
                )
                # Filter out any positives
                pos_set = set(pos)
                additional = [n for n in additional if n not in pos_set]
                neg = (neg + additional)[:self.adjust_neg_count]
            elif len(neg) > self.adjust_neg_count:
                neg = neg[:self.adjust_neg_count]

            # Build reranker format (no prompt/instruction)
            reranker_item = {
                "query": query,
                "pos": pos,
                "neg": neg,
            }
            results.append(reranker_item)

        LOG.info(f"Converted {len(results)} samples to reranker format.")

        # Save to file if specified
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved reranker data to {output_path}")

        return results

