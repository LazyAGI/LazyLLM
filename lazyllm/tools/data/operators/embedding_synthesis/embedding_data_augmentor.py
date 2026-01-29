"""
Embedding Data Augmentor Operator

This operator augments embedding training data through various techniques.
该算子通过多种技术增强 Embedding 训练数据。
"""
import json
import random
import pandas as pd
from typing import List, Optional
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryAugmentPrompt
funcs = data_register.new_group('function')
classes = data_register.new_group('class')

class EmbeddingDataAugmentor(classes):
    """
    Augment embedding training data through various techniques.
    通过多种技术增强 Embedding 训练数据。

    Augmentation techniques:
    - query_rewrite: Use LLM to rewrite queries with different expressions
    - back_translation: Translate and back-translate queries
    - synonym_replace: Replace words with synonyms
    - query_expansion: Expand queries with related terms

    Args:
        llm_serving: LLM serving instance for augmentation
        augment_methods: List of augmentation methods to apply
        num_augments: Number of augmented samples per original (default: 2)
        lang: Language for prompts ("zh" or "en", default: "zh")
    """

    def __init__(
            self,
            llm_serving=None,
            augment_methods: Optional[List[str]] = None,
            num_augments: int = 2,
            lang: str = "zh",
    ):
        self.llm_serving = llm_serving
        self.augment_methods = augment_methods or ["query_rewrite"]
        self.num_augments = num_augments
        self.lang = lang
        LOG.info(f"Initializing {self.__class__.__name__} with methods: {self.augment_methods}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingDataAugmentor 算子用于增强 Embedding 训练数据。\n\n"
                "核心功能：\n"
                "- 查询改写：使用 LLM 生成语义等价的不同表达\n"
                "- 同义词替换：替换查询中的词汇为同义词\n"
                "- 查询扩展：添加相关术语扩展查询\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- output_query_key: 增强后查询字段名（默认：'query'）\n\n"
                "输出：包含原始和增强样本的数据列表"
            )
        else:
            return (
                "EmbeddingDataAugmentor augments embedding training data.\n\n"
                "Features:\n"
                "- Query rewriting: Generate semantically equivalent expressions\n"
                "- Synonym replacement: Replace words with synonyms\n"
                "- Query expansion: Add related terms to queries\n\n"
                "Input:\n"
                "- input_query_key: Query field name (default: 'query')\n"
                "- output_query_key: Augmented query field name (default: 'query')"
            )

    def _clean_json_block(self, text: str) -> str:
        """Clean JSON code block markers from LLM output."""
        return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _generate_from_llm(self, user_prompts: List[str], system_prompt: str = "") -> List[str]:
        """Call LLM serving to generate responses."""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def _augment_query_rewrite(self, queries: List[str]) -> List[List[str]]:
        """Augment queries by rewriting with LLM."""
        prompt_template = EmbeddingQueryAugmentPrompt(lang=self.lang)
        system_prompt = prompt_template.build_system_prompt()

        user_prompts = [
            prompt_template.build_prompt(query=q, num_rewrites=self.num_augments)
            for q in queries
        ]

        responses = self._generate_from_llm(user_prompts, system_prompt)

        augmented_queries = []
        for response in responses:
            try:
                parsed = json.loads(self._clean_json_block(response))
                rewrites = parsed if isinstance(parsed, list) else parsed.get("rewrites", [])
                augmented_queries.append([str(r).strip() for r in rewrites if str(r).strip()])
            except Exception as e:
                LOG.warning(f"Failed to parse rewrite response: {e}")
                augmented_queries.append([])

        return augmented_queries

    def _augment_synonym_replace(self, queries: List[str]) -> List[List[str]]:
        """Augment queries by random synonym replacement (simple rule-based)."""
        # Simple character-level augmentation for Chinese, word-level for English
        augmented = []
        for query in queries:
            variants = []
            for _ in range(self.num_augments):
                # Simple shuffle-based augmentation (placeholder for more sophisticated methods)
                words = query.split()
                if len(words) > 2:
                    # Randomly swap two adjacent words
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
                    variants.append(" ".join(words))
                else:
                    variants.append(query)  # Keep original if too short
            augmented.append(variants)
        return augmented

    def __call__(
            self,
            data,
            input_query_key: str = "query",
            output_query_key: str = "query",
            keep_original: bool = True,
    ):
        """
        Augment the training data.

        Args:
            data: List of dict or pandas DataFrame
            input_query_key: Key for input query field
            output_query_key: Key for output query field
            keep_original: Whether to keep original samples in output

        Returns:
            List of dict with original and augmented samples
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        LOG.info(f"Augmenting {len(dataframe)} samples with methods: {self.augment_methods}")

        queries = dataframe[input_query_key].tolist()
        all_augmented = [[] for _ in queries]

        # Apply each augmentation method
        for method in self.augment_methods:
            if method == "query_rewrite":
                method_results = self._augment_query_rewrite(queries)
            elif method == "synonym_replace":
                method_results = self._augment_synonym_replace(queries)
            else:
                LOG.warning(f"Unknown augmentation method: {method}, skipping...")
                continue

            # Merge results
            for i, augments in enumerate(method_results):
                all_augmented[i].extend(augments)

        # Build output
        results = []
        for idx, row in dataframe.iterrows():
            row_dict = row.to_dict()

            # Keep original if requested
            if keep_original:
                results.append(row_dict.copy())

            # Add augmented samples
            for aug_query in all_augmented[idx]:
                if aug_query and aug_query != row_dict.get(input_query_key):
                    new_row = row_dict.copy()
                    new_row[output_query_key] = aug_query
                    new_row["is_augmented"] = True
                    results.append(new_row)

        original_count = len(dataframe) if keep_original else 0
        augmented_count = len(results) - original_count
        LOG.info(f"Augmentation completed: {original_count} original + {augmented_count} augmented = {len(results)} total")
        return results

