"""
Embedding Data Augmentor Operator

This operator augments embedding training data through various techniques.
该算子通过多种技术增强 Embedding 训练数据。

原始算法参考：
1. DataFlow/dataflow/operators/text_sft/generate/condor_generator.py (CondorGenerator - 使用 LLM 生成变体)
2. DataFlow/dataflow/operators/text_sft/generate/sft_generator_from_seed.py (基于种子生成)

核心思想：通过 LLM 改写、同义词替换等方法增强查询数据的多样性，
         提升 Embedding 模型的泛化能力。
"""
import json
import random
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryAugmentPrompt

# 复用已存在的 embedding 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingDataAugmentor(embedding):
    """
    Augment embedding training data through various techniques.
    通过多种技术增强 Embedding 训练数据。

    原始算法：参考 DataFlow CondorGenerator 和 SFTGeneratorSeed，使用 LLM 生成语义等价的查询变体。

    Augmentation techniques:
    - query_rewrite: Use LLM to rewrite queries with different expressions (参考 CondorGenerator)
    - back_translation: Translate and back-translate queries
    - synonym_replace: Replace words with synonyms
    - query_expansion: Expand queries with related terms

    Args:
        llm_serving: LLM serving instance for augmentation
        augment_methods: List of augmentation methods to apply
        num_augments: Number of augmented samples per original (default: 2)
        lang: Language for prompts ("zh" or "en", default: "zh")
        _concurrency_mode: Concurrency mode ('process', 'thread', 'single')
        _save_data: Whether to save intermediate data
    """

    def __init__(
            self,
            llm=None,
            augment_methods: Optional[List[str]] = None,
            num_augments: int = 2,
            lang: str = "zh",
            _concurrency_mode: str = 'single',
            _save_data: bool = True,
            **kwargs
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, _save_data=_save_data, **kwargs)
        self.llm = llm
        self.augment_methods = augment_methods or ["query_rewrite"]
        self.num_augments = num_augments
        self.lang = lang
        LOG.info(f"Initializing {self.__class__.__name__} with methods: {self.augment_methods}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingDataAugmentor 算子用于增强 Embedding 训练数据。\n\n"
                "原始算法：参考 DataFlow CondorGenerator（LLM 生成变体）\n"
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
                "Original Algorithm: Based on DataFlow CondorGenerator (LLM-based variants)\n"
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

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm is None:
            raise ValueError("LLM is not configured")
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        # prompter = lazyllm.ChatPrompter(system_prompt)
        # llm_serve = self.llm.prompt(prompter)
        # llm_serve.start()
        # LLM expects single string, need to iterate for batch
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results

    def _augment_query_rewrite(self, queries: List[str]) -> List[List[str]]:
        """Augment queries by rewriting with LLM.
        参考 CondorGenerator 的 LLM 生成方法。
        """
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

    def forward_batch_input(
            self,
            inputs: List[dict],
            input_query_key: str = "query",
            output_query_key: str = "query",
            keep_original: bool = True,
            **kwargs
    ) -> List[dict]:
        """
        Augment the training data.

        Args:
            inputs: List of dict
            input_query_key: Key for input query field
            output_query_key: Key for output query field
            keep_original: Whether to keep original samples in output

        Returns:
            List of dict with original and augmented samples
        """
        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Augmenting {len(inputs)} samples with methods: {self.augment_methods}")

        queries = [item.get(input_query_key, "") for item in inputs]
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
        for idx, item in enumerate(inputs):
            # Keep original if requested
            if keep_original:
                results.append(item.copy())

            # Add augmented samples
            for aug_query in all_augmented[idx]:
                if aug_query and aug_query != item.get(input_query_key):
                    new_row = item.copy()
                    new_row[output_query_key] = aug_query
                    new_row["is_augmented"] = True
                    results.append(new_row)

        original_count = len(inputs) if keep_original else 0
        augmented_count = len(results) - original_count
        LOG.info(f"Augmentation completed: {original_count} original + {augmented_count} augmented = {len(results)} total")
        return results
