"""
Embedding Query Generator Operator

This operator generates high-quality queries from document passages for embedding model training.
该算子从文档段落生成高质量的查询，用于 Embedding 模型训练。

原始算法参考：DataFlow/dataflow/operators/core_text/generate/text2qa_generator.py (Text2QAGenerator)
核心思想：使用 LLM 从文档段落中生成相关的查询-段落对，用于训练 Embedding 模型。
"""
import json
from typing import List, Optional
import warnings
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryGeneratorPrompt

# 复用已存在的 embedding 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingQueryGenerator(embedding):
    """
    Generate queries from document passages for embedding training data.
    从文档段落生成查询，用于构建 Embedding 训练数据。

    This operator uses an LLM to generate relevant queries that could be
    answered by the given passage, creating query-passage pairs for training.

    原始算法：参考 DataFlow Text2QAGenerator，使用两阶段方法：
    1. 先生成 prompt/问题类型
    2. 再基于文档生成具体的 QA 对

    Args:
        llm_serving: LLM serving instance for query generation
        num_queries: Number of queries to generate per passage (default: 3)
        lang: Language for prompts ("zh" or "en", default: "zh")
        query_types: List of query types to generate (default: ["factual", "semantic", "inferential"])
        _concurrency_mode: Concurrency mode ('process', 'thread', 'single')
        _save_data: Whether to save intermediate data
    """

    def __init__(
            self,
            num_queries: int = 3,
            lang: str = "zh",
            query_types: Optional[List[str]] = None,
            llm = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.num_queries = num_queries
        self.lang = lang
        self.query_types = query_types or ["factual", "semantic", "inferential"]
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingQueryGenerator 算子用于从文档段落生成高质量查询。\n\n"
                "核心功能：\n"
                "- 使用 LLM 分析文档内容并生成相关查询\n"
                "- 支持生成多种类型的查询（事实型、语义型、推理型）\n"
                "- 自动构建 query-passage 训练对\n\n"
                "输入参数：\n"
                "- input_key: 输入文档段落字段名（默认：'passage'）\n"
                "- output_query_key: 输出查询字段名（默认：'query'）\n\n"
                "输出：包含生成查询的数据列表"
            )
        else:
            return (
                "EmbeddingQueryGenerator generates high-quality queries from document passages.\n\n"
                "Features:\n"
                "- Uses LLM to analyze document content and generate relevant queries\n"
                "- Supports multiple query types (factual, semantic, inferential)\n"
                "- Automatically builds query-passage training pairs\n\n"
                "Input:\n"
                "- input_key: Field name for input passages (default: 'passage')\n"
                "- output_query_key: Field name for output queries (default: 'query')"
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

    def forward_batch_input(
            self,
            inputs: List[dict],
            input_key: str = "passage",
            output_query_key: str = "query",
            **kwargs
    ) -> List[dict]:
        """
        Generate queries for each passage in the data.

        Args:
            inputs: List of dict containing passages
            input_key: Key for input passage field
            output_query_key: Key for output query field

        Returns:
            List of dict with generated queries (expanded rows)
        """
        assert isinstance(inputs, list), "inputs must be a list of dict"

        LOG.info(f"Generating queries for {len(inputs)} passages...")

        # Build prompts
        prompt_template = EmbeddingQueryGeneratorPrompt(lang=self.lang)
        system_prompt = prompt_template.build_system_prompt()

        user_prompts = []
        for item in inputs:
            passage = item.get(input_key, "")
            user_prompts.append(prompt_template.build_prompt(
                passage=passage,
                num_queries=self.num_queries,
                query_types=self.query_types
            ))

        # Generate queries using LLM
        responses = self._generate_from_llm(user_prompts, system_prompt)

        # Parse responses and expand rows
        expanded_rows = []
        for idx, (item, response) in enumerate(zip(inputs, responses)):
            try:
                parsed = json.loads(self._clean_json_block(response))
                queries = parsed if isinstance(parsed, list) else parsed.get("queries", [])

                for query_item in queries:
                    if isinstance(query_item, dict):
                        query = query_item.get("query", "")
                        query_type = query_item.get("type", "unknown")
                    else:
                        query = str(query_item)
                        query_type = "unknown"

                    if query.strip():
                        new_row = item.copy()
                        new_row[output_query_key] = query.strip()
                        new_row["query_type"] = query_type
                        new_row["pos"] = [item[input_key]]  # Positive sample is the source passage
                        expanded_rows.append(new_row)

            except Exception as e:
                LOG.warning(f"Failed to parse response at idx={idx}: {e}")
                continue

        LOG.info(f"Generated {len(expanded_rows)} query-passage pairs from {len(inputs)} passages.")
        return expanded_rows
