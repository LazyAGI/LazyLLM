"""
Reranker Query Generator Operator

This operator generates high-quality queries from document passages for reranker model training.
该算子从文档段落生成高质量的查询，用于 Reranker 模型训练。
"""
import json
import pandas as pd
from typing import List, Optional
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.reranker_synthesis import RerankerQueryGeneratorPrompt

funcs = data_register.new_group('function')
classes = data_register.new_group('class')
class RerankerQueryGenerator(classes):
    def __init__(
            self,
            llm_serving=None,
            num_queries: int = 3,
            lang: str = "zh",
            difficulty_levels: Optional[List[str]] = None,
    ):
        self.llm_serving = llm_serving
        self.num_queries = num_queries
        self.lang = lang
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "RerankerQueryGenerator 算子用于从文档段落生成高质量查询。\n\n"
                "核心功能：\n"
                "- 使用 LLM 分析文档内容并生成不同难度的查询\n"
                "- 支持多难度级别查询生成（简单、中等、困难）\n"
                "- 自动构建 query-passage 训练对\n\n"
                "输入参数：\n"
                "- input_key: 输入文档段落字段名（默认：'passage'）\n"
                "- output_query_key: 输出查询字段名（默认：'query'）\n\n"
                "输出：包含生成查询的数据列表"
            )
        else:
            return (
                "RerankerQueryGenerator generates high-quality queries from document passages.\n\n"
                "Features:\n"
                "- Uses LLM to analyze document content and generate queries of varying difficulty\n"
                "- Supports multiple difficulty levels (easy, medium, hard)\n"
                "- Automatically builds query-passage training pairs"
            )

    def _clean_json_block(self, text: str) -> str:
        """Clean JSON code block markers from LLM output."""
        return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _generate_from_llm(self, user_prompts: List[str], system_prompt: str = "") -> List[str]:
        """Call LLM serving to generate responses."""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def __call__(
            self,
            data,
            input_key: str = "passage",
            output_query_key: str = "query",
    ):
        """
        Generate queries for each passage in the data.

        Args:
            data: List of dict or pandas DataFrame containing passages
            input_key: Key for input passage field
            output_query_key: Key for output query field

        Returns:
            List of dict with generated queries (expanded rows)
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        LOG.info(f"Generating queries for {len(dataframe)} passages...")

        # Build prompts
        prompt_template = RerankerQueryGeneratorPrompt(lang=self.lang)
        system_prompt = prompt_template.build_system_prompt()

        user_prompts = []
        for passage in dataframe[input_key].tolist():
            user_prompts.append(prompt_template.build_prompt(
                passage=passage,
                num_queries=self.num_queries,
                difficulty_levels=self.difficulty_levels
            ))

        # Generate queries using LLM
        responses = self._generate_from_llm(user_prompts, system_prompt)

        # Parse responses and expand rows
        expanded_rows = []
        for idx, (row, response) in enumerate(zip(dataframe.to_dict('records'), responses)):
            try:
                parsed = json.loads(self._clean_json_block(response))
                queries = parsed if isinstance(parsed, list) else parsed.get("queries", [])

                for query_item in queries:
                    if isinstance(query_item, dict):
                        query = query_item.get("query", "")
                        difficulty = query_item.get("difficulty", "medium")
                    else:
                        query = str(query_item)
                        difficulty = "medium"

                    if query.strip():
                        new_row = row.copy()
                        new_row[output_query_key] = query.strip()
                        new_row["difficulty"] = difficulty
                        new_row["pos"] = [row[input_key]]  # Positive sample is the source passage
                        expanded_rows.append(new_row)

            except Exception as e:
                LOG.warning(f"Failed to parse response at idx={idx}: {e}")
                continue

        LOG.info(f"Generated {len(expanded_rows)} query-passage pairs from {len(dataframe)} passages.")
        return expanded_rows

