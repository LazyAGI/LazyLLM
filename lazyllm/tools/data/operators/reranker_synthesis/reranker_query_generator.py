import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.reranker_synthesis import RerankerQueryGeneratorPrompt

# Get or create reranker group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'reranker' in LazyLLMRegisterMetaClass.all_clses['data']:
    reranker = LazyLLMRegisterMetaClass.all_clses['data']['reranker'].base
else:
    reranker = data_register.new_group('reranker')


def _clean_json_block(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class RerankerBuildQueryPrompt(reranker):
    def __init__(
        self,
        num_queries: int = 3,
        lang: str = "zh",
        difficulty_levels: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.num_queries = num_queries
        self.lang = lang
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self.prompt_template = RerankerQueryGeneratorPrompt(lang=lang)

    def forward(
        self,
        data: dict,
        input_key: str = "passage",
        **kwargs
    ) -> dict:
        passage = data.get(input_key, "")
        if not passage:
            return {**data, '_query_prompt': ''}

        user_prompt = self.prompt_template.build_prompt(
            passage=passage,
            num_queries=self.num_queries,
            difficulty_levels=self.difficulty_levels
        )

        return {**data, '_query_prompt': user_prompt}


class RerankerGenerateQueries(reranker):
    def __init__(
        self,
        llm_serving=None,
        lang: str = "zh",
        **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = RerankerQueryGeneratorPrompt(lang=lang)

        # Initialize LLM serve with system prompt and formatter
        if llm_serving is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm_serving.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError("LLM serving is not configured")

        user_prompt = data.get('_query_prompt', '')
        if not user_prompt:
            return {**data, '_query_response': ''}

        try:
            result = self._llm_serve(user_prompt)
            # JsonFormatter already parses JSON, handle both str and parsed result
            if isinstance(result, str):
                response = result
            else:
                response = json.dumps(result, ensure_ascii=False)
            return {**data, '_query_response': response}
        except Exception as e:
            LOG.warning(f"Failed to generate queries: {e}")
            return {**data, '_query_response': ''}


class RerankerParseQueries(reranker):
    def __init__(
        self,
        input_key: str = "passage",
        output_query_key: str = "query",
        **kwargs
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.input_key = input_key
        self.output_query_key = output_query_key

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> List[dict]:
        response = data.get('_query_response', '')
        if not response:
            return []

        passage = data.get(self.input_key, '')
        expanded_rows = []

        try:
            parsed = json.loads(_clean_json_block(response))
            queries = parsed if isinstance(parsed, list) else parsed.get("queries", [])

            for query_item in queries:
                if isinstance(query_item, dict):
                    query = query_item.get("query", "")
                    difficulty = query_item.get("difficulty", "medium")
                else:
                    query = str(query_item)
                    difficulty = "medium"

                if query.strip():
                    new_row = data.copy()
                    new_row[self.output_query_key] = query.strip()
                    new_row["difficulty"] = difficulty
                    new_row["pos"] = [passage]  # Positive sample is the source passage
                    # Clean up intermediate fields
                    new_row.pop('_query_prompt', None)
                    new_row.pop('_query_response', None)
                    expanded_rows.append(new_row)

        except Exception as e:
            LOG.warning(f"Failed to parse LLM response: {e}")
            return []

        return expanded_rows
