import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryGeneratorPrompt


# Get or create embedding group
if (
    'data' in LazyLLMRegisterMetaClass.all_clses
    and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']
):
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


def _clean_json_block(item: str) -> str:
    return (
        item.strip()
        .removeprefix('```json')
        .removeprefix('```')
        .removesuffix('```')
        .strip()
    )

class EmbeddingGenerateQueries(embedding):
    def __init__(
        self,
        llm=None,
        num_queries: int = 3,
        lang: str = 'zh',
        query_types: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = EmbeddingQueryGeneratorPrompt(lang=lang)
        self.num_queries = num_queries
        self.query_types = query_types or ['factual', 'semantic', 'inferential']
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = (
                llm.share()
                .prompt(system_prompt)
                .formatter(JsonFormatter())
            )
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        input_key: str = 'passage',
        **kwargs,
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        passage = data.get(input_key, '')
        if not passage:
            return {**data, '_query_response': ''}

        user_prompt = self.prompt_template.build_prompt(
            passage=passage,
            num_queries=self.num_queries,
            query_types=self.query_types,
        )
        if not user_prompt:
            return {**data, '_query_response': ''}

        try:
            result = self._llm_serve(user_prompt)

            if isinstance(result, str):
                response = result
            else:
                response = json.dumps(result, ensure_ascii=False)

            return {**data, '_query_response': response}

        except Exception as e:
            LOG.warning(f'Failed to generate queries: {e}')
            return {**data, '_query_response': ''}


class EmbeddingParseQueries(embedding):
    def __init__(
        self,
        input_key: str = 'passage',
        output_query_key: str = 'query',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.input_key = input_key
        self.output_query_key = output_query_key

    def forward(
        self,
        data: dict,
        **kwargs,
    ) -> List[dict]:
        response = data.get('_query_response', '')
        if not response:
            return []

        passage = data.get(self.input_key, '')
        expanded_rows = []

        try:
            parsed = json.loads(_clean_json_block(response))
            queries = (
                parsed if isinstance(parsed, list)
                else parsed.get('queries', [])
            )

            for query_item in queries:
                if isinstance(query_item, dict):
                    query = query_item.get('query', '')
                    query_type = query_item.get('type', 'unknown')
                else:
                    query = str(query_item)
                    query_type = 'unknown'

                if query.strip():
                    new_row = data.copy()
                    new_row[self.output_query_key] = query.strip()
                    new_row['query_type'] = query_type
                    new_row['pos'] = [passage]

                    new_row.pop('_query_prompt', None)
                    new_row.pop('_query_response', None)

                    expanded_rows.append(new_row)

        except Exception as e:
            LOG.warning(f'Failed to parse query response: {e}')
            return []

        return expanded_rows
