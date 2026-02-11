import json
import random
from typing import List

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter

from ...base_data import data_register
from ...prompts.embedding_synthesis import EmbeddingQueryAugmentPrompt


# Get or create embedding group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


def _clean_json_block(text: str) -> str:
    return (
        text.strip()
        .removeprefix('```json')
        .removeprefix('```')
        .removesuffix('```')
        .strip()
    )


class EmbeddingQueryRewrite(embedding):
    def __init__(
        self,
        llm=None,
        num_augments: int = 2,
        lang: str = 'zh',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.num_augments = num_augments
        self.lang = lang
        self.prompt_template = EmbeddingQueryAugmentPrompt(lang=lang)

        # Initialize LLM serve with system prompt and formatter
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

    def forward(self, data: dict) -> List[dict]:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        query = data.get('query', '')
        if not query:
            return []

        user_prompt = self.prompt_template.build_prompt(
            query=query,
            num_rewrites=self.num_augments,
        )

        try:
            result = self._llm_serve(user_prompt)

            # Parse result from JsonFormatter
            if isinstance(result, list):
                rewrites = result
            elif isinstance(result, dict):
                rewrites = result.get('rewrites', [])
            elif isinstance(result, str):
                parsed = json.loads(_clean_json_block(result))
                rewrites = (
                    parsed
                    if isinstance(parsed, list)
                    else parsed.get('rewrites', [])
                )
            else:
                rewrites = []

            # Create augmented samples
            augmented = []
            for rewrite in rewrites:
                rewrite_str = str(rewrite).strip()
                if rewrite_str and rewrite_str != query:
                    new_row = data.copy()
                    new_row['query'] = rewrite_str
                    new_row['is_augmented'] = True
                    new_row['augment_method'] = 'query_rewrite'
                    augmented.append(new_row)

            return augmented

        except Exception as e:
            LOG.warning(f'Failed to rewrite query: {e}')
            return []


class EmbeddingAdjacentWordSwap(embedding):
    def __init__(
        self,
        num_augments: int = 2,
        **kwargs,
    ):
        # Rule-based operation, use process mode for CPU-bound tasks
        super().__init__(_concurrency_mode='process', **kwargs)
        self.num_augments = num_augments

    def forward(self, data: dict) -> List[dict]:
        query = data.get('query', '')
        if not query:
            return []

        augmented = []
        words = query.split()

        for _ in range(self.num_augments):
            if len(words) > 2:
                # Randomly swap two adjacent words as simple augmentation
                idx = random.randint(0, len(words) - 2)
                new_words = words.copy()
                new_words[idx], new_words[idx + 1] = (
                    new_words[idx + 1],
                    new_words[idx],
                )
                new_query = ' '.join(new_words)

                if new_query != query:
                    new_row = data.copy()
                    new_row['query'] = new_query
                    new_row['is_augmented'] = True
                    new_row['augment_method'] = 'synonym_replace'
                    augmented.append(new_row)
            else:
                # Keep original if too short
                break

        return augmented
