from typing import Any

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.kbcleaning import DocRefinementPrompt


def _text_from_json_formatter_response(response: Any) -> str:
    if isinstance(response, dict):
        if 'text' not in response:
            LOG.warning(
                'Cleaned LLM JSON must contain key "text"; got keys: %s',
                list(response.keys()),
            )
            return ''
        val = response['text']
        return val if isinstance(val, str) else str(val)

    if isinstance(response, list):
        if not response:
            LOG.warning('Cleaned LLM JSON list is empty; expected one object with "text".')
            return ''
        if len(response) > 1:
            LOG.warning(
                'Cleaned LLM reply contained %d JSON objects; only the first is used.',
                len(response),
            )
        return _text_from_json_formatter_response(response[0])

    if isinstance(response, str):
        return response

    return str(response)

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCGenerateCleanedTextSingle(kbc):

    def __init__(self,
                 llm=None,
                 lang: str = 'en',
                 input_key: str = 'raw_chunk',
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompts = DocRefinementPrompt(lang=lang)
        if llm is not None:
            system_prompt = self.prompts.build_system_prompt()
            # Use JsonFormatter: model must return valid JSON as instructed by DocRefinementPrompt
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        raw_content = data.get(self.input_key, '')
        if not raw_content:
            return {**data, '_cleaned_response': raw_content}

        # Build prompt for the raw content
        user_prompt = self.prompts.build_prompt(raw_content)

        try:
            # Call LLM (system prompt and formatter already set in __init__)
            response = self._llm_serve(user_prompt)
            return {**data, '_cleaned_response': response}
        except Exception as e:
            LOG.warning(f'Failed to clean text: {e}')
            # Use raw content as fallback
            return {**data, '_cleaned_response': raw_content}


@data_register('data.kbc', rewrite_func='forward', _concurrency_mode='process')
def extract_cleaned_content_single(
    data: dict,
    output_key: str = 'cleaned_chunk',
) -> dict:
    response = data.get('_cleaned_response', '')
    text = _text_from_json_formatter_response(response)

    # Extract content between tags
    if '<cleaned_start>' in text and '<cleaned_end>' in text:
        try:
            cleaned_text = text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
        except IndexError:
            cleaned_text = text.strip()
    else:
        cleaned_text = text.strip()

    result = data.copy()
    result[output_key] = cleaned_text
    # Clean intermediate fields
    for key in ['_cleaned_response']:
        result.pop(key, None)
    return result
