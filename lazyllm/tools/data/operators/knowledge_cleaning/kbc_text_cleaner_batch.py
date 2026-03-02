import json
import os
from typing import Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts.kbcleaning import DocRefinementPrompt

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadRAWChunkFile(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = 'chunk_path',
        **kwargs
    ) -> dict:
        chunk_path = data.get(input_key, '')
        if not chunk_path or not os.path.exists(chunk_path):
            LOG.warning(f'Invalid chunk path: {chunk_path}')
            return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

        try:
            if chunk_path.endswith('.json'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
            elif chunk_path.endswith('.jsonl'):
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    file_data = [json.loads(line) for line in f]
            else:
                LOG.warning(f'Unsupported file format: {chunk_path}')
                return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

            if not file_data or 'raw_chunk' not in file_data[0]:
                LOG.warning(f"'raw_chunk' field not found in: {chunk_path}")
                return {**data, '_chunks_data': [], '_chunk_path': chunk_path}

            return {**data, '_chunks_data': file_data, '_chunk_path': chunk_path}

        except Exception as e:
            LOG.error(f'Error loading chunk file {chunk_path}: {e}')
            return {**data, '_chunks_data': [], '_chunk_path': chunk_path}


class KBCGenerateCleanedText(kbc):
    def __init__(self, llm=None, lang: str = 'en', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompts = DocRefinementPrompt(lang=lang)
        if llm is not None:
            # Note: DocRefinementPrompt may not have system prompt, use empty string
            system_prompt = getattr(self.prompts, 'build_system_prompt', lambda: '')()
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

        chunks_data = data.get('_chunks_data', [])
        if not chunks_data:
            return {**data, '_cleaned_results': []}

        cleaned_results = []
        for item in chunks_data:
            raw_chunk = item.get('raw_chunk', '')
            if not raw_chunk:
                continue

            # Build prompt for this chunk
            user_prompt = self.prompts.build_prompt(raw_chunk)

            try:
                # Call LLM (system prompt and formatter already set in __init__)
                response = self._llm_serve(user_prompt)

                cleaned_results.append({
                    'response': response,
                    'raw_chunk': raw_chunk,
                    'original_item': item
                })
            except Exception as e:
                LOG.warning(f'Failed to clean text: {e}')
                # Use raw chunk as fallback
                cleaned_results.append({
                    'response': raw_chunk,
                    'raw_chunk': raw_chunk,
                    'original_item': item
                })

        return {**data, '_cleaned_results': cleaned_results}


@data_register('data.kbc', rewrite_func='forward', _concurrency_mode='process')
def extract_cleaned_content(data: dict) -> dict:
    cleaned_results = data.get('_cleaned_results', [])
    if not cleaned_results:
        return {**data, '_cleaned_chunks': []}

    cleaned_chunks = []
    for result in cleaned_results:
        response = result.get('response', '')
        raw_chunk = result.get('raw_chunk', '')
        original_item = result.get('original_item', {})

        # Handle different response types from JsonFormatter
        if isinstance(response, dict):
            # JsonFormatter returned a dict, extract text field or convert to string
            text = response.get('text', '') or response.get('content', '') or str(response)
        elif isinstance(response, list):
            # JsonFormatter returned a list, join or take first item
            text = response[0] if response else ''
            if isinstance(text, dict):
                text = text.get('text', '') or text.get('content', '') or str(text)
        elif isinstance(response, str):
            # JsonFormatter failed to parse, use as-is
            text = response
        else:
            text = str(response)

        # Extract content between tags
        if '<cleaned_start>' in text and '<cleaned_end>' in text:
            try:
                cleaned_text = text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
            except IndexError:
                cleaned_text = text.strip()
        else:
            cleaned_text = text.strip()

        cleaned_chunks.append({
            'raw_chunk': raw_chunk,
            'cleaned_chunk': cleaned_text,
            'original_item': original_item
        })

    return {**data, '_cleaned_chunks': cleaned_chunks}


def _clean_save_result(result: dict, output_key: str, output_path: str = '') -> dict:
    result[output_key] = output_path
    for key in ['_chunks_data', '_chunk_path', '_cleaned_results', '_cleaned_chunks']:
        result.pop(key, None)
    return result


def _build_json_items(cleaned_chunks: list) -> list:
    return [{
        'raw_chunk': item['raw_chunk'],
        'cleaned_chunk': item['cleaned_chunk']
    } for item in cleaned_chunks]


def _get_save_output_path(chunk_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        abs_chunk_path = os.path.abspath(chunk_path)
        abs_cwd = os.path.abspath(os.getcwd())
        if abs_chunk_path.startswith(abs_cwd):
            rel_path = os.path.relpath(abs_chunk_path, abs_cwd)
        else:
            rel_path = abs_chunk_path.lstrip('/')

        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        base, ext = os.path.splitext(chunk_path)
        if base.endswith('_cleaned'):
            counter = 1
            output_path = f'{base}_v{counter}{ext}'
            while os.path.exists(output_path):
                counter += 1
                output_path = f'{base}_v{counter}{ext}'
        else:
            output_path = f'{base}_cleaned{ext}'
    return output_path


class KBCSaveCleaned(kbc):
    def __init__(self, output_dir: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_dir = output_dir

    def forward(self, data: dict, output_key: str = 'cleaned_chunk_path', **kwargs) -> dict:
        cleaned_chunks = data.get('_cleaned_chunks', [])
        chunk_path = data.get('_chunk_path', '')
        result = data.copy()

        if not chunk_path:
            return _clean_save_result(result, output_key)

        if not cleaned_chunks:
            LOG.warning(f'No cleaned chunks to save for {chunk_path}')
            return _clean_save_result(result, output_key, chunk_path)

        try:
            json_items = _build_json_items(cleaned_chunks)
            output_path = _get_save_output_path(chunk_path, self.output_dir)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_items, f, ensure_ascii=False, indent=4)

            LOG.info(f'Successfully saved cleaned chunks to {output_path}')
            return _clean_save_result(result, output_key, output_path)

        except Exception as e:
            LOG.error(f'Error saving cleaned chunks: {e}')
            return _clean_save_result(result, output_key)
