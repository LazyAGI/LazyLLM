import re
import os
import uuid
from datetime import datetime
from itertools import zip_longest
from ..base_data import data_register
from lazyllm import LOG, config
from lazyllm.thirdparty import transformers


Chunker = data_register.new_group('chunker')


class TokenChunker(Chunker):
    def __init__(self, input_key='content', model_path=None,
                 max_tokens=1024, min_tokens=200, _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.model_path = model_path
        self.tokenizer = self._load_tokenizer()

    def _try_load_tokenizer(self, path, is_local, default_cache_dir):
        if is_local or os.path.isdir(path) or os.path.isfile(path):
            return transformers.AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            )
        else:
            return transformers.AutoTokenizer.from_pretrained(
                path, cache_dir=default_cache_dir, trust_remote_code=True
            )

    def _try_load_from_config_path(self, default_model_name, default_cache_dir):
        try:
            config_model_path = config['model_path']
            if not config_model_path:
                return None
            if os.path.isdir(config_model_path):
                joined_path = os.path.join(config_model_path, default_model_name)
                if os.path.exists(joined_path):
                    LOG.info(f'Loading tokenizer from config model_path: {joined_path}')
                    try:
                        return self._try_load_tokenizer(joined_path, True, default_cache_dir)
                    except Exception as e:
                        LOG.warning(f'Failed to load from {joined_path}: {e}, trying cache directory')
            elif os.path.exists(config_model_path):
                LOG.info(f'Loading tokenizer from config model_path: {config_model_path}')
                try:
                    return self._try_load_tokenizer(config_model_path, True, default_cache_dir)
                except Exception as e:
                    LOG.warning(f'Failed to load from {config_model_path}: {e}, trying cache directory')
        except (KeyError, TypeError):
            pass
        return None

    def _try_load_from_cache(self, default_model_name, default_cache_dir):
        try:
            cache_model_path = os.path.join(default_cache_dir, default_model_name)
            if os.path.exists(cache_model_path):
                LOG.info(f'Loading tokenizer from cache directory: {cache_model_path}')
                return self._try_load_tokenizer(cache_model_path, True, default_cache_dir)
        except Exception:
            pass
        return None

    def _load_tokenizer(self):
        default_model = 'Qwen/Qwen2.5-0.5B-Instruct'
        default_model_name = 'qwen2.5-0.5b-instruct'
        model_or_path = self.model_path

        try:
            default_cache_dir = config['model_cache_dir']
        except (KeyError, TypeError):
            default_cache_dir = os.path.join(os.path.expanduser('~'), '.lazyllm', 'models')

        if model_or_path:
            try:
                is_local = os.path.isdir(model_or_path) or os.path.isfile(model_or_path)
                if is_local:
                    log_msg = f'Loading tokenizer from local path: {model_or_path}'
                else:
                    log_msg = f'Loading tokenizer from model: {model_or_path}'
                LOG.info(log_msg)
                return self._try_load_tokenizer(model_or_path, is_local, default_cache_dir)
            except Exception as e:
                LOG.warning(f'Failed to load from {model_or_path}: {e}, trying config model_path')

        if model_or_path is None:
            result = self._try_load_from_config_path(default_model_name, default_cache_dir)
            if result:
                return result

        result = self._try_load_from_cache(default_model_name, default_cache_dir)
        if result:
            return result

        LOG.info(f'Loading default tokenizer: {default_model} (will download to cache)')
        try:
            return self._try_load_tokenizer(default_model, False, default_cache_dir)
        except Exception as e:
            LOG.error(f'Failed to load default tokenizer: {e}')
            raise

    def _split_paragraphs(self, text):
        paragraphs = re.split(r'(\n{2,})', text)
        processed_paragraphs = []
        for i in range(0, len(paragraphs), 2):
            unit = paragraphs[i]
            if i + 1 < len(paragraphs):
                unit += paragraphs[i + 1]
            if unit:
                processed_paragraphs.append(unit)
        return processed_paragraphs

    def _split_sentences(self, text):
        sentences = re.split(r'([。！？\.!\?])', text)
        return [s for s in (''.join(filter(None, t)) for t in zip_longest(sentences[0::2], sentences[1::2])) if s]

    def _process_chunks(self, processed_paragraphs):
        chunks = []
        current_chunk_text_parts = []
        current_chunk_tokens = 0

        for p_text in processed_paragraphs:
            p_tokens = self.tokenizer.encode(p_text)

            if current_chunk_tokens + len(p_tokens) <= self.max_tokens:
                current_chunk_text_parts.append(p_text)
                current_chunk_tokens += len(p_tokens)
            else:
                if current_chunk_text_parts:
                    chunks.append(''.join(current_chunk_text_parts))

                if len(p_tokens) > self.max_tokens:
                    sentences = self._split_sentences(p_text)

                    sub_chunk_parts = []
                    sub_chunk_tokens = 0
                    for sent in sentences:
                        sent_tokens_count = len(self.tokenizer.encode(sent))
                        if sub_chunk_tokens + sent_tokens_count <= self.max_tokens:
                            sub_chunk_parts.append(sent)
                            sub_chunk_tokens += sent_tokens_count
                        else:
                            if sub_chunk_parts:
                                chunks.append(''.join(sub_chunk_parts))
                            sub_chunk_parts = [sent]
                            sub_chunk_tokens = sent_tokens_count

                    current_chunk_text_parts = sub_chunk_parts
                    current_chunk_tokens = sub_chunk_tokens
                else:
                    current_chunk_text_parts = [p_text]
                    current_chunk_tokens = len(p_tokens)

        if current_chunk_text_parts:
            final_chunk_text = ''.join(current_chunk_text_parts)
            final_chunk_tokens = len(self.tokenizer.encode(final_chunk_text))
            if len(chunks) > 0 and final_chunk_tokens < self.min_tokens:
                LOG.warning(f'Discarding small chunk (tokens: {final_chunk_tokens}, threshold: {self.min_tokens})')
            else:
                chunks.append(final_chunk_text)

        return chunks

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        orig_meta = data.get('meta_data', {})

        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = self._process_chunks(paragraphs)

        if not chunks and text:
            chunks = [text]

        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        total = len(chunks)

        return [
            {
                'uid': f'{ts}_{uuid.uuid4().hex}',
                'content': chunk,
                'meta_data': {
                    **orig_meta,
                    'index': idx,
                    'total': total,
                    'length': len(chunk),
                },
            }
            for idx, chunk in enumerate(chunks)
        ]
