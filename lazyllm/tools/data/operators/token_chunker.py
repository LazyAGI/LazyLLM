import re
import os
import uuid
from datetime import datetime
from ..base_data import data_register
from lazyllm import LOG, config
from lazyllm.thirdparty import transformers


Chunker = data_register.new_group('chunker')


class TokenChunker(Chunker):
    def __init__(self, input_key='content', tokenizer_path=None, model_cache_dir=None,
                 max_tokens=1024, min_tokens=200, _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.tokenizer_path = tokenizer_path
        self.model_cache_dir = model_cache_dir or config['model_cache_dir']
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        default_model = 'Qwen/Qwen2.5-0.5B-Instruct'

        def _load(path):
            return transformers.AutoTokenizer.from_pretrained(
                path, cache_dir=self.model_cache_dir, trust_remote_code=True
            )

        model_or_path = self.tokenizer_path
        if not model_or_path or (os.path.isabs(model_or_path) and not os.path.exists(model_or_path)):
            model_or_path = default_model

        LOG.info(f'Loading tokenizer from: {model_or_path}')
        try:
            return _load(model_or_path)
        except Exception as e:
            if model_or_path != default_model:
                LOG.warning(f'Failed to load from {model_or_path}: {e}, falling back to default')
                return _load(default_model)
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
        return [''.join(s) for s in zip(sentences[0::2], sentences[1::2])]

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
