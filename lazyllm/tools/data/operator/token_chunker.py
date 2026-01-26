from ..base_data import DataOperatorRegistry
from lazyllm import LOG
import re
import uuid
from datetime import datetime


@DataOperatorRegistry.register
class TokenChunker:
    def __init__(self, tokenizer, input_key='content', max_tokens=1024, min_tokens=200):
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

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

    def __call__(self, data):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        orig_meta = data.get('meta_data', '')

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
