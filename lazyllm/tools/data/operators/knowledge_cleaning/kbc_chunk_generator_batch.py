import os
import re
import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register


# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadText(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = 'text_path',
        **kwargs,
    ) -> dict:
        text_path = data.get(input_key, '')
        if not text_path:
            return {**data, '_text_content': '', '_load_error': 'Empty text path'}

        if not os.path.exists(text_path):
            LOG.error(f'Input file not found: {text_path}')
            return {
                **data,
                '_text_content': '',
                '_load_error': f'File not found: {text_path}',
            }

        try:
            if text_path.endswith(('.txt', '.md', '.xml')):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {**data, '_text_content': text}

            if text_path.endswith(('.json', '.jsonl')):
                with open(text_path, 'r', encoding='utf-8') as f:
                    if text_path.endswith('.json'):
                        file_data = json.load(f)
                    else:
                        file_data = [json.loads(line) for line in f]

                text_fields = ['text', 'content', 'body']
                for field in text_fields:
                    if isinstance(file_data, list) and file_data and field in file_data[0]:
                        text = '\n'.join(item[field] for item in file_data)
                        return {**data, '_text_content': text}
                    if isinstance(file_data, dict) and field in file_data:
                        text = file_data[field]
                        return {**data, '_text_content': text}

                LOG.error(f'No text field found in {text_path}')
                return {
                    **data,
                    '_text_content': '',
                    '_load_error': 'No text field found',
                }

            LOG.error(f'Unsupported file format for {text_path}')
            return {
                **data,
                '_text_content': '',
                '_load_error': 'Unsupported format',
            }

        except Exception as e:
            LOG.error(f'Error loading {text_path}: {e}')
            return {
                **data,
                '_text_content': '',
                '_load_error': str(e),
            }


class KBCChunkText(kbc):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        split_method: str = 'token',
        tokenizer_name: str = 'bert-base-uncased',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.tokenizer_name = tokenizer_name
        self._chunker = None
        self._tokenizer = None

    def _ensure_initialized(self):
        if self._tokenizer is None:
            try:
                from lazyllm.thirdparty.transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self._chunker = self._initialize_chunker()
            except ImportError as e:
                LOG.error(f'Missing dependencies: {e}')
                raise

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _chunk_by_token(self, text: str) -> List[str]:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return [text] if text.strip() else []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
        return chunks if chunks else [text]

    def _chunk_by_sentence(self, text: str) -> List[str]:
        # Sentence boundaries: . ! ? and common CJK terminators
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in parts if s.strip()]
        if not sentences:
            return [text] if text.strip() else []
        out = []
        current = []
        current_tokens = 0
        for sent in sentences:
            n = self._count_tokens(sent)
            if current_tokens + n > self.chunk_size and current:
                out.append(' '.join(current))
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current):
                    overlap_tokens += self._count_tokens(s)
                    if overlap_tokens <= self.chunk_overlap:
                        overlap_sents.append(s)
                    else:
                        break
                current = list(reversed(overlap_sents))
                current_tokens = overlap_tokens
            current.append(sent)
            current_tokens += n
        if current:
            out.append(' '.join(current))
        return out

    def _chunk_by_recursive(self, text: str) -> List[str]:
        separators = ['\n\n', '\n', '. ', '。', '! ', '? ', ' ', '']

        def _split(tex: str, sep_index: int) -> List[str]:
            if sep_index >= len(separators):
                return [tex] if tex.strip() else []
            sep = separators[sep_index]
            if not sep:
                return [tex] if tex.strip() else []
            parts = [p.strip() for p in tex.split(sep) if p.strip()]
            if len(parts) <= 1:
                return _split(tex, sep_index + 1)
            result = []
            for p in parts:
                result.extend(_split(p, sep_index + 1))
            return result

        def _merge(parts: List[str]) -> List[str]:
            if not parts:
                return []
            out = []
            acc = []
            acc_tokens = 0
            for p in parts:
                n = self._count_tokens(p)
                if acc_tokens + n > self.chunk_size and acc:
                    out.append(('\n' if '\n' in acc[0] else ' ').join(acc))
                    overlap_acc = []
                    overlap_tokens = 0
                    for x in reversed(acc):
                        overlap_tokens += self._count_tokens(x)
                        if overlap_tokens <= self.chunk_overlap:
                            overlap_acc.append(x)
                        else:
                            break
                    acc = list(reversed(overlap_acc))
                    acc_tokens = overlap_tokens
                acc.append(p)
                acc_tokens += n
            if acc:
                out.append(('\n' if '\n' in acc[0] else ' ').join(acc))
            return out

        parts = _split(text, 0)
        return _merge(parts)

    def _chunk_by_semantic(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return [text] if text.strip() else []
        out = []
        acc = []
        acc_tokens = 0
        for para in paragraphs:
            n = self._count_tokens(para)
            if acc_tokens + n > self.chunk_size and acc:
                out.append('\n\n'.join(acc))
                overlap_acc = []
                overlap_tokens = 0
                for x in reversed(acc):
                    overlap_tokens += self._count_tokens(x)
                    if overlap_tokens <= self.chunk_overlap:
                        overlap_acc.append(x)
                    else:
                        break
                acc = list(reversed(overlap_acc))
                acc_tokens = overlap_tokens
            acc.append(para)
            acc_tokens += n
        if acc:
            out.append('\n\n'.join(acc))
        return out

    def _initialize_chunker(self):
        if self.split_method == 'token':
            return lambda t: self._chunk_by_token(t)
        if self.split_method == 'sentence':
            return lambda t: self._chunk_by_sentence(t)
        if self.split_method == 'semantic':
            return lambda t: self._chunk_by_semantic(t)
        if self.split_method == 'recursive':
            return lambda t: self._chunk_by_recursive(t)
        raise ValueError(f'Unsupported split method: {self.split_method}')

    def forward(
        self,
        data: dict,
        **kwargs,
    ) -> dict:
        text = data.get('_text_content', '')
        if not text:
            return {**data, '_chunks': []}

        self._ensure_initialized()

        try:
            tokens = self._tokenizer.encode(text)
            total_tokens = len(tokens)
            max_tokens = self._tokenizer.model_max_length

            if total_tokens <= max_tokens:
                chunk_texts = self._chunker(text)
            else:
                x = (total_tokens + max_tokens - 1) // max_tokens
                words = text.split()
                words_per_chunk = (len(words) + x - 1) // x

                chunk_texts = []
                for j in range(0, len(words), words_per_chunk):
                    chunk_text = ' '.join(words[j:j + words_per_chunk])
                    chunk_texts.extend(self._chunker(chunk_text))

            LOG.info(f'Split text into {len(chunk_texts)} chunks.')
            return {**data, '_chunks': chunk_texts}

        except Exception as e:
            LOG.error(f'Error chunking text: {e}')
            return {**data, '_chunks': [], '_chunk_error': str(e)}


class KBCSaveChunks(kbc):
    def __init__(self, output_dir: Optional[str] = None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_dir = output_dir

    def forward(
        self,
        data: dict,
        input_key: str = 'text_path',
        output_key: str = 'chunk_path',
        **kwargs,
    ) -> dict:
        chunks = data.get('_chunks', [])
        text_path = data.get(input_key, '')

        result = data.copy()

        if not chunks:
            LOG.warning(f'No chunks to save for {text_path}')
            result[output_key] = ''
            for key in ['_text_content', '_load_error', '_chunks', '_chunk_error']:
                result.pop(key, None)
            return result

        try:
            # Determine output directory
            if self.output_dir:
                # Use specified output directory, preserving relative structure
                abs_text_path = os.path.abspath(text_path)
                abs_cwd = os.path.abspath(os.getcwd())

                if abs_text_path.startswith(abs_cwd):
                    rel_path = os.path.relpath(os.path.dirname(abs_text_path), abs_cwd)
                else:
                    rel_path = os.path.dirname(abs_text_path).lstrip('/')

                output_dir = os.path.join(self.output_dir, rel_path)
            else:
                # Default: save to 'extract' subdirectory
                output_dir = os.path.join(os.path.dirname(text_path), 'extract')

            os.makedirs(output_dir, exist_ok=True)

            file_name = os.path.splitext(os.path.basename(text_path))[0] + '_chunk.json'
            output_path = os.path.join(output_dir, file_name)

            json_chunks = [{'raw_chunk': chunk} for chunk in chunks]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_chunks, f, ensure_ascii=False, indent=4)

            LOG.info(f'Saved {len(chunks)} chunks to {output_path}')

            result[output_key] = output_path
            for key in ['_text_content', '_load_error', '_chunks', '_chunk_error']:
                result.pop(key, None)

            return result

        except Exception as e:
            LOG.error(f'Error saving chunks: {e}')
            result[output_key] = ''
            for key in ['_text_content', '_load_error', '_chunks', '_chunk_error']:
                result.pop(key, None)
            return result
