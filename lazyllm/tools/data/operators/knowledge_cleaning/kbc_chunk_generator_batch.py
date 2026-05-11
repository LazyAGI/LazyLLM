import os
import json
from typing import Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.thirdparty import chonkie, transformers
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
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
            self._chunker = self._initialize_chunker()

    def _initialize_chunker(self):
        if self.split_method == 'token':
            return chonkie.TokenChunker(
                tokenizer=self._tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        if self.split_method == 'sentence':
            return chonkie.SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        if self.split_method == 'semantic':
            return chonkie.SemanticChunker(
                chunk_size=self.chunk_size,
            )

        if self.split_method == 'recursive':
            return chonkie.RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

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
                chunks = self._chunker(text)
            else:
                x = (total_tokens + max_tokens - 1) // max_tokens
                words = text.split()
                words_per_chunk = (len(words) + x - 1) // x

                chunks = []
                for j in range(0, len(words), words_per_chunk):
                    chunk_text = ' '.join(words[j:j + words_per_chunk])
                    chunks.extend(self._chunker(chunk_text))

            chunk_texts = [chunk.text for chunk in chunks]
            LOG.info(f'Split text into {len(chunks)} chunks.')
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
