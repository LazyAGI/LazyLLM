import os
import json
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadTextSingle(kbc):
    def __init__(self, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

    def forward(
        self,
        data: dict,
        input_key: str = "text_path",
        **kwargs
    ) -> dict:
        text_path = data.get(input_key, "")
        if not text_path:
            LOG.error("Empty text path")
            return {**data, '_text_content': ''}

        if not os.path.exists(text_path):
            LOG.error(f"Input file not found: {text_path}")
            return {**data, '_text_content': ''}

        try:
            if text_path.endswith('.txt') or text_path.endswith('.md') or text_path.endswith('.xml'):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {**data, '_text_content': text}

            elif text_path.endswith(('.json', '.jsonl')):
                with open(text_path, 'r', encoding='utf-8') as f:
                    if text_path.endswith('.json'):
                        file_data = json.load(f)
                    else:
                        file_data = [json.loads(line) for line in f]

                text_fields = ['text', 'content', 'body']
                for field in text_fields:
                    if isinstance(file_data, list) and len(file_data) > 0 and field in file_data[0]:
                        text = "\n".join([item[field] for item in file_data])
                        return {**data, '_text_content': text}
                    elif isinstance(file_data, dict) and field in file_data:
                        text = file_data[field]
                        return {**data, '_text_content': text}

                LOG.error(f"No text field found in {text_path}")
                return {**data, '_text_content': ''}

            else:
                LOG.error(f"Unsupported file format for {text_path}")
                return {**data, '_text_content': ''}

        except Exception as e:
            LOG.error(f"Error loading {text_path}: {e}")
            return {**data, '_text_content': ''}


class KBCChunkTextSingle(kbc):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        split_method: str = "token",
        tokenizer_name: str = "bert-base-uncased",
        **kwargs
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
                # from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self._chunker = self._initialize_chunker()
            except ImportError as e:
                LOG.error(f"Missing dependencies: {e}")
                raise

    def _initialize_chunker(self):
        from lazyllm.thirdparty.chonkie import TokenChunker, SentenceChunker, SemanticChunker, RecursiveChunker
        # from chonkie import TokenChunker, SentenceChunker, SemanticChunker, RecursiveChunker

        if self.split_method == "token":
            return TokenChunker(
                tokenizer=self._tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.split_method == "sentence":
            return SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.split_method == "semantic":
            return SemanticChunker(
                chunk_size=self.chunk_size,
            )
        elif self.split_method == "recursive":
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported split method: {self.split_method}")

    def forward(
        self,
        data: dict,
        **kwargs
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
                # Handle long text by splitting into smaller parts first
                x = (total_tokens + max_tokens - 1) // max_tokens
                words = text.split()
                words_per_chunk = (len(words) + x - 1) // x

                chunks = []
                for i in range(0, len(words), words_per_chunk):
                    chunk_text = ' '.join(words[i:i + words_per_chunk])
                    chunks.extend(self._chunker(chunk_text))

            chunk_texts = [chunk.text for chunk in chunks]
            LOG.info(f"Split text into {len(chunks)} chunks.")
            return {**data, '_chunks': chunk_texts}

        except Exception as e:
            LOG.error(f"Error chunking text: {e}")
            return {**data, '_chunks': []}


class KBCExpandChunks(kbc):
    def __init__(self, output_key: str = "raw_chunk", **kwargs):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.output_key = output_key

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> List[dict]:
        chunks = data.get('_chunks', [])
        
        if not chunks:
            return []

        new_records = []
        for chunk_text in chunks:
            new_row = data.copy()
            new_row[self.output_key] = chunk_text
            # Clean intermediate fields
            new_row.pop('_text_content', None)
            new_row.pop('_chunks', None)
            new_records.append(new_row)

        return new_records
