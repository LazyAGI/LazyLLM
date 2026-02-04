"""KBC Chunk Generator operator"""
import os
import json
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCChunkGenerator(kbc):
    """
    Lightweight text splitting tool supporting token/sentence/semantic/recursive chunking.
    CorpusTextSplitter是轻量级文本分割工具，支持词/句/语义/递归分块。
    """

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            split_method: str = "token",
            min_tokens_per_chunk: int = 100,
            tokenizer_name: str = "bert-base-uncased",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.tokenizer_name = tokenizer_name
        self._init_chunker()

    def _init_chunker(self):
        """Initialize tokenizer and chunker lazily"""
        self.tokenizer = None
        self.chunker = None

    def _ensure_initialized(self):
        """Ensure tokenizer and chunker are initialized"""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                from chonkie import TokenChunker, SentenceChunker, SemanticChunker, RecursiveChunker
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self.chunker = self._initialize_chunker()
            except ImportError as e:
                LOG.error(f"Missing dependencies: {e}")
                raise

    def _initialize_chunker(self):
        """Initialize the appropriate chunker based on method"""
        from chonkie import TokenChunker, SentenceChunker, SemanticChunker, RecursiveChunker

        if self.split_method == "token":
            return TokenChunker(
                tokenizer=self.tokenizer,
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

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "CorpusTextSplitter是轻量级文本分割工具，"
                "支持词/句/语义/递归分块，"
                "可配置块大小、重叠和最小块长度"
            )
        elif lang == "en":
            return (
                "CorpusTextSplitter is a lightweight text segmentation tool "
                "that supports multiple chunking methods "
                "(token/sentence/semantic/recursive) with configurable size and overlap, "
                "optimized for RAG applications."
            )
        else:
            return "Text splitting tool supporting multiple chunking methods."

    def _load_text(self, file_path) -> str:
        """Load text from input file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if file_path.endswith('.txt') or file_path.endswith('.md') or file_path.endswith('.xml'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif file_path.endswith(('.json', '.jsonl')):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f) if file_path.endswith('.json') else [json.loads(line) for line in f]
            text_fields = ['text', 'content', 'body']
            for field in text_fields:
                if isinstance(data, list) and len(data) > 0 and field in data[0]:
                    return "\n".join([item[field] for item in data])
                elif isinstance(data, dict) and field in data:
                    return data[field]

            raise ValueError("No text field found in JSON input")
        else:
            raise ValueError("Unsupported file format")

    def forward_batch_input(
            self,
            data: List[dict],
            input_key: str = 'text_path',
            output_key: str = "raw_chunk",
    ) -> List[dict]:
        """
        Perform text splitting.

        Args:
            data: List of dict
            input_key: Key for input text file paths
            output_key: Key for output chunks

        Returns:
            List of dict with chunks added
        """
        assert isinstance(data, list), "Input data must be a list of dict"
        self._ensure_initialized()

        text_paths = [item.get(input_key, "") for item in data]
        for input_path in text_paths:
            if not input_path or not os.path.exists(input_path):
                LOG.error(f"Invalid input file path: {input_path}")

        new_records = []
        for row_dict, text_path in zip(data, text_paths):
            if not text_path:
                continue
            text = self._load_text(text_path)
            if text:
                tokens = self.tokenizer.encode(text)
                total_tokens = len(tokens)
                max_tokens = self.tokenizer.model_max_length

                if total_tokens <= max_tokens:
                    chunks = self.chunker(text)
                else:
                    x = (total_tokens + max_tokens - 1) // max_tokens
                    words = text.split()
                    words_per_chunk = (len(words) + x - 1) // x

                    chunks = []
                    for i in range(0, len(words), words_per_chunk):
                        chunk_text = ' '.join(words[i:i + words_per_chunk])
                        chunks.extend(self.chunker(chunk_text))

                for chunk in chunks:
                    new_row = row_dict.copy()
                    new_row[output_key] = chunk.text
                    new_records.append(new_row)

        LOG.info(f"Successfully split text for {len(text_paths)} files.")
        return new_records
