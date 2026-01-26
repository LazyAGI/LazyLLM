"""KBC Chunk Generator Batch operator"""
import os
import json
import pandas as pd
from lazyllm import LOG
from ...base_data import DataOperatorRegistry


@DataOperatorRegistry.register(one_item=False, tag='knowledge_cleaning')
class KBCChunkGeneratorBatch:
    """
    Batch text splitting tool supporting token/sentence/semantic/recursive chunking.
    批量文本分割工具，支持词/句/语义/递归分块。
    """

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            split_method: str = "token",
            min_tokens_per_chunk: int = 100,
            tokenizer_name: str = "bert-base-uncased",
    ):
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
                "批量文本分割工具，",
                "支持词/句/语义/递归分块，",
                "可配置块大小、重叠和最小块长度",
            )
        elif lang == "en":
            return (
                "Batch text segmentation tool",
                "supporting multiple chunking methods",
                "with configurable size and overlap."
            )

    def _load_text(self, text_paths):
        """Load text from file list"""
        texts = []
        for text_path in text_paths:
            if not os.path.exists(text_path):
                LOG.error(f"Input file not found: {text_path}")
                texts.append("")
            elif text_path.endswith('.txt') or text_path.endswith('.md') or text_path.endswith('.xml'):
                with open(text_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            elif text_path.endswith(('.json', '.jsonl')):
                with open(text_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) if text_path.endswith('.json') else [json.loads(line) for line in f]
                text_fields = ['text', 'content', 'body']
                for field in text_fields:
                    if isinstance(data, list) and len(data) > 0 and field in data[0]:
                        texts.append("\n".join([item[field] for item in data]))
                        break
                    elif isinstance(data, dict) and field in data:
                        texts.append(data[field])
                        break
            else:
                LOG.error(f"Unsupported file format for {text_path}")
                texts.append("")
        return texts

    def __call__(
            self,
            data,
            input_key: str = "text_path",
            output_key: str = "chunk_path",
    ):
        """
        Perform batch text splitting and save results to files.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input text file paths
            output_key: Key for output chunk file paths

        Returns:
            List of dict with chunk paths added
        """
        self._ensure_initialized()

        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        text_paths = dataframe[input_key].tolist()
        texts = self._load_text(text_paths)
        output_paths = []

        for i, text in enumerate(texts):
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
                    for j in range(0, len(words), words_per_chunk):
                        chunk_text = ' '.join(words[j:j + words_per_chunk])
                        chunks.extend(self.chunker(chunk_text))

                json_chunks = [{
                    "raw_chunk": chunk.text,
                } for chunk in chunks]

                output_dir = "/".join([os.path.dirname(text_paths[i]), "extract"])
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(text_paths[i]))[0] + '_chunk.json'
                output_path = os.path.join(output_dir, file_name)
                output_paths.append(output_path)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(json_chunks, f, ensure_ascii=False, indent=4)
                LOG.info(f"Successfully split {text_paths[i]} into {len(chunks)} chunks.")
            else:
                output_paths.append("")

        dataframe[output_key] = output_paths
        LOG.info(f"Successfully split text for {len(text_paths)} files.")
        return dataframe.to_dict('records')

