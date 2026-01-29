"""
Knowledge Cleaning Pipeline for LazyLLM

This pipeline demonstrates how to use Knowledge Cleaning operators for:
1. Converting files/URLs to Markdown
2. Chunking text into smaller pieces
3. Cleaning text content
4. Generating multi-hop QA pairs

Example usage:
    from lazyllm.tools.data.pipeline import KBCleaningPipeline

    # Create pipeline with LLM serving
    pipeline = KBCleaningPipeline(llm_serving=your_llm_serving)

    # Run with input data
    data = [{"source": "/path/to/document.pdf"}]
    results = pipeline.forward(data)
"""
import pandas as pd
from lazyllm import LOG
from ..operators.knowledge_cleaning import (
    KBCChunkGenerator,
    KBCChunkGeneratorBatch,
    KBCTextCleaner,
    KBCTextCleanerBatch,
    FileOrURLToMarkdownConverterBatch,
    FileOrURLToMarkdownConverterAPI,
    KBCMultiHopQAGeneratorBatch,
    QAExtractor,
)
from ..base_data import data_register
funcs = data_register.new_group('function')
classes = data_register.new_group('class')
class KBCleaningPipeline(classes):
    """
    Knowledge Cleaning Pipeline for processing documents and generating QA pairs.
    知识清洗流水线：处理文档并生成 QA 对。

    This pipeline includes:
    1. FileOrURLToMarkdownConverter: Convert files/URLs to Markdown
    2. KBCChunkGenerator: Split text into chunks
    3. KBCTextCleaner: Clean text content
    4. KBCMultiHopQAGenerator: Generate multi-hop QA pairs

    Args:
        llm_serving: LLM serving instance for text generation
        lang: Language for processing ("en" or "zh")
        intermediate_dir: Directory for intermediate files
        chunk_size: Size of text chunks (default: 512)
        split_method: Chunking method ("token", "sentence", "semantic", "recursive")
        tokenizer_name: Tokenizer name for chunking
        use_api: Whether to use API for file conversion
    """

    def __init__(
            self,
            llm_serving=None,
            lang: str = "en",
            intermediate_dir: str = "./kb_cleaning_cache",
            chunk_size: int = 512,
            split_method: str = "token",
            tokenizer_name: str = "bert-base-uncased",
            use_api: bool = False,
    ):
        self.llm_serving = llm_serving
        self.lang = lang

        # Step 1: File/URL to Markdown converter
        if use_api:
            self.converter = FileOrURLToMarkdownConverterAPI(
                intermediate_dir=intermediate_dir,
                mineru_backend="vlm",
            )
        else:
            self.converter = FileOrURLToMarkdownConverterBatch(
                intermediate_dir=intermediate_dir,
                mineru_backend="vlm-sglang-engine",
            )

        # Step 2: Chunk generator
        self.chunker = KBCChunkGenerator(
            split_method=split_method,
            chunk_size=chunk_size,
            tokenizer_name=tokenizer_name,
        )

        # Step 3: Text cleaner
        self.cleaner = KBCTextCleaner(
            llm_serving=llm_serving,
            lang=lang,
        )

        # Step 4: Multi-hop QA generator
        self.qa_generator = KBCMultiHopQAGeneratorBatch(
            llm_serving=llm_serving,
            lang=lang,
        )

    def forward(
            self,
            data,
            source_key: str = "source",
            run_conversion: bool = True,
            run_chunking: bool = True,
            run_cleaning: bool = True,
            run_qa_generation: bool = True,
    ):
        """
        Run the knowledge cleaning pipeline.

        Args:
            data: Input data (list of dict or DataFrame)
            source_key: Key for source file/URL field
            run_conversion: Whether to run file conversion
            run_chunking: Whether to run text chunking
            run_cleaning: Whether to run text cleaning
            run_qa_generation: Whether to run QA generation

        Returns:
            Processed data with cleaned text and QA pairs
        """
        LOG.info("Starting Knowledge Cleaning Pipeline...")

        if isinstance(data, pd.DataFrame):
            results = data.to_dict('records')
        else:
            results = data

        # Step 1: Convert files/URLs to Markdown
        if run_conversion:
            LOG.info("Step 1: Converting files/URLs to Markdown...")
            results = self.converter(
                results,
                input_key=source_key,
                output_key="text_path",
            )

        # Step 2: Chunk text
        if run_chunking and results:
            LOG.info("Step 2: Chunking text...")
            results = self.chunker(
                results,
                input_key="text_path",
                output_key="raw_chunk",
            )

        # Step 3: Clean text
        if run_cleaning and results:
            LOG.info("Step 3: Cleaning text...")
            results = self.cleaner(
                results,
                input_key="raw_chunk",
                output_key="cleaned_chunk",
            )

        # Step 4: Generate QA pairs
        if run_qa_generation and results:
            LOG.info("Step 4: Generating multi-hop QA pairs...")
            # Convert to batch format if needed
            if isinstance(results, list) and results:
                # Save intermediate results for batch processing
                import json
                import tempfile
                import os

                temp_dir = tempfile.mkdtemp()
                chunk_path = os.path.join(temp_dir, "chunks.json")
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False)

                batch_data = [{"chunk_path": chunk_path}]
                self.qa_generator(
                    batch_data,
                    input_key="chunk_path",
                    output_key="enhanced_chunk_path",
                )

                # Reload results
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)

        LOG.info("Knowledge Cleaning Pipeline completed!")
        return results


class KBCleaningBatchPipeline(classes):
    """
    Batch Knowledge Cleaning Pipeline for processing multiple documents.
    批量知识清洗流水线。

    This pipeline processes files in batch mode, suitable for large-scale processing.

    Args:
        llm_serving: LLM serving instance
        lang: Language for processing
        intermediate_dir: Directory for intermediate files
        chunk_size: Size of text chunks
        split_method: Chunking method
        tokenizer_name: Tokenizer name
    """

    def __init__(
            self,
            llm_serving=None,
            lang: str = "en",
            intermediate_dir: str = "./kb_cleaning_batch_cache",
            chunk_size: int = 512,
            split_method: str = "token",
            tokenizer_name: str = "bert-base-uncased",
    ):
        self.llm_serving = llm_serving
        self.lang = lang

        # Step 1: File/URL to Markdown converter
        self.converter = FileOrURLToMarkdownConverterBatch(
            intermediate_dir=intermediate_dir,
            mineru_backend="vlm-sglang-engine",
        )

        # Step 2: Batch chunk generator
        self.chunker = KBCChunkGeneratorBatch(
            split_method=split_method,
            chunk_size=chunk_size,
            tokenizer_name=tokenizer_name,
        )

        # Step 3: Batch text cleaner
        self.cleaner = KBCTextCleanerBatch(
            llm_serving=llm_serving,
            lang=lang,
        )

        # Step 4: Multi-hop QA generator
        self.qa_generator = KBCMultiHopQAGeneratorBatch(
            llm_serving=llm_serving,
            lang=lang,
        )

        # Step 5: QA extractor
        self.qa_extractor = QAExtractor()

    def forward(
            self,
            data,
            source_key: str = "source",
            run_conversion: bool = True,
            run_chunking: bool = True,
            run_cleaning: bool = True,
            run_qa_generation: bool = True,
            run_qa_extraction: bool = True,
    ):
        """
        Run the batch knowledge cleaning pipeline.

        Args:
            data: Input data (list of dict or DataFrame)
            source_key: Key for source file/URL field
            run_conversion: Whether to run file conversion
            run_chunking: Whether to run text chunking
            run_cleaning: Whether to run text cleaning
            run_qa_generation: Whether to run QA generation
            run_qa_extraction: Whether to run QA extraction

        Returns:
            Processed data with cleaned text and QA pairs
        """
        LOG.info("Starting Batch Knowledge Cleaning Pipeline...")

        if isinstance(data, pd.DataFrame):
            results = data.to_dict('records')
        else:
            results = data

        # Step 1: Convert files/URLs to Markdown
        if run_conversion:
            LOG.info("Step 1: Converting files/URLs to Markdown...")
            results = self.converter(
                results,
                input_key=source_key,
                output_key="text_path",
            )

        # Step 2: Chunk text (batch mode)
        if run_chunking and results:
            LOG.info("Step 2: Chunking text (batch mode)...")
            results = self.chunker(
                results,
                input_key="text_path",
                output_key="chunk_path",
            )

        # Step 3: Clean text (batch mode)
        if run_cleaning and results:
            LOG.info("Step 3: Cleaning text (batch mode)...")
            results = self.cleaner(
                results,
                input_key="chunk_path",
                output_key="cleaned_chunk_path",
            )

        # Step 4: Generate QA pairs
        if run_qa_generation and results:
            LOG.info("Step 4: Generating multi-hop QA pairs...")
            results = self.qa_generator(
                results,
                input_key="cleaned_chunk_path",
                output_key="enhanced_chunk_path",
            )

        # Step 5: Extract QA pairs to Alpaca format
        if run_qa_extraction and results:
            LOG.info("Step 5: Extracting QA pairs to Alpaca format...")
            qa_results = self.qa_extractor(
                results,
                output_instruction_key="instruction",
                output_question_key="input",
                output_answer_key="output",
            )
            return qa_results

        LOG.info("Batch Knowledge Cleaning Pipeline completed!")
        return results


__all__ = [
    'KBCleaningPipeline',
    'KBCleaningBatchPipeline',
]

