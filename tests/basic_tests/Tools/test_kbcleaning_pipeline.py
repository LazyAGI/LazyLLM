"""Test cases for Knowledge Cleaning pipelines"""
import pytest
from lazyllm.tools.data import (
    KBCleaningPipeline,
    KBCleaningBatchPipeline,
)


class TestKBCleaningPipeline:
    """Test KBCleaningPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = KBCleaningPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.lang == "en"
        assert pipeline.converter is not None
        assert pipeline.chunker is not None
        assert pipeline.cleaner is not None
        assert pipeline.qa_generator is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = KBCleaningPipeline(
            llm_serving=None,
            lang="zh",
            intermediate_dir="./custom_cache",
            chunk_size=256,
            split_method="sentence",
            tokenizer_name="bert-base-chinese",
            use_api=False
        )
        assert pipeline is not None
        assert pipeline.lang == "zh"
        assert pipeline.converter is not None
        assert pipeline.chunker is not None

    def test_init_with_api(self):
        """Test initialization with API converter"""
        pipeline = KBCleaningPipeline(use_api=True)
        assert pipeline is not None
        # Should use API converter
        assert pipeline.converter is not None


class TestKBCleaningBatchPipeline:
    """Test KBCleaningBatchPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = KBCleaningBatchPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.lang == "en"
        assert pipeline.converter is not None
        assert pipeline.chunker is not None
        assert pipeline.cleaner is not None
        assert pipeline.qa_generator is not None
        assert pipeline.qa_extractor is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = KBCleaningBatchPipeline(
            llm_serving=None,
            lang="zh",
            intermediate_dir="./custom_batch_cache",
            chunk_size=1024,
            split_method="recursive"
        )
        assert pipeline is not None
        assert pipeline.lang == "zh"
        assert pipeline.converter is not None
        assert pipeline.chunker is not None
