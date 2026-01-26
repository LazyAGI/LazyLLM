"""Test cases for Reranker Synthesis pipelines"""
import pytest
from lazyllm.tools.data import (
    RerankerSynthesisPipeline,
    RerankerFromEmbeddingPipeline,
    RerankerFineTunePipeline,
)


class TestRerankerSynthesisPipeline:
    """Test RerankerSynthesisPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = RerankerSynthesisPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.query_generator is not None
        assert pipeline.negative_miner is not None
        assert pipeline.formatter is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = RerankerSynthesisPipeline(
            llm_serving=None,
            num_queries=5,
            num_negatives=7,
            mining_strategy="bm25",
            output_format="flagreranker",
            lang="en",
            test_size=0.2,
            train_group_size=8,
        )
        assert pipeline is not None
        assert pipeline.lang == "en"
        assert pipeline.test_size == 0.2
        assert pipeline.train_group_size == 8

    def test_init_without_splitting(self):
        """Test initialization without train/test splitting"""
        pipeline = RerankerSynthesisPipeline(
            test_size=0,
        )
        assert pipeline.splitter is None

    def test_init_chinese(self):
        """Test initialization with Chinese language"""
        pipeline = RerankerSynthesisPipeline(lang="zh")
        assert pipeline.lang == "zh"

    def test_init_english(self):
        """Test initialization with English language"""
        pipeline = RerankerSynthesisPipeline(lang="en")
        assert pipeline.lang == "en"


class TestRerankerFromEmbeddingPipeline:
    """Test RerankerFromEmbeddingPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = RerankerFromEmbeddingPipeline()
        assert pipeline is not None
        assert pipeline.converter is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = RerankerFromEmbeddingPipeline(
            num_negatives=10,
            test_size=0.2,
        )
        assert pipeline.num_negatives == 10
        assert pipeline.test_size == 0.2

    def test_init_without_splitting(self):
        """Test initialization without splitting"""
        pipeline = RerankerFromEmbeddingPipeline(
            test_size=0,
        )
        assert pipeline.splitter is None


class TestRerankerFineTunePipeline:
    """Test RerankerFineTunePipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = RerankerFineTunePipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.rerank_model_path == "BAAI/bge-reranker-base"
        assert pipeline.synthesis_pipeline is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = RerankerFineTunePipeline(
            llm_serving=None,
            rerank_model_path="BAAI/bge-reranker-large",
            num_queries=5,
            num_negatives=7,
            mining_strategy="bm25",
            lang="en",
            per_device_batch_size=4,
            num_epochs=2,
            learning_rate=1e-5,
            ngpus=2,
            train_group_size=8,
        )
        assert pipeline.rerank_model_path == "BAAI/bge-reranker-large"
        assert pipeline.per_device_batch_size == 4
        assert pipeline.num_epochs == 2
        assert pipeline.learning_rate == 1e-5
        assert pipeline.ngpus == 2
        assert pipeline.train_group_size == 8

    def test_init_chinese(self):
        """Test initialization with Chinese language"""
        pipeline = RerankerFineTunePipeline(lang="zh")
        assert pipeline.lang == "zh"

    def test_init_english(self):
        """Test initialization with English language"""
        pipeline = RerankerFineTunePipeline(lang="en")
        assert pipeline.lang == "en"

