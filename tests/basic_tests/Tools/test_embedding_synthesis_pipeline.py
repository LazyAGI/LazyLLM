"""Test cases for Embedding Synthesis pipelines"""
import pytest
from lazyllm.tools.data import (
    EmbeddingSynthesisPipeline,
    EmbeddingFineTunePipeline,
)


class TestEmbeddingSynthesisPipeline:
    """Test EmbeddingSynthesisPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = EmbeddingSynthesisPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.query_generator is not None
        assert pipeline.negative_miner is not None
        assert pipeline.formatter is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = EmbeddingSynthesisPipeline(
            llm_serving=None,
            num_queries=5,
            num_negatives=10,
            mining_strategy="bm25",
            output_format="flagembedding",
            instruction="Query: ",
            lang="en",
            enable_augmentation=True,
            test_size=0.2,
        )
        assert pipeline is not None
        assert pipeline.lang == "en"
        assert pipeline.enable_augmentation is True
        assert pipeline.test_size == 0.2
        assert pipeline.instruction == "Query: "
        assert pipeline.augmentor is not None

    def test_init_without_augmentation(self):
        """Test initialization without augmentation"""
        pipeline = EmbeddingSynthesisPipeline(
            enable_augmentation=False,
        )
        assert pipeline.augmentor is None

    def test_init_without_splitting(self):
        """Test initialization without train/test splitting"""
        pipeline = EmbeddingSynthesisPipeline(
            test_size=0,
        )
        assert pipeline.splitter is None

    def test_init_chinese(self):
        """Test initialization with Chinese language"""
        pipeline = EmbeddingSynthesisPipeline(lang="zh")
        assert pipeline.lang == "zh"
        assert "检索" in pipeline.instruction

    def test_init_english(self):
        """Test initialization with English language"""
        pipeline = EmbeddingSynthesisPipeline(lang="en")
        assert pipeline.lang == "en"
        assert "searching" in pipeline.instruction.lower()


class TestEmbeddingFineTunePipeline:
    """Test EmbeddingFineTunePipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = EmbeddingFineTunePipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.embed_model_path == "BAAI/bge-base-zh-v1.5"
        assert pipeline.synthesis_pipeline is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = EmbeddingFineTunePipeline(
            llm_serving=None,
            embed_model_path="BAAI/bge-large-zh-v1.5",
            num_queries=5,
            num_negatives=10,
            mining_strategy="bm25",
            instruction="Query: ",
            lang="en",
            per_device_batch_size=32,
            num_epochs=3,
            ngpus=2,
        )
        assert pipeline.embed_model_path == "BAAI/bge-large-zh-v1.5"
        assert pipeline.per_device_batch_size == 32
        assert pipeline.num_epochs == 3
        assert pipeline.ngpus == 2

    def test_init_chinese(self):
        """Test initialization with Chinese language"""
        pipeline = EmbeddingFineTunePipeline(lang="zh")
        assert pipeline.lang == "zh"
        assert "检索" in pipeline.instruction

    def test_init_english(self):
        """Test initialization with English language"""
        pipeline = EmbeddingFineTunePipeline(lang="en")
        assert pipeline.lang == "en"
        assert "searching" in pipeline.instruction.lower()

