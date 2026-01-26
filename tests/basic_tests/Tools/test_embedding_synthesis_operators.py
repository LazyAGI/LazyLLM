"""Test cases for Embedding Synthesis operators"""
import pytest
import json
import tempfile
import os
from lazyllm.tools.data import (
    EmbeddingQueryGenerator,
    EmbeddingHardNegativeMiner,
    EmbeddingDataFormatter,
    EmbeddingDataAugmentor,
    EmbeddingTrainTestSplitter,
)


class TestEmbeddingQueryGenerator:
    """Test EmbeddingQueryGenerator"""

    def test_init(self):
        """Test initialization"""
        generator = EmbeddingQueryGenerator(
            llm_serving=None,
            num_queries=5,
            lang="zh",
            _save_data=False
        )
        assert generator is not None

    def test_init_english(self):
        """Test initialization with English"""
        generator = EmbeddingQueryGenerator(
            llm_serving=None,
            num_queries=3,
            lang="en",
            _save_data=False
        )
        assert generator is not None


class TestEmbeddingHardNegativeMiner:
    """Test EmbeddingHardNegativeMiner"""

    def test_init_random(self):
        """Test initialization with random strategy"""
        miner = EmbeddingHardNegativeMiner(
            mining_strategy="random",
            num_negatives=5,
            _save_data=False
        )
        assert miner is not None

    def test_init_bm25(self):
        """Test initialization with BM25 strategy"""
        miner = EmbeddingHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=7,
            _save_data=False
        )
        assert miner is not None

    def test_init_semantic(self):
        """Test initialization with semantic strategy"""
        miner = EmbeddingHardNegativeMiner(
            mining_strategy="semantic",
            num_negatives=10,
            embedding_serving=None,
            _save_data=False
        )
        assert miner is not None


class TestEmbeddingDataAugmentor:
    """Test EmbeddingDataAugmentor"""

    def test_init(self):
        """Test initialization"""
        augmentor = EmbeddingDataAugmentor(
            llm_serving=None,
            augment_methods=["query_rewrite"],
            num_augments=2,
            lang="zh",
            _save_data=False
        )
        assert augmentor is not None

    def test_init_english(self):
        """Test initialization with English"""
        augmentor = EmbeddingDataAugmentor(
            llm_serving=None,
            lang="en",
            _save_data=False
        )
        assert augmentor is not None


class TestEmbeddingDataFormatter:
    """Test EmbeddingDataFormatter"""

    def test_init_flagembedding(self):
        """Test initialization with flagembedding format"""
        formatter = EmbeddingDataFormatter(
            output_format="flagembedding",
            instruction="Represent this sentence: ",
            _save_data=False
        )
        assert formatter is not None

    def test_init_sentence_transformers(self):
        """Test initialization with sentence_transformers format"""
        formatter = EmbeddingDataFormatter(
            output_format="sentence_transformers",
            _save_data=False
        )
        assert formatter is not None

    def test_init_triplet(self):
        """Test initialization with triplet format"""
        formatter = EmbeddingDataFormatter(
            output_format="triplet",
            _save_data=False
        )
        assert formatter is not None

    def test_format_flagembedding(self):
        """Test formatting to flagembedding format"""
        formatter = EmbeddingDataFormatter(
            output_format="flagembedding",
            instruction="Query: ",
            _save_data=False
        )
        data = [
            {"query": "什么是AI？", "pos": ["人工智能是..."], "neg": ["天气是..."]},
        ]
        results = formatter(data)
        assert len(results) == 1
        assert results[0]["query"] == "什么是AI？"
        assert results[0]["pos"] == ["人工智能是..."]
        assert results[0]["neg"] == ["天气是..."]
        assert results[0]["prompt"] == "Query: "

    def test_format_sentence_transformers(self):
        """Test formatting to sentence_transformers format"""
        formatter = EmbeddingDataFormatter(
            output_format="sentence_transformers",
            _save_data=False
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = formatter(data)
        # Should expand to anchor-positive-negative triplets
        assert len(results) == 2  # 1 pos * 2 neg
        assert all("anchor" in r and "positive" in r and "negative" in r for r in results)

    def test_format_triplet(self):
        """Test formatting to triplet format"""
        formatter = EmbeddingDataFormatter(
            output_format="triplet",
            _save_data=False
        )
        data = [
            {"query": "Q1", "pos": ["P1", "P2"], "neg": ["N1"]},
        ]
        results = formatter(data)
        # Should expand: 2 pos * 1 neg = 2 triplets
        assert len(results) == 2
        assert all("query" in r and "positive" in r and "negative" in r for r in results)

    def test_format_with_output_file(self):
        """Test formatting with output file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.jsonl")
            formatter = EmbeddingDataFormatter(
                output_format="flagembedding",
                output_file=output_file,
                _save_data=False
            )
            data = [
                {"query": "Q1", "pos": ["P1"], "neg": ["N1"]},
            ]
            results = formatter(data)

            # Check file was created
            assert os.path.exists(output_file)

            # Check file contents
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            assert len(lines) == 1


class TestEmbeddingTrainTestSplitter:
    """Test EmbeddingTrainTestSplitter"""

    def test_init(self):
        """Test initialization"""
        splitter = EmbeddingTrainTestSplitter(
            test_size=0.2,
            seed=42,
            _save_data=False
        )
        assert splitter is not None

    def test_split(self):
        """Test train/test splitting"""
        splitter = EmbeddingTrainTestSplitter(
            test_size=0.3,
            seed=42,
            _save_data=False
        )
        data = [{"id": i, "query": f"query_{i}"} for i in range(10)]
        results = splitter(data)

        # All samples should have split label
        assert len(results) == 10
        assert all("split" in r for r in results)

        # Check split ratio
        train_count = sum(1 for r in results if r["split"] == "train")
        test_count = sum(1 for r in results if r["split"] == "test")
        assert train_count == 7  # 70%
        assert test_count == 3   # 30%

    def test_split_with_output_files(self):
        """Test splitting with output files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_file = os.path.join(temp_dir, "train.json")
            test_file = os.path.join(temp_dir, "test.json")

            splitter = EmbeddingTrainTestSplitter(
                test_size=0.2,
                seed=42,
                train_output_file=train_file,
                test_output_file=test_file,
                _save_data=False
            )
            data = [{"id": i, "query": f"query_{i}"} for i in range(10)]
            results = splitter(data)

            # Check files were created
            assert os.path.exists(train_file)
            assert os.path.exists(test_file)

            # Check file contents
            with open(train_file, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()
            with open(test_file, 'r', encoding='utf-8') as f:
                test_lines = f.readlines()

            assert len(train_lines) == 8  # 80%
            assert len(test_lines) == 2   # 20%
