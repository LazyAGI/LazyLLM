"""Test cases for Reranker Synthesis operators"""
import pytest
import json
import tempfile
import os
from lazyllm.tools.data import (
    RerankerQueryGenerator,
    RerankerHardNegativeMiner,
    RerankerDataFormatter,
    RerankerTrainTestSplitter,
    RerankerFromEmbeddingConverter,
)


class TestRerankerQueryGenerator:
    """Test RerankerQueryGenerator"""

    def test_init(self):
        """Test initialization"""
        generator = RerankerQueryGenerator(
            llm_serving=None,
            num_queries=5,
            lang="zh",
            _save_data=False
        )
        assert generator is not None

    def test_init_english(self):
        """Test initialization with English"""
        generator = RerankerQueryGenerator(
            llm_serving=None,
            num_queries=3,
            lang="en",
            _save_data=False
        )
        assert generator is not None

    def test_init_with_difficulty_levels(self):
        """Test initialization with custom difficulty levels"""
        generator = RerankerQueryGenerator(
            llm_serving=None,
            difficulty_levels=["easy", "hard"],
            _save_data=False
        )
        assert generator is not None


class TestRerankerHardNegativeMiner:
    """Test RerankerHardNegativeMiner"""

    def test_init_random(self):
        """Test initialization with random strategy"""
        miner = RerankerHardNegativeMiner(
            mining_strategy="random",
            num_negatives=7,
            _save_data=False
        )
        assert miner is not None

    def test_init_bm25(self):
        """Test initialization with BM25 strategy"""
        miner = RerankerHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=7,
            _save_data=False
        )
        assert miner is not None

    def test_init_semantic(self):
        """Test initialization with semantic strategy"""
        miner = RerankerHardNegativeMiner(
            mining_strategy="semantic",
            num_negatives=7,
            embedding_serving=None,
            _save_data=False
        )
        assert miner is not None

    def test_init_mixed(self):
        """Test initialization with mixed strategy"""
        miner = RerankerHardNegativeMiner(
            mining_strategy="mixed",
            num_negatives=7,
            bm25_ratio=0.5,
            _save_data=False
        )
        assert miner is not None


class TestRerankerDataFormatter:
    """Test RerankerDataFormatter"""

    def test_init_flagreranker(self):
        """Test initialization with flagreranker format"""
        formatter = RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=8,
            _save_data=False
        )
        assert formatter is not None

    def test_init_cross_encoder(self):
        """Test initialization with cross_encoder format"""
        formatter = RerankerDataFormatter(
            output_format="cross_encoder",
            _save_data=False
        )
        assert formatter is not None

    def test_init_pairwise(self):
        """Test initialization with pairwise format"""
        formatter = RerankerDataFormatter(
            output_format="pairwise",
            _save_data=False
        )
        assert formatter is not None

    def test_format_flagreranker(self):
        """Test formatting to flagreranker format"""
        formatter = RerankerDataFormatter(
            output_format="flagreranker",
            train_group_size=4,  # 1 pos + 3 neg
            _save_data=False
        )
        data = [
            {"query": "什么是AI？", "pos": ["人工智能是..."], "neg": ["天气是...", "美食是...", "音乐是..."]},
        ]
        results = formatter(data)
        assert len(results) == 1
        assert results[0]["query"] == "什么是AI？"
        assert results[0]["pos"] == ["人工智能是..."]
        assert len(results[0]["neg"]) == 3  # train_group_size - 1

    def test_format_cross_encoder(self):
        """Test formatting to cross_encoder format"""
        formatter = RerankerDataFormatter(
            output_format="cross_encoder",
            _save_data=False
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = formatter(data)
        # Should expand: 1 pos with label 1 + 2 neg with label 0 = 3 rows
        assert len(results) == 3
        assert all("label" in r for r in results)
        assert sum(r["label"] for r in results) == 1  # Only 1 positive

    def test_format_pairwise(self):
        """Test formatting to pairwise format"""
        formatter = RerankerDataFormatter(
            output_format="pairwise",
            _save_data=False
        )
        data = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        results = formatter(data)
        # Should expand: 1 pos * 2 neg = 2 pairwise
        assert len(results) == 2
        assert all("doc_pos" in r and "doc_neg" in r for r in results)

    def test_format_with_output_file(self):
        """Test formatting with output file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.jsonl")
            formatter = RerankerDataFormatter(
                output_format="flagreranker",
                output_file=output_file,
                _save_data=False
            )
            data = [
                {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]},
            ]
            results = formatter(data)

            # Check file was created
            assert os.path.exists(output_file)


class TestRerankerTrainTestSplitter:
    """Test RerankerTrainTestSplitter"""

    def test_init(self):
        """Test initialization"""
        splitter = RerankerTrainTestSplitter(
            test_size=0.1,
            seed=42,
            _save_data=False
        )
        assert splitter is not None

    def test_split(self):
        """Test train/test splitting"""
        splitter = RerankerTrainTestSplitter(
            test_size=0.3,
            seed=42,
            _save_data=False
        )
        data = [
            {"query": f"query_{i}", "pos": [f"pos_{i}"], "neg": [f"neg_{i}"]}
            for i in range(10)
        ]
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
            train_file = os.path.join(temp_dir, "train.jsonl")
            test_file = os.path.join(temp_dir, "eval.jsonl")

            splitter = RerankerTrainTestSplitter(
                test_size=0.2,
                seed=42,
                train_output_file=train_file,
                test_output_file=test_file,
                _save_data=False
            )
            data = [
                {"query": f"query_{i}", "pos": [f"pos_{i}"], "neg": [f"neg_{i}"]}
                for i in range(10)
            ]
            results = splitter(data)

            # Check files were created
            assert os.path.exists(train_file)
            assert os.path.exists(test_file)


class TestRerankerFromEmbeddingConverter:
    """Test RerankerFromEmbeddingConverter"""

    def test_init(self):
        """Test initialization"""
        converter = RerankerFromEmbeddingConverter(
            adjust_neg_count=7,
            _save_data=False
        )
        assert converter is not None

    def test_convert_data(self):
        """Test converting embedding data to reranker format"""
        converter = RerankerFromEmbeddingConverter(
            adjust_neg_count=3,
            _save_data=False
        )
        # Embedding format with prompt
        data = [
            {
                "query": "Q1",
                "pos": ["P1"],
                "neg": ["N1", "N2", "N3", "N4", "N5"],
                "prompt": "Represent this sentence: "
            },
        ]
        results = converter(data)

        assert len(results) == 1
        assert results[0]["query"] == "Q1"
        assert results[0]["pos"] == ["P1"]
        assert len(results[0]["neg"]) == 3  # Adjusted to 3
        assert "prompt" not in results[0]  # No prompt in reranker format

    def test_convert_with_output_file(self):
        """Test converting with output file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "reranker.jsonl")
            converter = RerankerFromEmbeddingConverter(
                adjust_neg_count=7,
                output_file=output_file,
                _save_data=False
            )
            data = [
                {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
            ]
            results = converter(data)

            # Check file was created
            assert os.path.exists(output_file)

