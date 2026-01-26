"""Test cases for AgenticRAG pipelines"""
import pytest
from lazyllm.tools.data import (
    AgenticRAGPipeline,
    AgenticRAGDepthPipeline,
    AgenticRAGWidthPipeline,
)


class TestAgenticRAGPipeline:
    """Test AgenticRAGPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = AgenticRAGPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.task_generator is not None
        assert pipeline.evaluator is not None

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        pipeline = AgenticRAGPipeline(
            llm_serving=None,
            data_num=50,
            max_per_task=5,
            max_question=5
        )
        assert pipeline is not None
        assert pipeline.task_generator is not None
        assert pipeline.evaluator is not None


class TestAgenticRAGDepthPipeline:
    """Test AgenticRAGDepthPipeline"""

    def test_init_default(self):
        """Test default initialization"""
        pipeline = AgenticRAGDepthPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.depth_generator is not None

    def test_init_with_n_rounds(self):
        """Test initialization with custom n_rounds"""
        pipeline = AgenticRAGDepthPipeline(n_rounds=5)
        assert pipeline is not None
        assert pipeline.depth_generator is not None


class TestAgenticRAGWidthPipeline:
    """Test AgenticRAGWidthPipeline"""

    def test_init(self):
        """Test initialization"""
        pipeline = AgenticRAGWidthPipeline()
        assert pipeline is not None
        assert pipeline.llm_serving is None
        assert pipeline.width_generator is not None
