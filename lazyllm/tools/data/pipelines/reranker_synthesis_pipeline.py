"""
Reranker Synthesis Pipeline for LazyLLM

This pipeline demonstrates how to synthesize high-quality reranker training data:
1. Generate queries from document passages
2. Mine hard negative samples
3. Format data for training
4. Convert from embedding data (optional)

Example usage:
    from lazyllm.tools.data.pipeline import RerankerSynthesisPipeline

    # Create pipeline with LLM serving
    pipeline = RerankerSynthesisPipeline(llm_serving=your_llm_serving)

    # Run with document data
    data = [{"passage": "Your document passage here..."}]
    results = pipeline.forward(data)
"""
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from lazyllm import LOG
from ..operators.reranker_synthesis import (
    RerankerQueryGenerator,
    RerankerHardNegativeMiner,
    RerankerDataFormatter,
    RerankerTrainTestSplitter,
    RerankerFromEmbeddingConverter,
)
from ..base_data import data_register
funcs = data_register.new_group('function')
classes = data_register.new_group('class')

class RerankerSynthesisPipeline(classes):
    """
    Complete pipeline for synthesizing reranker training data.
    完整的 Reranker 训练数据合成流水线。

    This pipeline includes:
    1. RerankerQueryGenerator: Generate queries from passages
    2. RerankerHardNegativeMiner: Mine hard negative samples
    3. RerankerDataFormatter: Format data for training
    4. RerankerTrainTestSplitter: Split into train/test sets

    Args:
        llm_serving: LLM serving instance for query generation
        embedding_serving: Embedding serving for semantic negative mining (optional)
        num_queries: Number of queries to generate per passage
        num_negatives: Number of negative samples per query (default: 7 for train_group_size=8)
        mining_strategy: Strategy for mining negatives ("random", "bm25", "semantic", "mixed")
        output_format: Output format ("flagreranker", "cross_encoder", "pairwise")
        lang: Language for prompts ("zh" or "en")
        test_size: Proportion for test set (0 to disable splitting)
        train_group_size: Training group size (default: 8, i.e., 1 pos + 7 neg)
    """

    def __init__(
            self,
            llm_serving=None,
            embedding_serving=None,
            num_queries: int = 3,
            num_negatives: int = 7,
            mining_strategy: str = "random",
            output_format: str = "flagreranker",
            lang: str = "zh",
            test_size: float = 0.1,
            train_group_size: int = 8,
    ):
        self.llm_serving = llm_serving
        self.embedding_serving = embedding_serving
        self.lang = lang
        self.test_size = test_size
        self.train_group_size = train_group_size

        # Step 1: Query Generator
        self.query_generator = RerankerQueryGenerator(
            llm_serving=llm_serving,
            num_queries=num_queries,
            lang=lang,
            _save_data=False,
        )

        # Step 2: Hard Negative Miner
        self.negative_miner = RerankerHardNegativeMiner(
            mining_strategy=mining_strategy,
            num_negatives=num_negatives,
            embedding_serving=embedding_serving,
            _save_data=False,
        )

        # Step 3: Data Formatter
        self.formatter = RerankerDataFormatter(
            output_format=output_format,
            train_group_size=train_group_size,
            _save_data=False,
        )

        # Step 4: Train/Test Splitter
        self.splitter = RerankerTrainTestSplitter(
            test_size=test_size,
            _save_data=False,
        ) if test_size > 0 else None

    def forward(
            self,
            data,
            input_key: str = "passage",
            output_dir: Optional[str] = None,
            run_query_generation: bool = True,
            run_negative_mining: bool = True,
            run_formatting: bool = True,
            run_splitting: bool = True,
    ):
        """
        Run the reranker synthesis pipeline.

        Args:
            data: Input data (list of dict or DataFrame with passages)
            input_key: Key for input passage field
            output_dir: Directory to save output files (optional)
            run_query_generation: Whether to run query generation
            run_negative_mining: Whether to run negative mining
            run_formatting: Whether to run formatting
            run_splitting: Whether to run train/test splitting

        Returns:
            Formatted training data ready for reranker model training
        """
        LOG.info("Starting Reranker Synthesis Pipeline...")

        if isinstance(data, pd.DataFrame):
            results = data.to_dict('records')
        else:
            results = list(data)

        # Build corpus from input passages
        corpus = [item[input_key] for item in results if input_key in item]

        # Step 1: Generate queries
        if run_query_generation:
            LOG.info("Step 1: Generating queries from passages...")
            results = self.query_generator(
                results,
                input_key=input_key,
                output_query_key="query",
            )
            if not results:
                LOG.warning("No queries generated. Pipeline stopped.")
                return []

        # Step 2: Mine hard negatives
        if run_negative_mining and results:
            LOG.info("Step 2: Mining hard negative samples...")
            results = self.negative_miner(
                results,
                input_query_key="query",
                input_pos_key="pos",
                output_neg_key="neg",
                corpus=corpus,
            )

        # Step 3: Format data
        if run_formatting and results:
            LOG.info("Step 3: Formatting training data...")
            results = self.formatter(
                results,
                input_query_key="query",
                input_pos_key="pos",
                input_neg_key="neg",
            )

        # Step 4: Split train/test
        if run_splitting and self.splitter and self.test_size > 0 and results:
            LOG.info("Step 4: Splitting into train/test sets...")

            if output_dir:
                self.splitter.train_output_file = os.path.join(output_dir, "rerank_train.jsonl")
                self.splitter.test_output_file = os.path.join(output_dir, "rerank_eval.jsonl")

            results = self.splitter(results)

        # Save all results if output_dir specified
        if output_dir and results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            all_data_path = output_path / "rerank_all_data.jsonl"
            with open(all_data_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved all data to {all_data_path}")

        LOG.info(f"Reranker Synthesis Pipeline completed! Generated {len(results)} samples.")
        return results


class RerankerFromEmbeddingPipeline(classes):
    """
    Pipeline for converting embedding training data to reranker format.
    从 Embedding 训练数据转换为 Reranker 格式的流水线。

    This is useful when you already have embedding training data and want to
    train a reranker model on the same data distribution.

    Args:
        num_negatives: Target number of negatives (default: 7 for train_group_size=8)
        test_size: Proportion for test set
    """

    def __init__(
            self,
            num_negatives: int = 7,
            test_size: float = 0.1,
    ):
        self.num_negatives = num_negatives
        self.test_size = test_size

        # Converter
        self.converter = RerankerFromEmbeddingConverter(
            adjust_neg_count=num_negatives,
            _save_data=False,
        )

        # Splitter
        self.splitter = RerankerTrainTestSplitter(
            test_size=test_size,
            _save_data=False,
        ) if test_size > 0 else None

    def forward(
            self,
            data,
            output_dir: Optional[str] = None,
    ):
        """
        Convert embedding data to reranker format.

        Args:
            data: Embedding training data (list, DataFrame, or file path)
            output_dir: Directory to save output files

        Returns:
            Reranker formatted training data
        """
        LOG.info("Starting Reranker From Embedding Pipeline...")

        # Step 1: Convert format
        LOG.info("Step 1: Converting embedding data to reranker format...")
        results = self.converter(data)

        # Step 2: Split train/test
        if self.splitter and self.test_size > 0 and results:
            LOG.info("Step 2: Splitting into train/test sets...")

            if output_dir:
                self.splitter.train_output_file = os.path.join(output_dir, "rerank_train.jsonl")
                self.splitter.test_output_file = os.path.join(output_dir, "rerank_eval.jsonl")

            results = self.splitter(results)

        LOG.info(f"Conversion completed! Generated {len(results)} samples.")
        return results


class RerankerFineTunePipeline(classes):
    """
    End-to-end pipeline for reranker model fine-tuning with LazyLLM.
    使用 LazyLLM 进行 Reranker 模型微调的端到端流水线。

    This pipeline combines data synthesis with LazyLLM's fine-tuning capabilities:
    1. Synthesize training data using RerankerSynthesisPipeline
    2. Fine-tune reranker model using TrainableModule
    3. Evaluate the fine-tuned model

    Args:
        llm_serving: LLM serving for data synthesis
        rerank_model_path: Path to the base reranker model
        num_queries: Number of queries per passage
        num_negatives: Number of negatives per query
        mining_strategy: Negative mining strategy
        lang: Language for prompts
        # Fine-tuning config
        per_device_batch_size: Training batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        ngpus: Number of GPUs
        train_group_size: Training group size
    """

    def __init__(
            self,
            llm_serving=None,
            rerank_model_path: str = "BAAI/bge-reranker-base",
            num_queries: int = 3,
            num_negatives: int = 7,
            mining_strategy: str = "random",
            lang: str = "zh",
            # Fine-tuning config
            per_device_batch_size: int = 2,
            num_epochs: int = 1,
            learning_rate: float = 6e-5,
            ngpus: int = 1,
            train_group_size: int = 8,
    ):
        self.llm_serving = llm_serving
        self.rerank_model_path = rerank_model_path
        self.lang = lang

        # Data synthesis pipeline
        self.synthesis_pipeline = RerankerSynthesisPipeline(
            llm_serving=llm_serving,
            num_queries=num_queries,
            num_negatives=num_negatives,
            mining_strategy=mining_strategy,
            output_format="flagreranker",
            lang=lang,
            test_size=0.1,
            train_group_size=train_group_size,
        )

        # Fine-tuning configuration
        self.per_device_batch_size = per_device_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.ngpus = ngpus
        self.train_group_size = train_group_size

    def synthesize_data(
            self,
            documents,
            input_key: str = "passage",
            output_dir: str = "./reranker_data",
    ):
        """
        Synthesize training data from documents.

        Args:
            documents: List of documents with passages
            input_key: Key for passage field
            output_dir: Directory to save synthesized data

        Returns:
            Paths to train and eval data files
        """
        LOG.info("Synthesizing reranker training data...")

        results = self.synthesis_pipeline.forward(
            documents,
            input_key=input_key,
            output_dir=output_dir,
        )

        train_path = os.path.join(output_dir, "rerank_train.jsonl")
        eval_path = os.path.join(output_dir, "rerank_eval.jsonl")

        return train_path, eval_path, results

    def finetune(
            self,
            train_data_path: str,
    ):
        """
        Fine-tune the reranker model using LazyLLM.

        Args:
            train_data_path: Path to training data file

        Returns:
            Fine-tuned TrainableModule
        """
        try:
            import lazyllm
            from lazyllm import TrainableModule, finetune, launchers
        except ImportError:
            raise ImportError("LazyLLM is required for fine-tuning")

        LOG.info(f"Fine-tuning reranker model: {self.rerank_model_path}")

        # Configure reranker model with fine-tuning
        reranker_model = TrainableModule(self.rerank_model_path) \
            .mode('finetune') \
            .trainset(train_data_path) \
            .finetune_method((
                finetune.auto,
                {
                    'launcher': launchers.remote(nnode=1, nproc=1, ngpus=self.ngpus),
                    'per_device_train_batch_size': self.per_device_batch_size,
                    'num_train_epochs': self.num_epochs,
                    'learning_rate': self.learning_rate,
                    'train_group_size': self.train_group_size,
                    'query_max_len': 256,
                    'passage_max_len': 256,
                }
            ))

        # Start fine-tuning
        reranker_model.update()

        LOG.info("Fine-tuning completed!")
        return reranker_model

    def forward(
            self,
            documents,
            input_key: str = "passage",
            output_dir: str = "./reranker_finetune",
            run_synthesis: bool = True,
            run_finetune: bool = True,
    ):
        """
        Run the complete fine-tuning pipeline.

        Args:
            documents: List of documents with passages
            input_key: Key for passage field
            output_dir: Directory for output files
            run_synthesis: Whether to run data synthesis
            run_finetune: Whether to run fine-tuning

        Returns:
            Fine-tuned model (if run_finetune=True) or training data path
        """
        LOG.info("Starting Reranker Fine-Tune Pipeline...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Synthesize data
        if run_synthesis:
            train_path, eval_path, _ = self.synthesize_data(
                documents,
                input_key=input_key,
                output_dir=output_dir,
            )
        else:
            train_path = os.path.join(output_dir, "rerank_train.jsonl")

        # Step 2: Fine-tune model
        if run_finetune:
            model = self.finetune(train_data_path=train_path)
            LOG.info("Reranker Fine-Tune Pipeline completed!")
            return model

        LOG.info("Data synthesis completed!")
        return train_path


__all__ = [
    'RerankerSynthesisPipeline',
    'RerankerFromEmbeddingPipeline',
    'RerankerFineTunePipeline',
]

