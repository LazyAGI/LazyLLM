"""
Embedding Synthesis Pipeline for LazyLLM

This pipeline demonstrates how to synthesize high-quality embedding training data:
1. Generate queries from document passages
2. Mine hard negative samples
3. Augment data for diversity
4. Format data for training

Example usage:
    from lazyllm.tools.data.pipeline import EmbeddingSynthesisPipeline

    # Create pipeline with LLM serving
    pipeline = EmbeddingSynthesisPipeline(llm_serving=your_llm_serving)

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
from ..operator.embedding_synthesis import (
    EmbeddingQueryGenerator,
    EmbeddingHardNegativeMiner,
    EmbeddingDataAugmentor,
    EmbeddingDataFormatter,
)
from ..operator.embedding_synthesis.embedding_data_formatter import EmbeddingTrainTestSplitter


class EmbeddingSynthesisPipeline:
    """
    Complete pipeline for synthesizing embedding training data.
    完整的 Embedding 训练数据合成流水线。

    This pipeline includes:
    1. EmbeddingQueryGenerator: Generate queries from passages
    2. EmbeddingHardNegativeMiner: Mine hard negative samples
    3. EmbeddingDataAugmentor: Augment data for diversity (optional)
    4. EmbeddingDataFormatter: Format data for training
    5. EmbeddingTrainTestSplitter: Split into train/test sets (optional)

    Args:
        llm_serving: LLM serving instance for query generation
        embedding_serving: Embedding serving for semantic negative mining (optional)
        num_queries: Number of queries to generate per passage
        num_negatives: Number of negative samples per query
        mining_strategy: Strategy for mining negatives ("random", "bm25", "semantic")
        output_format: Output format ("flagembedding", "sentence_transformers", "triplet")
        instruction: Instruction/prompt for embedding model training
        lang: Language for prompts ("zh" or "en")
        enable_augmentation: Whether to enable data augmentation
        test_size: Proportion for test set (0 to disable splitting)
    """

    def __init__(
            self,
            llm_serving=None,
            embedding_serving=None,
            num_queries: int = 3,
            num_negatives: int = 7,
            mining_strategy: str = "random",
            output_format: str = "flagembedding",
            instruction: Optional[str] = None,
            lang: str = "zh",
            enable_augmentation: bool = False,
            test_size: float = 0.1,
    ):
        self.llm_serving = llm_serving
        self.embedding_serving = embedding_serving
        self.lang = lang
        self.enable_augmentation = enable_augmentation
        self.test_size = test_size

        # Default instruction based on language
        if instruction is None:
            instruction = (
                "为这个句子生成表示以用于检索相关段落: "
                if lang == "zh" else
                "Represent this sentence for searching relevant passages: "
            )
        self.instruction = instruction

        # Step 1: Query Generator
        self.query_generator = EmbeddingQueryGenerator(
            llm_serving=llm_serving,
            num_queries=num_queries,
            lang=lang,
            _save_data=False,
        )

        # Step 2: Hard Negative Miner
        self.negative_miner = EmbeddingHardNegativeMiner(
            mining_strategy=mining_strategy,
            num_negatives=num_negatives,
            embedding_serving=embedding_serving,
            _save_data=False,
        )

        # Step 3: Data Augmentor (optional)
        if enable_augmentation:
            self.augmentor = EmbeddingDataAugmentor(
                llm_serving=llm_serving,
                augment_methods=["query_rewrite"],
                num_augments=2,
                lang=lang,
                _save_data=False,
            )
        else:
            self.augmentor = None

        # Step 4: Data Formatter
        self.formatter = EmbeddingDataFormatter(
            output_format=output_format,
            instruction=instruction,
            _save_data=False,
        )

        # Step 5: Train/Test Splitter
        self.splitter = EmbeddingTrainTestSplitter(
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
            run_augmentation: bool = True,
            run_formatting: bool = True,
            run_splitting: bool = True,
    ):
        """
        Run the embedding synthesis pipeline.

        Args:
            data: Input data (list of dict or DataFrame with passages)
            input_key: Key for input passage field
            output_dir: Directory to save output files (optional)
            run_query_generation: Whether to run query generation
            run_negative_mining: Whether to run negative mining
            run_augmentation: Whether to run augmentation
            run_formatting: Whether to run formatting
            run_splitting: Whether to run train/test splitting

        Returns:
            Formatted training data ready for embedding model training
        """
        LOG.info("Starting Embedding Synthesis Pipeline...")

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

        # Step 3: Augment data (optional)
        if run_augmentation and self.enable_augmentation and self.augmentor and results:
            LOG.info("Step 3: Augmenting training data...")
            results = self.augmentor(
                results,
                input_query_key="query",
                output_query_key="query",
                keep_original=True,
            )

        # Step 4: Format data
        if run_formatting and results:
            LOG.info("Step 4: Formatting training data...")
            results = self.formatter(
                results,
                input_query_key="query",
                input_pos_key="pos",
                input_neg_key="neg",
            )

        # Step 5: Split train/test
        if run_splitting and self.splitter and self.test_size > 0 and results:
            LOG.info("Step 5: Splitting into train/test sets...")

            if output_dir:
                self.splitter.train_output_file = os.path.join(output_dir, "train.json")
                self.splitter.test_output_file = os.path.join(output_dir, "eval.json")

            results = self.splitter(results)

        # Save all results if output_dir specified
        if output_dir and results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            all_data_path = output_path / "all_data.json"
            with open(all_data_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            LOG.info(f"Saved all data to {all_data_path}")

        LOG.info(f"Embedding Synthesis Pipeline completed! Generated {len(results)} samples.")
        return results


class EmbeddingFineTunePipeline:
    """
    End-to-end pipeline for embedding model fine-tuning with LazyLLM.
    使用 LazyLLM 进行 Embedding 模型微调的端到端流水线。

    This pipeline combines data synthesis with LazyLLM's fine-tuning capabilities:
    1. Synthesize training data using EmbeddingSynthesisPipeline
    2. Fine-tune embedding model using TrainableModule
    3. Evaluate the fine-tuned model

    Args:
        llm_serving: LLM serving for data synthesis
        embed_model_path: Path to the base embedding model
        synthesis_config: Configuration for data synthesis
        finetune_config: Configuration for fine-tuning
    """

    def __init__(
            self,
            llm_serving=None,
            embed_model_path: str = "BAAI/bge-base-zh-v1.5",
            num_queries: int = 3,
            num_negatives: int = 7,
            mining_strategy: str = "random",
            instruction: Optional[str] = None,
            lang: str = "zh",
            # Fine-tuning config
            per_device_batch_size: int = 16,
            num_epochs: int = 2,
            ngpus: int = 1,
    ):
        self.llm_serving = llm_serving
        self.embed_model_path = embed_model_path
        self.lang = lang

        # Default instruction
        if instruction is None:
            instruction = (
                "为这个句子生成表示以用于检索相关段落: "
                if lang == "zh" else
                "Represent this sentence for searching relevant passages: "
            )
        self.instruction = instruction

        # Data synthesis pipeline
        self.synthesis_pipeline = EmbeddingSynthesisPipeline(
            llm_serving=llm_serving,
            num_queries=num_queries,
            num_negatives=num_negatives,
            mining_strategy=mining_strategy,
            output_format="flagembedding",
            instruction=instruction,
            lang=lang,
            enable_augmentation=False,
            test_size=0.1,
        )

        # Fine-tuning configuration
        self.per_device_batch_size = per_device_batch_size
        self.num_epochs = num_epochs
        self.ngpus = ngpus

    def synthesize_data(
            self,
            documents,
            input_key: str = "passage",
            output_dir: str = "./embedding_data",
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
        LOG.info("Synthesizing embedding training data...")

        results = self.synthesis_pipeline.forward(
            documents,
            input_key=input_key,
            output_dir=output_dir,
        )

        train_path = os.path.join(output_dir, "train.json")
        eval_path = os.path.join(output_dir, "eval.json")

        return train_path, eval_path, results

    def finetune(
            self,
            train_data_path: str,
            output_model_dir: Optional[str] = None,
    ):
        """
        Fine-tune the embedding model using LazyLLM.

        Args:
            train_data_path: Path to training data file
            output_model_dir: Directory to save fine-tuned model

        Returns:
            Fine-tuned TrainableModule
        """
        try:
            import lazyllm
            from lazyllm import TrainableModule
        except ImportError:
            raise ImportError("LazyLLM is required for fine-tuning")

        LOG.info(f"Fine-tuning embedding model: {self.embed_model_path}")

        # Configure embedding model with fine-tuning
        embed = TrainableModule(self.embed_model_path)\
            .mode('finetune')\
            .trainset(train_data_path)\
            .finetune_method((
                lazyllm.finetune.flagembedding,
                {
                    'launcher': lazyllm.launchers.remote(nnode=1, nproc=1, ngpus=self.ngpus),
                    'per_device_train_batch_size': self.per_device_batch_size,
                    'num_train_epochs': self.num_epochs,
                }
            ))

        # Start fine-tuning
        embed.update()

        LOG.info("Fine-tuning completed!")
        return embed

    def forward(
            self,
            documents,
            input_key: str = "passage",
            output_dir: str = "./embedding_finetune",
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
        LOG.info("Starting Embedding Fine-Tune Pipeline...")

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
            train_path = os.path.join(output_dir, "train.json")
            eval_path = os.path.join(output_dir, "eval.json")

        # Step 2: Fine-tune model
        if run_finetune:
            model = self.finetune(
                train_data_path=train_path,
                output_model_dir=os.path.join(output_dir, "model"),
            )
            LOG.info("Embedding Fine-Tune Pipeline completed!")
            return model

        LOG.info("Data synthesis completed!")
        return train_path


__all__ = [
    'EmbeddingSynthesisPipeline',
    'EmbeddingFineTunePipeline',
]

