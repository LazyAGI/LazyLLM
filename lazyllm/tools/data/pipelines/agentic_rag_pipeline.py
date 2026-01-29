import pandas as pd
from lazyllm import LOG
from ..operators.agentic_rag import (
    AgenticRAGQAF1SampleEvaluator,
    AgenticRAGAtomicTaskGenerator,
    AgenticRAGDepthQAGenerator,
    AgenticRAGWidthQAGenerator,
)
from ..base_data import data_register
funcs = data_register.new_group('function')
classes = data_register.new_group('class')
class AgenticRAGPipeline(classes):
    """
    AgenticRAG Pipeline for generating and evaluating QA pairs.
    用于生成和评估 QA 对的 AgenticRAG 流水线。

    This pipeline includes:
    - AgenticRAGAtomicTaskGenerator: Generate atomic QA tasks from text
    - AgenticRAGQAF1SampleEvaluator: Evaluate QA results using F1 score

    Args:
        llm_serving: LLM serving instance for text generation
        data_num: Number of data samples to process (default: 100)
        max_per_task: Maximum candidate tasks per input (default: 10)
        max_question: Maximum questions per document (default: 10)
    """

    def __init__(
            self,
            llm_serving=None,
            data_num: int = 100,
            max_per_task: int = 10,
            max_question: int = 10,
    ):
        self.llm_serving = llm_serving

        # Step 1: Generate atomic tasks
        self.task_generator = AgenticRAGAtomicTaskGenerator(
            llm_serving=llm_serving,
            data_num=data_num,
            max_per_task=max_per_task,
            max_question=max_question,
        )

        # Step 2: Evaluate with F1 score
        self.evaluator = AgenticRAGQAF1SampleEvaluator()

    def forward(
            self,
            data,
            input_key: str = "contents",
            run_evaluation: bool = True,
    ):
        """
        Run the AgenticRAG pipeline.

        Args:
            data: Input data (list of dict or DataFrame)
            input_key: Key for input content field
            run_evaluation: Whether to run F1 evaluation

        Returns:
            Processed data with QA pairs and optional F1 scores
        """
        LOG.info("Starting AgenticRAG Pipeline...")

        # Step 1: Generate atomic QA tasks
        LOG.info("Step 1: Generating atomic QA tasks...")
        results = self.task_generator(
            data,
            input_key=input_key,
        )

        if not results:
            LOG.warning("No QA pairs generated.")
            return results

        # Step 2: Evaluate (optional)
        if run_evaluation and results:
            LOG.info("Step 2: Evaluating with F1 score...")
            results = self.evaluator(
                results,
                prediction_key="refined_answer",
                ground_truth_key="golden_doc_answer",
                output_key="F1Score",
            )

        LOG.info("AgenticRAG Pipeline completed!")
        return results


class AgenticRAGDepthPipeline(classes):
    """
    AgenticRAG Depth Pipeline for generating deeper QA pairs.
    用于生成更深层次 QA 对的 AgenticRAG 深度流水线。

    Args:
        llm_serving: LLM serving instance
        n_rounds: Number of depth expansion rounds (default: 2)
    """

    def __init__(
            self,
            llm_serving=None,
            n_rounds: int = 2,
    ):
        self.llm_serving = llm_serving

        # Depth QA generator
        self.depth_generator = AgenticRAGDepthQAGenerator(
            llm_serving=llm_serving,
            n_rounds=n_rounds,
        )

    def forward(
            self,
            data,
            input_key: str = "question",
            output_key: str = "depth_question",
    ):
        """
        Run the depth QA generation pipeline.

        Args:
            data: Input data with existing QA pairs
            input_key: Key for input questions
            output_key: Key for output depth questions

        Returns:
            Data with depth-expanded QA pairs
        """
        LOG.info("Starting AgenticRAG Depth Pipeline...")

        results = self.depth_generator(
            data,
            input_key=input_key,
            output_key=output_key,
        )

        LOG.info("AgenticRAG Depth Pipeline completed!")
        return results


class AgenticRAGWidthPipeline(classes) :
    """
    AgenticRAG Width Pipeline for combining QA pairs.
    用于合并 QA 对的 AgenticRAG 宽度流水线。

    Args:
        llm_serving: LLM serving instance
    """

    def __init__(
            self,
            llm_serving=None,
    ):
        self.llm_serving = llm_serving

        # Width QA generator
        self.width_generator = AgenticRAGWidthQAGenerator(
            llm_serving=llm_serving,
        )

    def forward(
            self,
            data,
            input_question_key: str = "question",
            input_identifier_key: str = "identifier",
            input_answer_key: str = "answer",
            output_question_key: str = "generated_width_task",
    ):
        """
        Run the width QA generation pipeline.

        Args:
            data: Input data with existing QA pairs
            input_question_key: Key for input questions
            input_identifier_key: Key for identifiers
            input_answer_key: Key for answers
            output_question_key: Key for output width questions

        Returns:
            Data with width-combined QA pairs
        """
        LOG.info("Starting AgenticRAG Width Pipeline...")

        results = self.width_generator(
            data,
            input_question_key=input_question_key,
            input_identifier_key=input_identifier_key,
            input_answer_key=input_answer_key,
            output_question_key=output_question_key,
        )

        LOG.info("AgenticRAG Width Pipeline completed!")
        return results


__all__ = [
    'AgenticRAGPipeline',
    'AgenticRAGDepthPipeline',
    'AgenticRAGWidthPipeline',
]

