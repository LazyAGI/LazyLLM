"""AgenticRAG Depth QA Generator operator"""
import json
import pandas as pd
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.agenticrag import (
    DepthQAGeneratorGetIdentifierPrompt,
    DepthQAGeneratorBackwardTaskPrompt,
    DepthQAGeneratorSupersetCheckPrompt,
    DepthQAGeneratorQuestionPrompt,
    DepthQAGeneratorAnswerPrompt,
    DepthQAGeneratorRecallScorePrompt
)
funcs = data_register.new_group('function')
classes = data_register.new_group('class')

class AgenticRAGDepthQAGenerator(classes):
    """
    Operator for generating deeper questions based on existing QA pairs.
    该算子以已有问答生成更深度的问题。
    """

    def __init__(
            self,
            llm_serving=None,
            n_rounds: int = 2,
    ):
        self.llm_serving = llm_serving
        self.n_rounds = n_rounds

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子以已有问答生成更深度的问题。\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（默认值：\"question\"）\n"
                "- output_key: 输出字段名（默认值：\"depth_question\"）\n"
            )
        elif lang == "en":
            return "This operator generates deeper questions based on existing QA pairs."
        else:
            return "DepthQAGenerator generate deeper questions based on existing QA pairs."

    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def recall_score(self, dataframe):
        prompt_template = DepthQAGeneratorRecallScorePrompt()
        golden_answers = dataframe["refined_answer"].tolist()
        llm_answers = dataframe["llm_answer"]
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt(golden_answer, llm_answer)
            for golden_answer, llm_answer in zip(golden_answers, llm_answers)
        ]
        recall_scores = self._generate_from_llm(user_prompts, system_prompt)
        valid_scores = []
        for score_str in recall_scores:
            if score_str is not None:
                try:
                    score_dict = json.loads(score_str)
                    valid_scores.append(score_dict["answer_score"])
                except (json.JSONDecodeError, KeyError):
                    LOG.warning(f"Failed to parse score: {score_str}")
                    valid_scores.append(0)
        return valid_scores

    def __call__(
            self,
            data,
            input_key: str = "question",
            output_key: str = "depth_question",
    ):
        """
        Process data to generate depth QA pairs.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input questions
            output_key: Key for output depth questions

        Returns:
            List of dict with generated depth QA pairs
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        # Get identifier if not present
        if "identifier" not in dataframe.columns:
            prompt_template = DepthQAGeneratorGetIdentifierPrompt()
            input_prompts = dataframe[input_key].tolist()
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
            identifiers = self._generate_from_llm(user_prompts, system_prompt)
            dataframe["identifier"] = identifiers

        for round_id in range(1, self.n_rounds + 1):
            LOG.info(f"=== Iteration Round {round_id} ===")

            identifier_key = "identifier" if round_id == 1 else f"new_identifier_{round_id - 1}"
            new_identifier_key = f"new_identifier_{round_id}"
            relation_key = f"relation_{round_id}"

            # Step 1: Generate relation and superset
            prompt_template = DepthQAGeneratorBackwardTaskPrompt()
            input_prompts = dataframe[identifier_key].tolist()
            user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
            backward_results = self._generate_from_llm(user_prompts, "")

            identifiers = []
            relations = []
            valid_indices = []

            for idx, result in enumerate(backward_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "identifier" in result and "relation" in result:
                        identifiers.append(result["identifier"])
                        relations.append(result["relation"])
                        valid_indices.append(idx)
                    else:
                        LOG.warning(f"[Skipped]: Result at index {idx} is invalid")
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to parse at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()
            dataframe[new_identifier_key] = identifiers
            dataframe[relation_key] = relations

            # Step 2: Check if superset is valid
            prompt_template = DepthQAGeneratorSupersetCheckPrompt()
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [
                prompt_template.build_prompt(new_id, relation, identifier)
                for new_id, relation, identifier in zip(
                    dataframe[new_identifier_key].tolist(),
                    dataframe[relation_key].tolist(),
                    dataframe[identifier_key].tolist()
                )
            ]
            check_results = self._generate_from_llm(user_prompts, system_prompt)

            valid_indices = []
            for idx, result in enumerate(check_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and result.get("new_query") == "valid":
                        valid_indices.append(idx)
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to check superset at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()

            # Step 3: Generate question
            prompt_template = DepthQAGeneratorQuestionPrompt()
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [
                prompt_template.build_prompt(new_id, relation, identifier)
                for new_id, relation, identifier in zip(
                    dataframe[new_identifier_key].tolist(),
                    dataframe[relation_key].tolist(),
                    dataframe[identifier_key].tolist()
                )
            ]
            check_results = self._generate_from_llm(user_prompts, system_prompt)

            new_queries = []
            valid_indices = []
            for idx, result in enumerate(check_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "new_query" in result:
                        new_queries.append(result["new_query"])
                        valid_indices.append(idx)
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to parse question at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()
            question_key = f"{output_key}_{round_id}"
            dataframe[question_key] = new_queries

        LOG.info("Depth QA generation completed!")
        return dataframe.to_dict('records')

