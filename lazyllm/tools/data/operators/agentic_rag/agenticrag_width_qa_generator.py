"""AgenticRAG Width QA Generator operator"""
import json
import pandas as pd
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.agenticrag import (
    WidthQAGeneratorMergePrompt,
    WidthQAGeneratorOriginCheckPrompt,
    WidthQAGeneratorQuestionVerifyPrompt,
    WidthQAGeneratorAnswerPrompt,
    WidthQAGeneratorRecallScorePrompt
)
funcs = data_register.new_group('function')
classes = data_register.new_group('class')

class AgenticRAGWidthQAGenerator(classes):
    """
    Operator for combining two QA pairs to generate new questions.
    该算子用于结合两个问答，生成新的问题。
    """

    def __init__(
            self,
            llm_serving=None,
    ):
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于结合两个问答，生成新的问题。\n\n"
                "输入参数：\n"
                "- input_question_key: 输入问题字段名（默认值：\"question\"）\n"
                "- input_identifier_key: 输入标识符字段名（默认值：\"identifier\"）\n"
                "- input_answer_key: 输入答案字段名（默认值：\"answer\"）\n"
                "- output_question_key: 输出问题字段名（默认值：\"generated_width_task\"）\n"
            )
        elif lang == "en":
            return "This operator combines two QA pairs to generate a new question."
        else:
            return "WidthQAGenerator combine two QA pairs and generate a new question."

    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def recall_score(self, dataframe):
        prompt_template = WidthQAGeneratorRecallScorePrompt()
        golden_answers = dataframe["original_answer"].tolist()
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
            input_question_key: str = "question",
            input_identifier_key: str = "identifier",
            input_answer_key: str = "answer",
            output_question_key: str = "generated_width_task",
    ):
        """
        Process data to generate width QA pairs.

        Args:
            data: List of dict or pandas DataFrame
            input_question_key: Key for input questions
            input_identifier_key: Key for input identifiers
            input_answer_key: Key for input answers
            output_question_key: Key for output width questions

        Returns:
            List of dict with generated width QA pairs
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        # Step 0: Prepare input batch
        input_batch = []
        for i, (_, row) in enumerate(dataframe.iterrows()):
            input_batch.append({
                "index": i,
                "question": row[input_question_key],
                "content_identifier": row[input_identifier_key],
                "golden_answer": row[input_answer_key]
            })

        # Merge prompt
        prompt_template = WidthQAGeneratorMergePrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt([input_batch[i], input_batch[i + 1]])
            for i in range(len(input_batch) - 1)
        ]
        merge_results = self._generate_from_llm(user_prompts, system_prompt)

        merged_rows = []
        for idx, result in enumerate(merge_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))[0]

                if not isinstance(result, dict) or "question" not in result or "index" not in result:
                    LOG.warning(f"[Skipped]: Invalid result at index {idx}")
                    continue

                indices = result["index"] if isinstance(result["index"], list) else [result["index"]]
                group_items = [input_batch[i] for i in indices]

                merged_rows.append({
                    "question": result["question"],
                    "content_identifier": result["content_identifier"],
                    "qa_index": indices,
                    "index": idx,
                    "original_answer": [item["golden_answer"] for item in group_items],
                    "original_question": [item["question"] for item in group_items],
                })

            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse merge result at index {idx}: {e}")
                continue

        dataframe = pd.DataFrame(merged_rows)

        # Step 1: Check origin
        prompt_template = WidthQAGeneratorOriginCheckPrompt()
        system_prompt = prompt_template.build_system_prompt()
        check_input_batch = []
        for idx, q, ori_q in zip(dataframe["index"], dataframe["question"], dataframe["original_question"]):
            check_input_batch.append({
                "index": idx,
                "complex_question": q,
                "original_questions": ori_q if isinstance(ori_q, list) else [ori_q]
            })
        user_prompts = [prompt_template.build_prompt(input) for input in check_input_batch]
        check_query_results = self._generate_from_llm(user_prompts, system_prompt)

        states = []
        complex_questions = []

        for idx, result in enumerate(check_query_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))
                    if isinstance(result, list):
                        result = result[0]

                if isinstance(result, dict):
                    states.append(result.get("state", None))
                    complex_questions.append(result.get("complex_question", None))
                else:
                    states.append(None)
                    complex_questions.append(None)
            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse result at index {idx}: {e}")
                states.append(None)
                complex_questions.append(None)

        dataframe["state"] = states
        dataframe[output_question_key] = complex_questions
        dataframe = dataframe[dataframe["state"] == 1].copy()

        # Step 2: Verify questions
        prompt_template = WidthQAGeneratorQuestionVerifyPrompt()
        system_prompt = prompt_template.build_system_prompt()
        verify_input_batch = []
        for idx, q in zip(dataframe["index"], dataframe[output_question_key]):
            verify_input_batch.append({
                "index": idx,
                "complex_question": q,
            })
        user_prompts = [prompt_template.build_prompt(input) for input in verify_input_batch]
        question_verify_results = self._generate_from_llm(user_prompts, system_prompt)

        llm_answers = []
        for idx, result in enumerate(question_verify_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))[0]

                if isinstance(result, dict):
                    llm_answers.append(result.get("llm_answer", None))
                else:
                    llm_answers.append(None)
            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse result at index {idx}: {e}")
                llm_answers.append(None)

        dataframe["llm_answer"] = llm_answers

        # Filter by recall score
        llm_score = self.recall_score(dataframe)
        dataframe["llm_score"] = llm_score
        dataframe = dataframe[dataframe["llm_score"] < 1].drop(columns=["llm_score"]).reset_index(drop=True)
        dataframe = dataframe.drop(columns="llm_answer")

        LOG.info("Width QA generation completed!")
        return dataframe.to_dict('records')

