"""AgenticRAG Atomic Task Generator operator"""
import json
import string
import re
from collections import Counter
import pandas as pd
from lazyllm import LOG
from ...base_data import DataOperatorRegistry
from ...prompts.agenticrag import (
    AtomicTaskGeneratorGetIdentifierPrompt,
    AtomicTaskGeneratorGetConlcusionPrompt,
    AtomicTaskGeneratorQuestionPrompt,
    AtomicTaskGeneratorCleanQAPrompt,
    AtomicTaskGeneratorAnswerPrompt,
    AtomicTaskGeneratorRecallScorePrompt,
    AtomicTaskGeneratorOptionalAnswerPrompt,
    AtomicTaskGeneratorGoldenDocAnswerPrompt
)


@DataOperatorRegistry.register(one_item=False, tag='agentic_rag')
class AgenticRAGAtomicTaskGenerator:
    """
    Operator for generating high-quality questions and verifiable answers from text content.
    该算子用于为提供的文本内容生成合适的高质量问题与可验证答案。
    """

    def __init__(
            self,
            llm_serving=None,
            data_num: int = 100,
            max_per_task: int = 10,
            max_question: int = 10,
    ):
        self.llm_serving = llm_serving
        self.data_num = data_num
        self.max_per_task = max_per_task
        self.max_question = max_question

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为提供的文本内容生成合适的高质量问题与可验证答案。\n\n"
                "输入参数：\n"
                "- input_key: 输入文本内容字段名（默认值：\"prompts\"）\n"
                "- output_question_key: 输出问题字段名（默认值：\"question\"）\n"
                "- output_answer_key: 输出答案字段名（默认值：\"answer\"）\n"
            )
        elif lang == "en":
            return (
                "This operator generates high-quality questions and verifiable answers for text content."
            )
        else:
            return "AtomicTaskGenerator generate high-quality questions and verifiable answers."

    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def __call__(
            self,
            data,
            input_key: str = "prompts",
            output_question_key: str = "question",
            output_answer_key: str = "answer",
            output_refined_answer_key: str = "refined_answer",
            output_optional_answer_key: str = "optional_answer",
            output_llm_answer_key: str = "llm_answer",
            output_golden_doc_answer_key: str = "golden_doc_answer",
    ):
        """
        Process data to generate atomic QA pairs.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input text content
            output_question_key: Key for output question
            output_answer_key: Key for output answer
            output_refined_answer_key: Key for refined answer
            output_optional_answer_key: Key for optional answers
            output_llm_answer_key: Key for LLM answers
            output_golden_doc_answer_key: Key for golden doc answers

        Returns:
            List of dict with generated QA pairs
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        # Step 0: Get identifier
        LOG.info("Get identifier...")
        prompt_template = AtomicTaskGeneratorGetIdentifierPrompt()
        input_prompts = dataframe[input_key].tolist()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
        identifiers = self._generate_from_llm(user_prompts, system_prompt)

        # Step 1: Get conclusions
        LOG.info("Get conclusions...")
        prompt_template = AtomicTaskGeneratorGetConlcusionPrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
        conclusions = self._generate_from_llm(user_prompts, system_prompt)

        # Expand each conclusion into multiple candidate tasks
        expanded_rows = []
        for idx, (row, output_str, identifier) in enumerate(zip(dataframe.to_dict('records'), conclusions, identifiers)):
            try:
                parsed = json.loads(self._clean_json_block(output_str))
                parsed = parsed[:self.max_per_task] if isinstance(parsed, list) else parsed
            except Exception as e:
                LOG.warning(f"JSON parse failed at idx={idx}: {e}")
                continue

            if not isinstance(parsed, list):
                continue

            for item in parsed:
                if isinstance(item, dict) and "conclusion" in item and "R" in item:
                    expanded_rows.append({
                        **row,
                        "identifier": str(identifier),
                        "candidate_tasks_str": json.dumps(item, ensure_ascii=False)
                    })

        if not expanded_rows:
            LOG.warning("No valid candidate tasks extracted.")
            return []

        dataframe = pd.DataFrame(expanded_rows)

        # Step 2: Generate questions
        LOG.info("Generate questions based on conclusion + reasoning...")
        prompt_template = AtomicTaskGeneratorQuestionPrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = []
        for s, raw_id in zip(dataframe["candidate_tasks_str"].tolist(), dataframe["identifier"].tolist()):
            try:
                task_item = json.loads(self._clean_json_block(s))
                clean_id_str = self._clean_json_block(raw_id)
                identifier_obj = json.loads(clean_id_str)
                identifier = identifier_obj.get("content_identifier", "Unknown")
                user_prompts.append(prompt_template.build_prompt(identifier, task_item["conclusion"], task_item["R"]))
            except Exception as e:
                LOG.warning(f"Failed to parse: {e}")
                user_prompts.append("")

        question_outputs = self._generate_from_llm(user_prompts, system_prompt)

        questions = []
        answers = []
        valid_rows = []

        for idx, (res, row) in enumerate(zip(question_outputs, dataframe.to_dict('records'))):
            try:
                parsed = json.loads(self._clean_json_block(res))
            except Exception as e:
                LOG.warning(f"Failed to parse question JSON at idx={idx}: {e}")
                continue

            if isinstance(parsed, dict) and "Q" in parsed:
                question = parsed["Q"]
                try:
                    task = json.loads(self._clean_json_block(row['candidate_tasks_str']))
                    answer = task.get("conclusion", "")
                except Exception:
                    answer = ""
                valid_rows.append(row)
                questions.append(str(question))
                answers.append(str(answer))

        if not valid_rows:
            LOG.warning("No valid QA pairs generated.")
            return []

        dataframe = pd.DataFrame(valid_rows)
        dataframe[output_question_key] = questions
        dataframe[output_answer_key] = answers

        # Step 3: Clean QA
        LOG.info("Clean QA...")
        prompt_template = AtomicTaskGeneratorCleanQAPrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt({"question": q, "original_answer": a})
            for q, a in zip(questions, answers)
        ]
        clean_outputs = self._generate_from_llm(user_prompts, system_prompt)

        final_answers = []
        for idx, res in enumerate(clean_outputs):
            try:
                parsed = json.loads(self._clean_json_block(res))
                final_answers.append(str(parsed.get("refined_answer", "")))
            except Exception as e:
                LOG.warning(f"Failed to parse cleaned QA at idx={idx}: {e}")
                final_answers.append("")

        dataframe[output_refined_answer_key] = final_answers

        LOG.info("QA generation completed!")
        return dataframe.to_dict('records')

