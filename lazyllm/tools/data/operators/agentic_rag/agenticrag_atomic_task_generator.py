"""AgenticRAG Atomic Task Generator operator"""
import json
import string
import re
from collections import Counter
from typing import List
import lazyllm
from lazyllm import LOG
from ...base_data import data_register
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
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 获取或创建 agenticrag 组（确保所有模块共享同一个组）
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')

class AgenticRAGAtomicTaskGenerator(agenticrag):
    """
    Operator for generating high-quality questions and verifiable answers from text content.
    该算子用于为提供的文本内容生成合适的高质量问题与可验证答案。
    """

    def __init__(
            self,
            data_num: int = 100,
            max_per_task: int = 10,
            max_question: int = 10,
            llm = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
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
        if self.llm is None:
            raise ValueError("LLM is not configured")
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        # prompter = lazyllm.ChatPrompter(system_prompt)
        # llm_serve = self.llm.prompt(prompter)
        # llm_serve.start()
        # LLM expects single string, need to iterate for batch
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results
    def _reformat_prompt(self, data_list: List[dict], prompt_type: str = None):
        """
        Reformat the prompts in the data_list to generate LLM input.
        All input columns are expected to be strings.
        """
        if prompt_type == "get_identifier":
            self.prompt_template = AtomicTaskGeneratorGetIdentifierPrompt()
            input_prompts = [item.get(self.input_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [self.prompt_template.build_prompt(p) for p in input_prompts]

        elif prompt_type == "get_conclusion":
            self.prompt_template = AtomicTaskGeneratorGetConlcusionPrompt()
            input_prompts = [item.get(self.input_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [self.prompt_template.build_prompt(p) for p in input_prompts]

        elif prompt_type == "init_question":
            self.prompt_template = AtomicTaskGeneratorQuestionPrompt()
            candidate_strs = [item.get("candidate_tasks_str", '') for item in data_list]
            raw_identifiers = [item.get("identifier", '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = []
            for s, raw_id in zip(candidate_strs, raw_identifiers):
                try:
                    # 解析 candidate_tasks_str 字段
                    task_item = json.loads(self._clean_json_block(s))

                    # 清理并解析 identifier 字段
                    clean_id_str = self._clean_json_block(raw_id)
                    identifier_obj = json.loads(clean_id_str)
                    identifier = identifier_obj.get("content_identifier", "Unknown")

                    prompts.append(
                        self.prompt_template.build_prompt(identifier, task_item["conclusion"], task_item["R"])
                    )
                except Exception as e:
                    print(f"[WARN] Failed to parse candidate_tasks_str or identifier: {e} | value: {s} | id: {raw_id}")
                    prompts.append("")  # fallback

        elif prompt_type == "clean_qa":
            self.prompt_template = AtomicTaskGeneratorCleanQAPrompt()
            questions = [item.get(self.output_question_key, '') for item in data_list]
            answers = [item.get(self.output_answer_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [
                self.prompt_template.build_prompt({"question": q, "original_answer": a})
                for q, a in zip(questions, answers)
            ]
        elif prompt_type == "llm_answer":
            self.prompt_template = AtomicTaskGeneratorAnswerPrompt()
            questions = [item.get(self.output_question_key, '') for item in data_list]
            system_prompt = ""
            prompts = [
                self.prompt_template.build_prompt(question) for question in questions
            ]
        elif prompt_type == "get_recall_score":
            self.prompt_template = AtomicTaskGeneratorRecallScorePrompt()
            golden_answers = [item.get(self.output_refined_answer_key, '') for item in data_list]
            llm_answers = [item.get(self.output_llm_answer_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [
                self.prompt_template.build_prompt(golden_answer, llm_answer) for golden_answer, llm_answer in zip(golden_answers, llm_answers)
            ]
        elif prompt_type == "get_golden_answer_score":
            self.prompt_template = AtomicTaskGeneratorRecallScorePrompt()
            golden_answers = [item.get(self.output_refined_answer_key, '') for item in data_list]
            llm_answers = [item.get(self.output_golden_doc_answer_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [
                self.prompt_template.build_prompt(golden_answer, llm_answer) for golden_answer, llm_answer in zip(golden_answers, llm_answers)
            ]
        elif prompt_type == "more_optional_answer":
            self.prompt_template = AtomicTaskGeneratorOptionalAnswerPrompt()
            answers = [item.get(self.output_refined_answer_key, '') for item in data_list]
            system_prompt = self.prompt_template.build_system_prompt()
            prompts = [
                self.prompt_template.build_prompt(answer) for answer in answers
            ]
        elif prompt_type == "golden_doc_answer":
            self.prompt_template = AtomicTaskGeneratorGoldenDocAnswerPrompt()
            golden_docs = [item.get(self.input_key, '') for item in data_list]
            questions = [item.get(self.output_question_key, '') for item in data_list]
            system_prompt = ""
            prompts = [
                self.prompt_template.build_prompt(golden_doc, question) 
                for golden_doc, question in zip(golden_docs, questions)
            ]
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        return system_prompt, prompts
    def recall_score(self, data_list: List[dict]):
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "get_recall_score")
        recall_scores = self._generate_from_llm(user_prompts, sys_prompts)
        valid_scores = []
        for score_str in recall_scores:
            try:
                score_dict = json.loads(self._clean_json_block(score_str))
                valid_scores.append(score_dict["answer_score"])
            except Exception as e:
                print("recall score_str error:", score_str, "\nError:", e)
                valid_scores.append(0)
                continue
        return valid_scores

    def recall_score_golden_doc(self, data_list: List[dict]):
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "get_golden_answer_score")
        recall_scores = self._generate_from_llm(user_prompts, sys_prompts)
        valid_scores = []
        for score_str in recall_scores:
            try:
                score_dict = json.loads(self._clean_json_block(score_str))
                valid_scores.append(score_dict["answer_score"])
            except Exception as e:
                print("recall score_str error:", score_str, "\nError:", e)
                valid_scores.append(0)
                continue
        return valid_scores

    def more_optional_answer(self, data_list: List[dict]):
        original_answers = [item.get(self.output_refined_answer_key, '') for item in data_list]
        system_prompt, user_prompts = self._reformat_prompt(data_list, "more_optional_answer")
        optional_answers = self._generate_from_llm(user_prompts, system_prompt)
        valid_answers = []
        for idx, optional_answer in enumerate(optional_answers):
            try:
                if isinstance(optional_answer, str):
                    optional_answer = json.loads(self._clean_json_block(optional_answer))
                    valid_answers.append(optional_answer)
                else:
                    valid_answers.append(optional_answer)
            except Exception as e:
                print(f"Error parsing optional answer: {optional_answer} | Error: {e}")
                valid_answers.append(original_answers[idx] if idx < len(original_answers) else '')
        return valid_answers
    def get_f1_score(self, data_list: List[dict]):

        f1_scores = []
        for item in data_list:
            prediction = item.get(self.output_golden_doc_answer_key)
            ground_truths = item.get(self.output_optional_answer_key)

            final_metric = {"f1": 0, "precision": 0, "recall": 0}

            if ground_truths is None or prediction is None:
                f1_scores.append(final_metric['f1'])
                continue
                
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]

            for ground_truth in ground_truths:
                
                if ground_truth is None:
                    continue

                normalized_prediction = self.normalize_answer(prediction)
                normalized_ground_truth = self.normalize_answer(ground_truth)

                if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue

                if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue

                prediction_tokens = normalized_prediction.split()
                ground_truth_tokens = normalized_ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    continue

                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)

                final_metric["precision"] = max(precision, final_metric["precision"])
                final_metric["recall"] = max(recall, final_metric["recall"])
                final_metric["f1"] = max(f1, final_metric["f1"])

            f1_scores.append(final_metric['f1'])

        return f1_scores
        
        
    def forward_batch_input(
        self,
        data: List[dict],
        input_key: str = "prompts",
        output_question_key: str = "question",
        output_answer_key:str = "answer",
        output_refined_answer_key:str = "refined_answer",
        output_optional_answer_key: str = "optional_answer",
        output_llm_answer_key: str = "llm_answer",
        output_golden_doc_answer_key: str = "golden_doc_answer",
    ):
        self.input_key, self.output_question_key = input_key, output_question_key

        self.output_answer_key, self.output_refined_answer_key, self.output_optional_answer_key = output_answer_key, output_refined_answer_key, output_optional_answer_key

        self.output_llm_answer_key, self.output_golden_doc_answer_key = output_llm_answer_key, output_golden_doc_answer_key

        # Ensure data is a list of dicts
        assert isinstance(data, list), "Input data must be a list"
        data_list = data

        # === Step 0: Get identifier
        LOG.info("Get identifier...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "get_identifier")
        identifiers = self._generate_from_llm(user_prompts, sys_prompts)

        # === Step 1: Get conclusions
        LOG.info("Get conclusions...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "get_conclusion")
        conclusions = self._generate_from_llm(user_prompts, sys_prompts)

        # === Expand each conclusion into multiple candidate tasks (rows)
        expanded_rows = []
        for idx, (row, output_str, identifier) in enumerate(zip(data_list, conclusions, identifiers)):
            try:
                parsed = json.loads(self._clean_json_block(output_str))
                parsed = parsed[:self.max_per_task] if isinstance(parsed, list) else parsed
            except Exception as e:
                print(f"[WARN] JSON parse failed at idx={idx}: {e} | output: {output_str}")
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

        data_list = expanded_rows

        # === Step 2: Generate questions based on conclusion + reasoning
        LOG.info("Generate questions based on conclusion + reasoning...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "init_question")
        question_outputs = self._generate_from_llm(user_prompts, sys_prompts)

        questions = []
        answers = []
        valid_rows = []

        for idx, (res, row) in enumerate(zip(question_outputs, data_list)):
            try:
                parsed = json.loads(self._clean_json_block(res))
            except Exception as e:
                print(f"[WARN] Failed to parse question JSON at idx={idx}: {e} | res: {res}")
                continue

            if isinstance(parsed, dict) and "Q" in parsed:
                question = parsed["Q"]
                try:
                    task = json.loads(self._clean_json_block(row['candidate_tasks_str']))
                    answer = task.get("conclusion", "")
                except Exception:
                    answer = ""
                row_copy = row.copy()
                row_copy[self.output_question_key] = str(question)
                row_copy[self.output_answer_key] = str(answer)
                valid_rows.append(row_copy)

        if not valid_rows:
            LOG.warning("No valid QA pairs generated.")
            return []

        data_list = valid_rows

        # === Step 3: Clean QA
        LOG.info("Clean QA...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "clean_qa")
        clean_outputs = self._generate_from_llm(user_prompts, sys_prompts)

        for idx, (item, res) in enumerate(zip(data_list, clean_outputs)):
            try:
                parsed = json.loads(self._clean_json_block(res))
                item[self.output_refined_answer_key] = str(parsed.get("refined_answer", ""))
            except Exception as e:
                print(f"[WARN] Failed to parse cleaned QA at idx={idx}: {e} | res: {res}")
                item[self.output_refined_answer_key] = ""

        # Verify module
        LOG.info("LLM reasoning verify...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "llm_answer")
        llm_answer_results = self._generate_from_llm(user_prompts, sys_prompts)
        
        for item, llm_answer in zip(data_list, llm_answer_results):
            item[self.output_llm_answer_key] = llm_answer
        
        llm_scores = self.recall_score(data_list)
        for item, score in zip(data_list, llm_scores):
            item["llm_score"] = score
        
        # Filter out items with llm_score >= 1
        data_list = [item for item in data_list if item.get("llm_score", 0) < 1]

        if not data_list:
            LOG.warning("No data left after LLM score filtering. All questions were answered correctly by LLM.")
            return []

        LOG.info("Get golden doc answer...")
        sys_prompts, user_prompts = self._reformat_prompt(data_list, "golden_doc_answer")
        llm_answer_results = self._generate_from_llm(user_prompts, sys_prompts)
        
        for item, llm_answer in zip(data_list, llm_answer_results):
            item[self.output_golden_doc_answer_key] = llm_answer


        # golden doc answer verify
        LOG.info("Golden doc LLM verifying...")
        golden_doc_scores = self.recall_score_golden_doc(data_list)
        for item, score in zip(data_list, golden_doc_scores):
            item["golden_doc_score"] = score
        
        # Filter items with golden_doc_score >= 1
        data_list = [item for item in data_list if item.get("golden_doc_score", 0) >= 1]

        # more optional answer
        LOG.info("Generating more optional answer...")
        optional_answers = self.more_optional_answer(data_list)
        for item, optional_answer in zip(data_list, optional_answers):
            item[self.output_optional_answer_key] = optional_answer
        
        # Group by input_key and limit max_question per group
        grouped_data = {}
        for item in data_list:
            key_value = item.get(input_key, '')
            if key_value not in grouped_data:
                grouped_data[key_value] = []
            if len(grouped_data[key_value]) < self.max_question:
                grouped_data[key_value].append(item)
        
        # Flatten back to list
        result_list = []
        for items in grouped_data.values():
            result_list.extend(items)
        
        LOG.info(f"Generated {len(result_list)} QA pairs")
        return result_list