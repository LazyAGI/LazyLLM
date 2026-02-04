"""AgenticRAG Width QA Generator operator"""
import json
from typing import List
import lazyllm
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.agenticrag import (
    WidthQAGeneratorMergePrompt,
    WidthQAGeneratorOriginCheckPrompt,
    WidthQAGeneratorQuestionVerifyPrompt,
    WidthQAGeneratorAnswerPrompt,
    WidthQAGeneratorRecallScorePrompt
)
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 获取或创建 agenticrag 组（确保所有模块共享同一个组）
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


class AgenticRAGWidthQAGenerator(agenticrag):
    """
    Operator for combining two QA pairs to generate new questions.
    该算子用于结合两个问答，生成新的问题。
    """

    def __init__(self, llm = None, **kwargs):
        super().__init__()
        self.llm = llm
    
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

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于结合两个问答，生成新的问题。\n\n"
                "输入参数：\n"
                "- input_question_key: 输入问题字段名（默认值：\"question\"）\n"
                "- input_identifier_key: 输入标识符字段名（默认值：\"identifier\"）\n"
                "- input_answer_key: 输入答案字段名（默认值：\"answer\"）\n"
                "- output_question_key: 输出问题字段名（默认值：\"generated_width_task\")\n"
            )
        elif lang == "en":
            return (
                "This operator combines two QA pairs to generate a new question.\n"
                "Input Parameters:\n"
                "- input_question_key: Field name for the input question (default: \"question\")\n"
                "- input_identifier_key: Field name for the input identifier (default: \"identifier\")\n"
                "- input_answer_key: Field name for the input answer (default: \"answer\")\n"
                "- output_question_key: Field name for the output question (default: \"generated_width_task\")\n"
            )
        else:
            return "WidthQAGenerator combine two QA pairs and generate a new question."

    def _clean_json_block(self, item: str) -> str:
        """Remove JSON code block markers"""
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def recall_score(self, data_list: List[dict]) -> List[float]:
        """Calculate recall scores for generated questions"""
        prompt_template = WidthQAGeneratorRecallScorePrompt()
        golden_answers = [item.get("original_answer", []) for item in data_list]
        llm_answers = [item.get("llm_answer", "") for item in data_list]
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
                    score_dict = json.loads(self._clean_json_block(score_str))
                    valid_scores.append(score_dict["answer_score"])
                except (json.JSONDecodeError, KeyError) as e:
                    LOG.warning(f"Failed to parse score: {score_str}, Error: {e}")
                    valid_scores.append(0)
            else:
                valid_scores.append(0)
        
        return valid_scores

    def forward_batch_input(
            self,
            data: List[dict],
            input_question_key: str = "question",
            input_identifier_key: str = "identifier",
            input_answer_key: str = "answer",
            output_question_key: str = "generated_width_task",
    ) -> List[dict]:
        """
        Process data to generate width QA pairs by merging adjacent QA pairs.

        Args:
            data: List of dict containing questions, identifiers, and answers
            input_question_key: Key for input questions
            input_identifier_key: Key for input identifiers  
            input_answer_key: Key for input answers
            output_question_key: Key for output width questions

        Returns:
            List of dict with generated width QA pairs
        """
        # Ensure data is a list of dicts
        assert isinstance(data, list), "Input data must be a list"
        
        # Step 0: Prepare input batch
        LOG.info("Preparing input batch...")
        input_batch = []
        for i, item in enumerate(data):
            input_batch.append({
                "index": i,
                "question": item.get(input_question_key, ""),
                "content_identifier": item.get(input_identifier_key, ""),
                "golden_answer": item.get(input_answer_key, "")
            })

        if len(input_batch) < 2:
            LOG.warning("Need at least 2 items to merge. Returning empty list.")
            return []

        # Step 1: Merge adjacent QA pairs
        LOG.info(f"Merging {len(input_batch)} items into width questions...")
        prompt_template = WidthQAGeneratorMergePrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt([input_batch[i], input_batch[i + 1]])
            for i in range(len(input_batch) - 1)
        ]
        merge_results = self._generate_from_llm(user_prompts, system_prompt)

        # Parse merge results
        merged_data_list = []
        for idx, result in enumerate(merge_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))
                    # Handle list wrapper
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]

                if not isinstance(result, dict) or "question" not in result or "index" not in result:
                    LOG.warning(f"[Skipped]: Invalid merge result at index {idx}")
                    continue

                indices = result["index"] if isinstance(result["index"], list) else [result["index"]]
                group_items = [input_batch[i] for i in indices if i < len(input_batch)]

                if not group_items:
                    continue

                merged_data_list.append({
                    "question": result["question"],
                    "content_identifier": result.get("content_identifier", ""),
                    "qa_index": indices,
                    "index": idx,
                    "original_answer": [item["golden_answer"] for item in group_items],
                    "original_question": [item["question"] for item in group_items],
                })

            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse merge result at index {idx}: {e}")
                continue

        if not merged_data_list:
            LOG.warning("No valid merged questions generated.")
            return []

        LOG.info(f"Generated {len(merged_data_list)} merged questions.")

        # Step 2: Check if complex questions can be decomposed to original questions
        LOG.info("Checking if complex questions decompose to original questions...")
        prompt_template = WidthQAGeneratorOriginCheckPrompt()
        system_prompt = prompt_template.build_system_prompt()
        check_input_batch = []
        for item in merged_data_list:
            ori_q = item.get("original_question", [])
            check_input_batch.append({
                "index": item.get("index", 0),
                "complex_question": item.get("question", ""),
                "original_questions": ori_q if isinstance(ori_q, list) else [ori_q]
            })
        
        user_prompts = [prompt_template.build_prompt(inp) for inp in check_input_batch]
        check_query_results = self._generate_from_llm(user_prompts, system_prompt)

        # Parse check results
        new_data_list = []
        for item, result in zip(merged_data_list, check_query_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]

                if isinstance(result, dict):
                    state = result.get("state", None)
                    complex_question = result.get("complex_question", item.get("question"))
                    
                    if state == 1:  # Valid decomposition
                        new_item = item.copy()
                        new_item["state"] = state
                        new_item[output_question_key] = complex_question
                        new_data_list.append(new_item)
                else:
                    LOG.warning(f"[Skipped]: Invalid check result")
            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse check result: {e}")
                continue

        if not new_data_list:
            LOG.warning("No valid questions after origin check.")
            return []

        data_list = new_data_list
        LOG.info(f"{len(data_list)} questions passed origin check.")

        # Step 3: Verify questions with LLM
        LOG.info("Verifying questions with LLM...")
        prompt_template = WidthQAGeneratorQuestionVerifyPrompt()
        system_prompt = prompt_template.build_system_prompt()
        verify_input_batch = []
        for item in data_list:
            verify_input_batch.append({
                "index": item.get("index", 0),
                "complex_question": item.get(output_question_key, ""),
            })
        
        user_prompts = [prompt_template.build_prompt(inp) for inp in verify_input_batch]
        question_verify_results = self._generate_from_llm(user_prompts, system_prompt)

        # Parse verification results
        for item, result in zip(data_list, question_verify_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]

                if isinstance(result, dict):
                    item["llm_answer"] = result.get("llm_answer", None)
                else:
                    item["llm_answer"] = None
            except Exception as e:
                LOG.warning(f"[Error]: Failed to parse verification result: {e}")
                item["llm_answer"] = None

        # Step 4: Filter by recall score
        LOG.info("Filtering questions by recall score...")
        llm_scores = self.recall_score(data_list)
        
        # Keep only difficult questions (score < 1)
        result_list = []
        for item, score in zip(data_list, llm_scores):
            item["llm_score"] = score
            if score < 1:
                # Clean up temporary fields
                item.pop("llm_answer", None)
                item.pop("llm_score", None)
                item.pop("state", None)
                result_list.append(item)

        LOG.info(f"Width QA generation completed! Final count: {len(result_list)}")
        return result_list
