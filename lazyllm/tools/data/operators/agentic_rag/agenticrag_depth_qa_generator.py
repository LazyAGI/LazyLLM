"""AgenticRAG Depth QA Generator operator"""
import json
from typing import List
import lazyllm
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
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 获取或创建 agenticrag 组（确保所有模块共享同一个组）
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


class AgenticRAGDepthQAGenerator(agenticrag):
    """
    Operator for generating deeper questions based on existing QA pairs.
    该算子以已有问答生成更深度的问题。
    """

    def __init__(
            self,
            llm = None,
            n_rounds: int = 2,
            **kwargs
    ):
        super().__init__()
        self.n_rounds = n_rounds
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
                "该算子以已有问答生成更深度的问题。\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（默认值：\"question\"）\n"
                "- output_key: 输出字段名（默认值：\"depth_question\"）\n"
            )
        elif lang == "en":
            return (
                "This operator generates deeper questions based on existing QA pairs.\n"
                "Input Parameters:\n"
                "- input_key: Field name for the input (default: \"question\")\n"
                "- output_key: Field name for the output (default: \"depth_question\")\n"
            )
        else:
            return "DepthQAGenerator generate deeper questions based on existing QA pairs."

    def _clean_json_block(self, item: str) -> str:
        """Remove JSON code block markers"""
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def recall_score(self, data_list: List[dict]) -> List[float]:
        """Calculate recall scores for generated questions"""
        prompt_template = DepthQAGeneratorRecallScorePrompt()
        golden_answers = [item.get("refined_answer", "") for item in data_list]
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
            input_key: str = "question",
            output_key: str = "depth_question",
    ) -> List[dict]:
        """
        Process data to generate depth QA pairs.

        Args:
            data: List of dict containing questions
            input_key: Key for input questions
            output_key: Key for output depth questions

        Returns:
            List of dict with generated depth QA pairs
        """
        # Ensure data is a list of dicts
        assert isinstance(data, list), "Input data must be a list"
        data_list = data.copy()

        # Get identifier if not present
        if not data_list or "identifier" not in data_list[0]:
            LOG.info("Getting identifiers...")
            prompt_template = DepthQAGeneratorGetIdentifierPrompt()
            input_prompts = [item.get(input_key, "") for item in data_list]
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
            identifiers = self._generate_from_llm(user_prompts, system_prompt)
            
            for item, identifier in zip(data_list, identifiers):
                item["identifier"] = identifier

        # Iterative depth question generation
        for round_id in range(1, self.n_rounds + 1):
            LOG.info(f"=== Iteration Round {round_id} ===")

            identifier_key = "identifier" if round_id == 1 else f"new_identifier_{round_id - 1}"
            new_identifier_key = f"new_identifier_{round_id}"
            relation_key = f"relation_{round_id}"

            # Step 1: Generate backward task (relation and superset)
            LOG.info(f"Generating backward tasks (round {round_id})...")
            prompt_template = DepthQAGeneratorBackwardTaskPrompt()
            input_prompts = [item.get(identifier_key, "") for item in data_list]
            user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
            backward_results = self._generate_from_llm(user_prompts, "")

            # Parse backward results
            new_data_list = []
            for item, result in zip(data_list, backward_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "identifier" in result and "relation" in result:
                        new_item = item.copy()
                        new_item[new_identifier_key] = result["identifier"]
                        new_item[relation_key] = result["relation"]
                        new_data_list.append(new_item)
                    else:
                        LOG.warning(f"[Skipped]: Invalid backward result")
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to parse backward result: {e}")
                    continue

            if not new_data_list:
                LOG.warning(f"No valid data after backward task generation in round {round_id}")
                break

            data_list = new_data_list

            # Step 2: Check if superset is valid
            LOG.info(f"Checking superset validity (round {round_id})...")
            prompt_template = DepthQAGeneratorSupersetCheckPrompt()
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [
                prompt_template.build_prompt(
                    item.get(new_identifier_key, ""),
                    item.get(relation_key, ""),
                    item.get(identifier_key, "")
                )
                for item in data_list
            ]
            check_results = self._generate_from_llm(user_prompts, system_prompt)

            # Filter valid supersets
            new_data_list = []
            for item, result in zip(data_list, check_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and result.get("new_query") == "valid":
                        new_data_list.append(item)
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to check superset: {e}")
                    continue

            if not new_data_list:
                LOG.warning(f"No valid data after superset check in round {round_id}")
                break

            data_list = new_data_list

            # Step 3: Generate new questions
            LOG.info(f"Generating new questions (round {round_id})...")
            prompt_template = DepthQAGeneratorQuestionPrompt()
            system_prompt = prompt_template.build_system_prompt()
            user_prompts = [
                prompt_template.build_prompt(
                    item.get(new_identifier_key, ""),
                    item.get(relation_key, ""),
                    item.get(identifier_key, "")
                )
                for item in data_list
            ]
            question_results = self._generate_from_llm(user_prompts, system_prompt)

            # Parse questions
            new_data_list = []
            question_key = f"{output_key}_{round_id}"
            
            for item, result in zip(data_list, question_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "new_query" in result:
                        new_item = item.copy()
                        new_item[question_key] = result["new_query"]
                        new_data_list.append(new_item)
                except Exception as e:
                    LOG.warning(f"[Error]: Failed to parse question: {e}")
                    continue

            if not new_data_list:
                LOG.warning(f"No valid questions generated in round {round_id}")
                break

            data_list = new_data_list

            # Step 4: Verify with LLM (filter out easy questions)
            LOG.info(f"Verifying questions with LLM (round {round_id})...")
            
            # Ensure refined_answer exists (use answer if refined_answer not present)
            for item in data_list:
                if "refined_answer" not in item and "answer" in item:
                    item["refined_answer"] = item["answer"]
            
            # Generate LLM answers for the new depth questions
            prompt_template = DepthQAGeneratorAnswerPrompt()
            # Use the newly generated question as input
            temp_questions = [item.get(question_key, "") for item in data_list]
            user_prompts = [prompt_template.build_prompt(q) for q in temp_questions]
            llm_answer_results = self._generate_from_llm(user_prompts, "")

            # Add LLM answers to items
            for item, llm_answer in zip(data_list, llm_answer_results):
                item["llm_answer"] = llm_answer

            # Calculate recall scores
            llm_scores = self.recall_score(data_list)
            
            # Filter out questions that LLM can answer (score >= 1)
            new_data_list = []
            for item, score in zip(data_list, llm_scores):
                item["llm_score"] = score
                if score < 1:  # Keep only difficult questions
                    # Clean up temporary fields
                    item.pop("llm_answer", None)
                    item.pop("llm_score", None)
                    new_data_list.append(item)

            if not new_data_list:
                LOG.warning(f"No data left after LLM verification in round {round_id}. All questions were too easy.")
                break

            data_list = new_data_list
            LOG.info(f"Round {round_id} completed. Remaining items: {len(data_list)}")

        LOG.info(f"Depth QA generation completed! Final count: {len(data_list)}")
        return data_list
