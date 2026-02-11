import json
from pathlib import Path
from typing import Optional, List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# Get or create kbc (knowledge base cleaning) group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCLoadQAData(kbc):
    def __init__(self, qa_key: str = "QA_pairs", **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.qa_key = qa_key

    def forward(
        self,
        data: dict,
        **kwargs
    ) -> dict:
        # Check if QA data already exists in the data
        if self.qa_key in data:
            return {**data, '_qa_data': data.get(self.qa_key)}

        # Try to load from chunk files
        path_keys = ['enhanced_chunk_path', 'cleaned_chunk_path', 'chunk_path']
        
        for path_key in path_keys:
            file_path = data.get(path_key)
            if not file_path or not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks = chunks if isinstance(chunks, list) else [chunks]

                    for chunk in chunks:
                        if self.qa_key in chunk:
                            return {
                                **data,
                                '_qa_data': chunk[self.qa_key],
                                '_source_file': file_path
                            }
            except Exception as e:
                LOG.error(f"Failed to load {file_path}: {e}")
                continue

        # No QA data found
        return {**data, '_qa_data': None}


class KBCExtractQAPairs(kbc):
    def __init__(
        self,
        qa_key: str = "QA_pairs",
        instruction: str = "Please answer the following question based on the provided information.",
        **kwargs
    ):
        super().__init__(_concurrency_mode='process', **kwargs)
        self.qa_key = qa_key
        self.instruction = instruction

    def forward(
        self,
        data: dict,
        output_instruction_key: str = "instruction",
        output_question_key: str = "input",
        output_answer_key: str = "output",
        **kwargs
    ) -> List[dict]:
        qa_data = data.get('_qa_data')
        if not qa_data:
            return []

        # Extract qa_pairs - handle both dict with 'qa_pairs' key and direct list
        qa_list = qa_data.get('qa_pairs', []) if isinstance(qa_data, dict) else qa_data
        if not isinstance(qa_list, list):
            qa_list = [qa_list] if isinstance(qa_list, dict) else []

        results = []
        for qa in qa_list:
            if not isinstance(qa, dict):
                continue

            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            
            if not question or not answer:
                continue

            item = {
                output_instruction_key: self.instruction,
                output_question_key: question,
                output_answer_key: answer
            }
            results.append(item)

        return results


class KBCSaveQAResults(kbc):
    def __init__(self, output_json_file: Optional[str] = None, **kwargs):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.output_json_file = output_json_file

    def forward_batch_input(
        self,
        data: List[dict],
        **kwargs
    ) -> List[dict]:
        if not self.output_json_file or not data:
            return data

        try:
            output_path = Path(self.output_json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            LOG.info(f"Saved QA results to {output_path}")
        except Exception as e:
            LOG.error(f"Failed to save QA results: {e}")

        return data
