"""QA Extractor operator - Extract QA pairs and convert to Alpaca format"""
import json
from pathlib import Path
from typing import Optional, List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class QAExtractor(kbc):
    """
    Extract QA pairs from structured data and convert to Alpaca fine-tuning format.
    从QA_pairs字段提取问答对，转换为Alpaca微调格式。
    """

    def __init__(
            self,
            input_qa_key: str = "QA_pairs",
            output_json_file: Optional[str] = None,
            input_instruction: Optional[str] = "Please answer the following question based on the provided information.",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.qa_key = input_qa_key
        self.output_json_file = output_json_file
        self.instruction = input_instruction

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "QA对提取器 - 将嵌套的QA_pairs转换为Alpaca微调格式\n"
                "核心功能:\n"
                "从结构化的QA对数据中提取问答内容，自动整合推理步骤和支持事实，\n"
                "输出符合Stanford Alpaca标准的instruction-input-output格式。\n\n"
                "初始化参数:\n"
                "• input_qa_key: QA对的字段名 (默认: 'QA_pairs')\n"
                "• output_json_file: 输出JSON文件路径 (可选)\n"
                "• input_instruction: 统一的指令前缀\n"
                "• include_fields: 要包含的字段列表 (默认: ['question', 'reasoning_steps', 'supporting_facts'])"
            )
        else:
            return (
                "QA Extractor - Convert nested QA_pairs to Alpaca fine-tuning format\n"
                "Extracts question-answer pairs from structured data and outputs\n"
                "in Stanford Alpaca standard instruction-input-output format.\n\n"
                "Parameters:\n"
                "• input_qa_key: Field name for QA pairs (default: 'QA_pairs')\n"
                "• output_json_file: Output JSON path (optional)\n"
                "• input_instruction: Unified instruction prefix\n"
                "• include_fields: Fields to include (default: ['question', 'reasoning_steps', 'supporting_facts'])"
            )

    def _parse_fields(self, input_fields) -> Optional[List[str]]:
        """解析要包含的字段"""
        if input_fields is None:
            return None  # 包含所有默认字段
        if isinstance(input_fields, list):
            return input_fields
        if isinstance(input_fields, str):
            return [f.strip() for f in input_fields.split(',') if f.strip()] if input_fields.strip() else []
        return None

    def _extract_qa(
            self,
            row: dict,
            fields: Optional[List[str]],
            key_inst: str,
            key_q: str,
            key_a: str
    ) -> List[dict]:
        """Core extraction logic"""
        qa_data = row.get(self.qa_key)
        if not qa_data:
            return []

        qa_list = qa_data.get('qa_pairs', []) if isinstance(qa_data, dict) else qa_data
        if not isinstance(qa_list, list):
            qa_list = [qa_list] if isinstance(qa_list, dict) else []

        results = []
        # 默认字段
        default_fields = ['question', 'reasoning_steps', 'supporting_facts']
        fields = fields if fields is not None else default_fields

        for qa in qa_list:
            if not isinstance(qa, dict):
                continue

            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            if not question or not answer:
                continue

            item = {
                key_inst: self.instruction,
                key_q: question,
                key_a: answer
            }
            results.append(item)
        return results

    def _load_from_files(self, data: List[dict]) -> List[dict]:
        """Load QA data from chunk files"""
        path_keys = ['enhanced_chunk_path', 'cleaned_chunk_path', 'chunk_path']
        
        # Find which key exists in the data
        path_col = None
        if data:
            for k in path_keys:
                if k in data[0]:
                    path_col = k
                    break

        if not path_col:
            raise ValueError(f"Need one of these fields: {path_keys}")

        rows = []
        for item in data:
            file_path = item.get(path_col)
            if not file_path or not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks = chunks if isinstance(chunks, list) else [chunks]

                    for chunk in chunks:
                        if self.qa_key in chunk:
                            rows.append({
                                self.qa_key: chunk[self.qa_key],
                                'source_file': file_path
                            })
            except Exception as e:
                LOG.error(f"Failed to load {file_path}: {e}")

        if not rows:
            raise ValueError("No valid QA data found")

        return rows

    def forward_batch_input(
            self,
            data: List[dict],
            output_instruction_key: str = "instruction",
            output_question_key: str = "input",
            output_answer_key: str = "output",
            include_fields: Optional[str] = None,
    ) -> List[dict]:
        """
        Extract QA pairs.

        Args:
            data: List of dict
            output_instruction_key: Key for instruction column
            output_question_key: Key for question column (context with reasoning)
            output_answer_key: Key for answer column
            include_fields: Fields to include in context, comma-separated string or list
                           Default: ['question', 'reasoning_steps', 'supporting_facts']
                           Set to 'question' to only include the question without context

        Returns:
            List of dict with extracted QA pairs
        """
        assert isinstance(data, list), "Input data must be a list of dict"

        # 检查输出字段名变更并提示
        is_modified = False
        modified_details = []

        if output_question_key != "input":
            is_modified = True
            modified_details.append(f"output_question_key -> '{output_question_key}'")

        if output_answer_key != "output":
            is_modified = True
            modified_details.append(f"output_answer_key -> '{output_answer_key}'")

        if is_modified:
            LOG.warning(
                f"\n{'='*20} Configuration Change Warning {'='*20}\n"
                f"Detected changes in output field names: {', '.join(modified_details)}\n\n"
                f"Please note:\n"
                f"1. [SFT / LLaMA-Factory]: If using LLaMA-Factory, DO NOT modify default keys, "
                f"or update 'dataset_info.json' manually.\n"
                f"2. [Downstream]: Ensure downstream operators use matching keys.\n"
                f"{'='*66}"
            )

        LOG.info("Starting QA extraction from QA pairs")

        # If no QA_pairs column, load from files
        if data and self.qa_key not in data[0]:
            data = self._load_from_files(data)

        # Parse fields to include
        fields = self._parse_fields(include_fields)

        # Extract all QA pairs
        all_qas = []
        for row in data:
            qas = self._extract_qa(
                row,
                fields=fields,
                key_inst=output_instruction_key,
                key_q=output_question_key,
                key_a=output_answer_key
            )
            all_qas.extend(qas)

        LOG.info(f"Extracted {len(all_qas)} QA pairs")

        if not all_qas:
            LOG.warning("No QA pairs found!")
            return []

        # Save JSON (optional)
        if self.output_json_file:
            output_path = Path(self.output_json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qas, f, indent=2, ensure_ascii=False)
            LOG.info(f"Saved to {output_path}")

        return all_qas
