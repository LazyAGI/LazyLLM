"""QA Extractor operator - Extract QA pairs and convert to Alpaca format"""
import json
from pathlib import Path
from typing import Optional, List
import pandas as pd
from lazyllm import LOG
from ...base_data import DataOperatorRegistry


@DataOperatorRegistry.register(one_item=False, tag='knowledge_cleaning')
class QAExtractor:
    """
    Extract QA pairs from structured data and convert to Alpaca fine-tuning format.
    从QA_pairs字段提取问答对，转换为Alpaca微调格式。
    """

    def __init__(
            self,
            input_qa_key: str = "QA_pairs",
            output_json_file: Optional[str] = None,
            input_instruction: Optional[str] = "Please answer the following question based on the provided information.",
    ):
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
                "输出符合Stanford Alpaca标准的instruction-input-output格式。"
            )
        else:
            return (
                "QA Extractor - Convert nested QA_pairs to Alpaca fine-tuning format\n"
                "Extracts question-answer pairs from structured data and outputs\n"
                "in Stanford Alpaca standard instruction-input-output format."
            )

    def _extract_qa(self, row, key_inst: str, key_q: str, key_a: str) -> List[dict]:
        """Core extraction logic"""
        qa_data = row.get(self.qa_key)
        if not qa_data:
            return []

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
                key_inst: self.instruction,
                key_q: question,
                key_a: answer
            }
            results.append(item)
        return results

    def _load_from_files(self, df):
        """Load QA data from chunk files"""
        path_keys = ['enhanced_chunk_path', 'cleaned_chunk_path', 'chunk_path']
        path_col = next((k for k in path_keys if k in df.columns), None)

        if not path_col:
            raise ValueError(f"Need one of these fields: {path_keys}")

        rows = []
        for _, row in df.iterrows():
            file_path = row[path_col]
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

        return pd.DataFrame(rows)

    def __call__(
            self,
            data,
            output_instruction_key: Optional[str] = "instruction",
            output_question_key: Optional[str] = "input",
            output_answer_key: Optional[str] = "output",
    ) -> List[dict]:
        """
        Extract QA pairs.

        Args:
            data: List of dict or pandas DataFrame
            output_instruction_key: Key for instruction column
            output_question_key: Key for question column
            output_answer_key: Key for answer column

        Returns:
            List of dict with extracted QA pairs
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        LOG.info("Starting QA extraction from QA pairs")

        # If no QA_pairs column, load from files
        if self.qa_key not in df.columns:
            df = self._load_from_files(df)

        # Extract all QA pairs
        all_qas = []
        for _, row in df.iterrows():
            qas = self._extract_qa(
                row,
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

