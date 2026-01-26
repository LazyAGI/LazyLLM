"""KBC Text Cleaner Batch operator"""
import json
import pandas as pd
from lazyllm import LOG
from ...base_data import DataOperatorRegistry
from ...prompts.kbcleaning import KnowledgeCleanerPrompt


@DataOperatorRegistry.register(one_item=False, tag='knowledge_cleaning')
class KBCTextCleanerBatch:
    """
    Batch knowledge cleaner for RAG.
    批量知识清洗算子。
    """

    def __init__(
            self,
            llm_serving=None,
            lang: str = "en",
            prompt_template=None,
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.prompts = KnowledgeCleanerPrompt(lang=lang)
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量知识清洗算子：对原始知识内容进行标准化处理。"
            )
        elif lang == "en":
            return (
                "Batch Knowledge Cleaning Operator for RAG content standardization."
            )
        else:
            return "Batch knowledge cleaning operator for RAG content standardization"

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def _reformat_prompt_from_path(self, chunk_path: str):
        """Load and reformat prompts from file path"""
        if chunk_path.endswith(".json"):
            dataframe = pd.read_json(chunk_path)
        elif chunk_path.endswith(".jsonl"):
            dataframe = pd.read_json(chunk_path, lines=True)
        else:
            raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")

        if "raw_chunk" not in dataframe.columns:
            raise KeyError("'raw_chunk' field not found in the input file.")

        raw_contents = dataframe["raw_chunk"].tolist()
        inputs = [self.prompts.build_prompt(content) for content in raw_contents]
        return raw_contents, inputs

    def __call__(
            self,
            data,
            input_key: str = "chunk_path",
            output_key: str = "cleaned_chunk_path",
    ):
        """
        Batch clean text content from chunk files.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input chunk file paths
            output_key: Key for output cleaned chunk file paths

        Returns:
            List of dict with cleaned chunk paths added
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        chunk_paths = dataframe[input_key].tolist()

        for chunk_path in chunk_paths:
            if chunk_path:
                raw_chunks, formatted_prompts = self._reformat_prompt_from_path(chunk_path)
                cleaned = self._generate_from_llm(formatted_prompts, "")

                # Extract content between <cleaned_start> and <cleaned_end>
                cleaned_extracted = [
                    text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
                    if '<cleaned_start>' in str(text) and '<cleaned_end>' in str(text)
                    else str(text).strip()
                    for text in cleaned
                ]

                json_items = [{
                    "raw_chunk": raw_chunk,
                    "cleaned_chunk": cleaned_chunk
                } for raw_chunk, cleaned_chunk in zip(raw_chunks, cleaned_extracted)]

                with open(chunk_path, "w", encoding="utf-8") as f:
                    json.dump(json_items, f, ensure_ascii=False, indent=4)
                LOG.info(f"Successfully cleaned contents in {chunk_path}")

        dataframe[output_key] = chunk_paths
        LOG.info("Batch text cleaning completed!")
        return dataframe.to_dict('records')

