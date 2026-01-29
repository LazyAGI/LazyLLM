"""KBC Text Cleaner operator"""
import pandas as pd
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.kbcleaning import KnowledgeCleanerPrompt


funcs = data_register.new_group('function')
classes = data_register.new_group('class')
class KBCTextCleaner(classes):
    """
    Knowledge cleaner for RAG to make content more accurate, reliable and readable.
    知识清洗算子：对原始知识内容进行标准化处理。
    """

    def __init__(
            self,
            llm_serving=None,
            lang: str = "en",
            prompt_template=None,
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识清洗算子：对原始知识内容进行标准化处理，包括HTML标签清理、特殊字符规范化、"
                "链接处理和结构优化，提升RAG知识库的质量。"
            )
        elif lang == "en":
            return (
                "Knowledge Cleaning Operator: Standardizes raw HTML/text content for RAG quality improvement."
            )
        else:
            return "Knowledge cleaning operator for RAG content standardization."

    def _generate_from_llm(self, user_prompts, system_prompt=""):
        """Helper to call LLM serving"""
        if self.llm_serving is None:
            raise ValueError("LLM serving is not configured")
        return self.llm_serving.generate_from_input(user_prompts, system_prompt)

    def __call__(
            self,
            data,
            input_key: str = "raw_chunk",
            output_key: str = "cleaned_chunk",
    ):
        """
        Clean raw text content.

        Args:
            data: List of dict or pandas DataFrame
            input_key: Key for input raw content
            output_key: Key for output cleaned content

        Returns:
            List of dict with cleaned content added
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        raw_contents = dataframe[input_key].tolist()
        formatted_prompts = [self.prompt_template.build_prompt(content) for content in raw_contents]
        cleaned = self._generate_from_llm(formatted_prompts, "")

        # Extract content between <cleaned_start> and <cleaned_end>
        cleaned_extracted = [
            str(text).split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
            if '<cleaned_start>' in str(text) and '<cleaned_end>' in str(text)
            else str(text).strip()
            for text in cleaned
        ]

        dataframe[output_key] = cleaned_extracted
        LOG.info("Text cleaning completed!")
        return dataframe.to_dict('records')

