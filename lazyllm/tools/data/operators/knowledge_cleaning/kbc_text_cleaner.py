"""KBC Text Cleaner operator"""
from typing import List
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register
from ...prompts.kbcleaning import KnowledgeCleanerPrompt

# 复用已存在的 kbc 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')


class KBCTextCleaner(kbc):
    """
    Knowledge cleaner for RAG to make content more accurate, reliable and readable.
    知识清洗算子：对原始知识内容进行标准化处理。
    """

    def __init__(
            self,
            llm=None,
            lang: str = "en",
            prompt_template=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
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
        if self.llm is None:
            raise ValueError("LLM is not configured")
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results

    def forward_batch_input(
            self,
            data: List[dict],
            input_key: str = "raw_chunk",
            output_key: str = "cleaned_chunk",
    ) -> List[dict]:
        """
        Clean raw text content.

        Args:
            data: List of dict
            input_key: Key for input raw content
            output_key: Key for output cleaned content

        Returns:
            List of dict with cleaned content added
        """
        assert isinstance(data, list), "Input data must be a list of dict"

        raw_contents = [item.get(input_key, "") for item in data]
        formatted_prompts = [self.prompt_template.build_prompt(content) for content in raw_contents]
        cleaned = self._generate_from_llm(formatted_prompts, "")

        # Extract content between <cleaned_start> and <cleaned_end>
        cleaned_extracted = [
            str(text).split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
            if '<cleaned_start>' in str(text) and '<cleaned_end>' in str(text)
            else str(text).strip()
            for text in cleaned
        ]

        results = []
        for item, cleaned_text in zip(data, cleaned_extracted):
            new_item = item.copy()
            new_item[output_key] = cleaned_text
            results.append(new_item)

        LOG.info("Text cleaning completed!")
        return results
