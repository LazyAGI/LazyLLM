# Knowledge cleaning operators for data pipeline
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from ...base_data import data_register

# 获取或创建 kbc (knowledge base cleaning) 组（确保所有模块共享同一个组）
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'kbc' in LazyLLMRegisterMetaClass.all_clses['data']:
    kbc = LazyLLMRegisterMetaClass.all_clses['data']['kbc'].base
else:
    kbc = data_register.new_group('kbc')

from .kbc_chunk_generator import KBCChunkGenerator
from .kbc_chunk_generator_batch import KBCChunkGeneratorBatch
from .kbc_text_cleaner import KBCTextCleaner
from .kbc_text_cleaner_batch import KBCTextCleanerBatch
from .file_or_url_to_markdown_converter_batch import FileOrURLToMarkdownConverterBatch
from .file_or_url_to_markdown_converter_api import FileOrURLToMarkdownConverterAPI
from .kbc_multihop_qa_generator_batch import KBCMultiHopQAGeneratorBatch
from .qa_extract import QAExtractor

__all__ = [
    'kbc',
    'KBCChunkGenerator',
    'KBCChunkGeneratorBatch',
    'KBCTextCleaner',
    'KBCTextCleanerBatch',
    'FileOrURLToMarkdownConverterBatch',
    'FileOrURLToMarkdownConverterAPI',
    'KBCMultiHopQAGeneratorBatch',
    'QAExtractor',
]
