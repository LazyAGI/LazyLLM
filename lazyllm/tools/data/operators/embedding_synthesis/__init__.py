# Embedding Synthesis Operators
from ...base_data import data_register
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 创建 embedding 组（复用已存在的组）
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')

from .embedding_query_generator import EmbeddingQueryGenerator  # noqa: F401
from .embedding_hard_negative_miner import EmbeddingHardNegativeMiner  # noqa: F401
from .embedding_data_formatter import EmbeddingDataFormatter, EmbeddingTrainTestSplitter  # noqa: F401
from .embedding_data_augmentor import EmbeddingDataAugmentor  # noqa: F401

__all__ = [
    'embedding',
    'EmbeddingQueryGenerator',
    'EmbeddingHardNegativeMiner',
    'EmbeddingDataFormatter',
    'EmbeddingDataAugmentor',
    'EmbeddingTrainTestSplitter',
]
