
# Import Pipelines
from .pipelines import (  # noqa: F401
    AgenticRAGPipeline,
    AgenticRAGDepthPipeline,
    AgenticRAGWidthPipeline,
    KBCleaningPipeline,
    KBCleaningBatchPipeline,
    EmbeddingSynthesisPipeline,
    EmbeddingFineTunePipeline,
    RerankerSynthesisPipeline,
    RerankerFromEmbeddingPipeline,
    RerankerFineTunePipeline,
)

# Import AgenticRAG operators
from .operators.agentic_rag import (  # noqa: F401
    AgenticRAGQAF1SampleEvaluator,
    AgenticRAGAtomicTaskGenerator,
    AgenticRAGDepthQAGenerator,
    AgenticRAGWidthQAGenerator,
)

# Import Knowledge Cleaning operators
from .operators.knowledge_cleaning import (  # noqa: F401
    KBCChunkGenerator,
    KBCChunkGeneratorBatch,
    KBCTextCleaner,
    KBCTextCleanerBatch,
    FileOrURLToMarkdownConverterBatch,
    FileOrURLToMarkdownConverterAPI,
    KBCMultiHopQAGeneratorBatch,
    QAExtractor,
)

# Import Embedding Synthesis operators
from .operators.embedding_synthesis import (  # noqa: F401
    EmbeddingQueryGenerator,
    EmbeddingHardNegativeMiner,
    EmbeddingDataFormatter,
    EmbeddingDataAugmentor,
    EmbeddingTrainTestSplitter,
)

# Import Reranker Synthesis operators
from .operators.reranker_synthesis import (  # noqa: F401
    RerankerQueryGenerator,
    RerankerHardNegativeMiner,
    RerankerDataFormatter,
    RerankerTrainTestSplitter,
    RerankerFromEmbeddingConverter,
)

# Import prompts
from .prompts import PromptABC  # noqa: F401
import importlib
import lazyllm
from .base_data import LazyLLMDataBase, data_register
from .operators import demo_ops  # noqa: F401

def __getattr__(name):
    if name == 'pipelines':
        return importlib.import_module('.pipelines', __package__)
    if name in lazyllm.data:
        return lazyllm.data[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['LazyLLMDataBase', 'data_register', 'PromptABC']
# Add AgenticRAG operators to __all__
__all__.extend([
    'AgenticRAGQAF1SampleEvaluator',
    'AgenticRAGAtomicTaskGenerator',
    'AgenticRAGDepthQAGenerator',
    'AgenticRAGWidthQAGenerator',
])

# Add Knowledge Cleaning operators to __all__
__all__.extend([
    'KBCChunkGenerator',
    'KBCChunkGeneratorBatch',
    'KBCTextCleaner',
    'KBCTextCleanerBatch',
    'FileOrURLToMarkdownConverterBatch',
    'FileOrURLToMarkdownConverterAPI',
    'KBCMultiHopQAGeneratorBatch',
    'QAExtractor',
])

# Add Embedding Synthesis operators to __all__
__all__.extend([
    'EmbeddingQueryGenerator',
    'EmbeddingHardNegativeMiner',
    'EmbeddingDataFormatter',
    'EmbeddingDataAugmentor',
    'EmbeddingTrainTestSplitter',
])

# Add Reranker Synthesis operators to __all__
__all__.extend([
    'RerankerQueryGenerator',
    'RerankerHardNegativeMiner',
    'RerankerDataFormatter',
    'RerankerTrainTestSplitter',
    'RerankerFromEmbeddingConverter',
])

# Add Pipelines to __all__
__all__.extend([
    'AgenticRAGPipeline',
    'AgenticRAGDepthPipeline',
    'AgenticRAGWidthPipeline',
    'KBCleaningPipeline',
    'KBCleaningBatchPipeline',
    'EmbeddingSynthesisPipeline',
    'EmbeddingFineTunePipeline',
    'RerankerSynthesisPipeline',
    'RerankerFromEmbeddingPipeline',
    'RerankerFineTunePipeline',
])

