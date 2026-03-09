import importlib
import lazyllm
from .base_data import LazyLLMDataBase, data_register
from .operators import demo_ops  # noqa: F401
from .operators import preference_ops  # noqa: F401
from .operators import sql_evalhardness  # noqa: F401
from .operators import text2sql_ops  # noqa: F401
from .operators import tool_use_ops  # noqa: F401
from .operators import pt_op  # noqa: F401
from .operators import refine_op  # noqa: F401
from .operators import token_chunker  # noqa: F401
from .operators import filter_op  # noqa: F401
from .operators import cot_ops  # noqa: F401
from .operators import math_ops  # noqa: F401
from .operators import pdf_ops  # noqa: F401
from .operators import enQa_ops  # noqa: F401
from .operators import text2qa_ops  # noqa: F401
from .operators import codegen_ops  # noqa: F401
from .operators import llm_base_ops  # noqa: F401
from .operators import llm_json_ops  # noqa: F401
from .operators import agentic_rag  # noqa: F401
from .operators import embedding_synthesis  # noqa: F401
from .operators import knowledge_cleaning  # noqa: F401
from .operators import reranker_synthesis  # noqa: F401

def __getattr__(name):
    if name == 'pipelines':
        return importlib.import_module('.pipelines', __package__)
    if name in lazyllm.data:
        return lazyllm.data[name]
    raise AttributeError(f'module {name!r} has no attribute {name!r}')

__all__ = ['LazyLLMDataBase', 'data_register']
