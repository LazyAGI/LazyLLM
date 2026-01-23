from importlib import import_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # flake8: noqa: E401
    from .rag import (Document, GraphDocument, UrlGraphDocument, Reranker, Retriever, TempDocRetriever,
                    GraphRetriever, SentenceSplitter, LLMParser)
    from .webpages import WebModule
    from .agent import (
        ToolManager,
        FunctionCall,
        FunctionCallAgent,
        register as fc_register,
        ReactAgent,
        PlanAndSolveAgent,
        ReWOOAgent,
        ModuleTool,
    )
    from .classifier import IntentClassifier
    from .sql import SqlManager, MongoDBManager, DBResult, DBStatus, DBManager
    from .sql_call import SqlCall
    from .tools.http_tool import HttpTool
    from .servers.graphrag.graphrag_server_module import GraphRagServerModule
    from .mcp.client import MCPClient  # noqa NID002
    from .actors import ParameterExtractor, QustionRewrite, CodeGenerator
    from .common import StreamCallHelper
    from .eval import (BaseEvaluator, ResponseRelevancy, Faithfulness, LLMContextRecall,
                    NonLLMContextRecall, ContextRelevance)
    from .http_request import HttpRequest, HttpExecutorResponse
    from .review import get_errors, ChineseCorrector


def __getattr__(name: str):
    if name not in _SUBMOD_MAP_REVERSE and name not in _SUBMOD_MAP:
        raise AttributeError(f"Module 'tools' has no attribute '{name}'")

    if name == 'fc_register':
        agent = import_module('.agent', package=__package__)
        globals()['fc_register'] = value = agent.register
    elif name in _SUBMOD_MAP:
        return import_module(f'.{name}', package=__package__)
    elif name in _SUBMOD_MAP_REVERSE:
        module = import_module(f'.{_SUBMOD_MAP_REVERSE[name]}', package=__package__)
        globals()[name] = value = getattr(module, name)
    return value

_SUBMOD_MAP = {
    'rag': [
        'Document',
        'GraphDocument',
        'UrlGraphDocument',
        'Reranker',
        'TempDocRetriever',
        'Retriever',
        'GraphRetriever',
        'LLMParser',
        'SentenceSplitter'
    ],
    'webpages': ['WebModule'],
    'agent': [
        'ToolManager',
        'ModuleTool',
        'FunctionCall',
        'FunctionCallAgent',
        'fc_register',
        'ReactAgent',
        'PlanAndSolveAgent',
        'ReWOOAgent'
    ],
    'classifier': ['IntentClassifier'],
    'sql': [
        'SqlManager',
        'MongoDBManager',
        'DBManager',
        'DBResult',
        'DBStatus'
    ],
    'sql_call': ['SqlCall'],
    'tools.http_tool': ['HttpTool'],
    'servers.graphrag.graphrag_server_module': ['GraphRagServerModule'],
    'mcp.client': ['MCPClient'],
    'actors': [
        'ParameterExtractor',
        'QustionRewrite',
        'CodeGenerator'
    ],
    'common': [
        'StreamCallHelper'
    ],
    'eval': [
        'BaseEvaluator',
        'ResponseRelevancy',
        'Faithfulness',
        'LLMContextRecall',
        'NonLLMContextRecall',
        'ContextRelevance',
    ],
    'http_request': [
        'HttpRequest',
        'HttpExecutorResponse'
    ],
    'review': [
        'get_errors',
        'ChineseCorrector'
    ],
}
_SUBMOD_MAP_REVERSE = {v: k for k, vs in _SUBMOD_MAP.items() for v in vs}
__all__ = sum(_SUBMOD_MAP.values(), [])
