from importlib import import_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # flake8: noqa: E401
    from .rag import (Document, GraphDocument, UrlGraphDocument, Reranker, Retriever, TempDocRetriever,
                    GraphRetriever, SentenceSplitter, LLMParser, DocServer,
                    DeleteRequest, ReparseRequest, AddRequest, UploadRequest, AddFileItem,
                    TransferItem, TransferRequest, MetadataPatchItem, MetadataPatchRequest,
                    DocStatus, DocServiceError, SourceType)
    from .webpages import WebModule
    from .fs import (LazyLLMFSBase, CloudFSBufferedFile, CloudFsWatchdog, FS, dynamic_fs_config,
                     FeishuFS, ConfluenceFS, NotionFS, GoogleDriveFS, OneDriveFS, YuqueFS, OnesFS, S3FS,
                     ObsidianFS)
    from .agent import (
        ToolManager,
        FunctionCall,
        FunctionCallAgent,
        register as fc_register,
        LazyLLMAgentBase,
        ReactAgent,
        PlanAndSolveAgent,
        ReWOOAgent,
        ModuleTool,
        SkillManager,
        install_skill,
    )
    from .sandbox import LazyLLMSandboxBase, DummySandbox, SandboxFusion
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
    from .data import data_register
    from .review import get_errors, ChineseCorrector
    from .git import (LazyLLMGitBase, PrInfo, ReviewCommentInfo, Git,
                      GitHub, GitLab, Gitee, GitCode)


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
        'SentenceSplitter',
        'DocServer',
        'DeleteRequest',
        'ReparseRequest',
        'AddRequest',
        'UploadRequest',
        'AddFileItem',
        'TransferItem',
        'TransferRequest',
        'MetadataPatchItem',
        'MetadataPatchRequest',
        'DocStatus',
        'DocServiceError',
        'SourceType',
    ],
    'webpages': ['WebModule'],
    'agent': [
        'ToolManager',
        'ModuleTool',
        'FunctionCall',
        'FunctionCallAgent',
        'fc_register',
        'LazyLLMAgentBase',
        'ReactAgent',
        'PlanAndSolveAgent',
        'ReWOOAgent',
        'SkillManager',
        'install_skill',
    ],
    'sandbox': [
        'LazyLLMSandboxBase',
        'DummySandbox',
        'SandboxFusion'
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
    'data': [
        'data_register'
    ],
    'review': [
        'get_errors',
        'ChineseCorrector'
    ],
    'git': [
        'LazyLLMGitBase',
        'PrInfo',
        'ReviewCommentInfo',
        'Git',
        'GitHub',
        'GitLab',
        'Gitee',
        'GitCode',
        'review',
    ],
    'fs': [
        'LazyLLMFSBase',
        'CloudFSBufferedFile',
        'CloudFsWatchdog',
        'FS',
        'dynamic_fs_config',
        'FeishuFS',
        'ConfluenceFS',
        'NotionFS',
        'GoogleDriveFS',
        'OneDriveFS',
        'YuqueFS',
        'OnesFS',
        'S3FS',
        'ObsidianFS',
    ],
}
_SUBMOD_MAP_REVERSE = {v: k for k, vs in _SUBMOD_MAP.items() for v in vs}
__all__ = sum(_SUBMOD_MAP.values(), [])
