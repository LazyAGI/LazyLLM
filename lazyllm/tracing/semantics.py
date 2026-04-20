from typing import Final


class SemanticType:
    LLM: Final[str] = 'llm'
    AGENT: Final[str] = 'agent'
    RETRIEVER: Final[str] = 'retriever'
    EMBEDDING: Final[str] = 'embedding'
    TOOL: Final[str] = 'tool'
    RERANK: Final[str] = 'rerank'
    WORKFLOW_CONTROL: Final[str] = 'workflow_control'


__all__ = ['SemanticType']
