from lazyllm.tracing.semantics import SemanticType

SEMANTIC_TO_LANGFUSE_OBSERVATION_TYPE = {
    SemanticType.AGENT: 'chain',
    SemanticType.LLM: 'generation',
    SemanticType.RETRIEVER: 'retriever',
    SemanticType.EMBEDDING: 'embedding',
    SemanticType.TOOL: 'tool',
    SemanticType.RERANK: 'span',
    SemanticType.WORKFLOW_CONTROL: 'chain',
}
