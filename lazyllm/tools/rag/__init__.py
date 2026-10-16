from importlib import import_module

# Readers decode documents independently of ingestion and vector stores.
_EXPORTS = {
    'Document': ('.document', 'Document'),
    'DocServer': ('.doc_service', 'DocServer'),
    'DeleteRequest': ('.doc_service', 'DeleteRequest'),
    'ReparseRequest': ('.doc_service', 'ReparseRequest'),
    'AddRequest': ('.doc_service', 'AddRequest'),
    'UploadRequest': ('.doc_service', 'UploadRequest'),
    'AddFileItem': ('.doc_service', 'AddFileItem'),
    'TransferItem': ('.doc_service', 'TransferItem'),
    'TransferRequest': ('.doc_service', 'TransferRequest'),
    'MetadataPatchItem': ('.doc_service', 'MetadataPatchItem'),
    'MetadataPatchRequest': ('.doc_service', 'MetadataPatchRequest'),
    'DocStatus': ('.doc_service', 'DocStatus'),
    'DocServiceError': ('.doc_service', 'DocServiceError'),
    'SourceType': ('.doc_service', 'SourceType'),
    'GraphDocument': ('.graph_document', 'GraphDocument'),
    'UrlGraphDocument': ('.graph_document', 'UrlGraphDocument'),
    'Retriever': ('.retriever', 'Retriever'),
    'TempDocRetriever': ('.retriever', 'TempDocRetriever'),
    'ContextRetriever': ('.retriever', 'ContextRetriever'),
    'WeightedRetriever': ('.retriever', 'WeightedRetriever'),
    'PriorityRetriever': ('.retriever', 'PriorityRetriever'),
    'GraphRetriever': ('.graph_retriever', 'GraphRetriever'),
    'Reranker': ('.rerank', 'Reranker'),
    'register_reranker': ('.rerank', 'register_reranker'),
    'SentenceSplitter': ('.transform', 'SentenceSplitter'),
    'LLMParser': ('.transform', 'LLMParser'),
    'NodeTransform': ('.transform', 'NodeTransform'),
    'TransformArgs': ('.transform', 'TransformArgs'),
    'AdaptiveTransform': ('.transform', 'AdaptiveTransform'),
    'CharacterSplitter': ('.transform', 'CharacterSplitter'),
    'RecursiveSplitter': ('.transform', 'RecursiveSplitter'),
    'MarkdownSplitter': ('.transform', 'MarkdownSplitter'),
    'CodeSplitter': ('.transform', 'CodeSplitter'),
    'JSONSplitter': ('.transform', 'JSONSplitter'),
    'YAMLSplitter': ('.transform', 'YAMLSplitter'),
    'HTMLSplitter': ('.transform', 'HTMLSplitter'),
    'XMLSplitter': ('.transform', 'XMLSplitter'),
    'GeneralCodeSplitter': ('.transform', 'GeneralCodeSplitter'),
    'JSONLSplitter': ('.transform', 'JSONLSplitter'),
    'register_similarity': ('.similarity', 'register_similarity'),
    'DocNode': ('.doc_node', 'DocNode'),
    'RichDocNode': ('.doc_node', 'RichDocNode'),
    'PDFReader': ('.readers', 'PDFReader'),
    'DocxReader': ('.readers', 'DocxReader'),
    'HWPReader': ('.readers', 'HWPReader'),
    'PPTXReader': ('.readers', 'PPTXReader'),
    'ImageReader': ('.readers', 'ImageReader'),
    'IPYNBReader': ('.readers', 'IPYNBReader'),
    'EpubReader': ('.readers', 'EpubReader'),
    'MarkdownReader': ('.readers', 'MarkdownReader'),
    'MboxReader': ('.readers', 'MboxReader'),
    'PandasCSVReader': ('.readers', 'PandasCSVReader'),
    'PandasExcelReader': ('.readers', 'PandasExcelReader'),
    'VideoAudioReader': ('.readers', 'VideoAudioReader'),
    'MineruPDFReader': ('.readers', 'MineruPDFReader'),
    'inject_reader_config': ('.readers', 'inject_reader_config'),
    'SimpleDirectoryReader': ('.dataReader', 'SimpleDirectoryReader'),
    'FileReader': ('.dataReader', 'FileReader'),
    'DocField': ('.global_metadata', 'GlobalMetadataDesc'),
    'DataType': ('.data_type', 'DataType'),
    'IndexBase': ('.index_base', 'IndexBase'),
    'LazyLLMStoreBase': ('.store', 'LazyLLMStoreBase'),
    'SchemaExtractor': ('.doc_to_db', 'SchemaExtractor'),
    'QueryEnhACProcessor': ('.query_enh_ac', 'QueryEnhACProcessor'),
}


def __getattr__(name):
    if name == 'add_post_action_for_default_reader':
        return __getattr__('SimpleDirectoryReader').add_post_action_for_default_reader
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = ['add_post_action_for_default_reader', *_EXPORTS]
