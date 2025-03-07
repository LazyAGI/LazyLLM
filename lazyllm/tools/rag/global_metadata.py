from typing import Optional, Any

class GlobalMetadataDesc:
    # max_size MUST be set when data_type is DataType.VARCHAR or DataType.ARRAY
    def __init__(self, data_type: int, element_type: Optional[int] = None,
                 default_value: Optional[Any] = None, max_size: Optional[int] = None):
        self.data_type = data_type
        self.element_type = element_type
        self.default_value = default_value
        self.max_size = max_size

# ---------------------------------------------------------------------------- #

RAG_DOC_ID = 'docid'
RAG_DOC_PATH = 'lazyllm_doc_path'
RAG_DOC_FILE_NAME = 'file_name'
RAG_DOC_FILE_TYPE = 'file_type'
RAG_DOC_FILE_SIZE = 'file_size'
RAG_DOC_CREATION_DATE = 'creation_date'
RAG_DOC_LAST_MODIFIED_DATE = 'last_modified_date'
RAG_DOC_LAST_ACCESSED_DATE = 'last_accessed_date'

RAG_SYSTEM_META_KEYS = set([RAG_DOC_ID, RAG_DOC_PATH, RAG_DOC_FILE_NAME, RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE,
                            RAG_DOC_CREATION_DATE, RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE])
