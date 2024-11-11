from typing import Optional, Any

class GlobalMetadataDesc:
    DTYPE_VARCHAR = 0
    DTYPE_ARRAY = 1
    DTYPE_INT32 = 2

    # max_size MUST be set when data_type is DTYPE_VARCHAR or DTYPE_ARRAY
    def __init__(self, data_type: int, element_type: Optional[int] = None,
                 default_value: Optional[Any] = None, max_size: Optional[int] = None):
        self.data_type = data_type
        self.element_type = element_type
        self.default_value = default_value
        self.max_size = max_size

# ---------------------------------------------------------------------------- #

RAG_DOC_ID = 'docid'
RAG_DOC_PATH = 'lazyllm_doc_path'
