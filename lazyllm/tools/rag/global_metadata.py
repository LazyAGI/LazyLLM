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
