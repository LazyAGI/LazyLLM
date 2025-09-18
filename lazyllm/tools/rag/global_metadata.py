from typing import Optional, Any

class GlobalMetadataDesc:
    """A descriptor for global metadata, defining its type, optional element type, default value, and size constraints.
`class GlobalMetadataDesc`
This class is used to describe metadata properties such as type, optional constraints, and default values. It supports scalar and array data types, with specific size limitations for certain types.

Args:
    data_type (int): The type of the metadata as an integer, representing various data types (e.g., VARCHAR, ARRAY, etc.).
    element_type (Optional[int]): The type of individual elements if `data_type` is an array. Defaults to `None`.
    default_value (Optional[Any]): The default value for the metadata. If not provided, the default will be `None`.
    max_size (Optional[int]): The maximum size or length for the metadata. Required if `data_type` is `VARCHAR` or `ARRAY`.
"""
    # max_size MUST be set when data_type is DataType.VARCHAR or DataType.ARRAY
    def __init__(self, data_type: int, element_type: Optional[int] = None,
                 default_value: Optional[Any] = None, max_size: Optional[int] = None):
        self.data_type = data_type
        self.element_type = element_type
        self.default_value = default_value
        self.max_size = max_size

# ---------------------------------------------------------------------------- #
# RAG system metadata keys
RAG_KB_ID = 'kb_id'
RAG_DOC_ID = 'docid'
RAG_DOC_PATH = 'lazyllm_doc_path'
RAG_DOC_FILE_NAME = 'file_name'
RAG_DOC_FILE_TYPE = 'file_type'
RAG_DOC_FILE_SIZE = 'file_size'
RAG_DOC_CREATION_DATE = 'creation_date'
RAG_DOC_LAST_MODIFIED_DATE = 'last_modified_date'
RAG_DOC_LAST_ACCESSED_DATE = 'last_accessed_date'

RAG_SYSTEM_META_KEYS = set([RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID, RAG_DOC_FILE_NAME, RAG_DOC_FILE_TYPE,
                            RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE, RAG_DOC_LAST_MODIFIED_DATE,
                            RAG_DOC_LAST_ACCESSED_DATE])
