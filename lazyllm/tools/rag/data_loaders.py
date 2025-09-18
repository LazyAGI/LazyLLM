from typing import List, Optional, Dict
from lazyllm import LOG

from .doc_node import DocNode, ImageDocNode
from .store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .dataReader import SimpleDirectoryReader

class DirectoryReader:
    """A directory reader class for loading and processing documents from file directories.

This class provides functionality to read documents from specified directories and convert them into document nodes. It supports both local and global file readers, and can handle different types of documents including images.

Args:
    input_files (Optional[List[str]]): A list of file paths to read. If None, files will be loaded when calling load_data method.
    local_readers (Optional[Dict]): A dictionary of local file readers specific to this instance. Keys are file patterns, values are reader functions.
    global_readers (Optional[Dict]): A dictionary of global file readers shared across all instances. Keys are file patterns, values are reader functions.


Examples:
    >>> from lazyllm.tools.rag.data_loaders import DirectoryReader
    >>> from lazyllm.tools.rag.readers import DocxReader, PDFReader
    >>> local_readers = {
    ...     "**/*.docx": DocxReader,
    ...     "**/*.pdf": PDFReader
    >>> }
    >>> reader = DirectoryReader(
    ...     input_files=["path/to/documents"],
    ...     local_readers=local_readers,
    ...     global_readers={}
    >>> )
    >>> documents = reader.load_data()
    >>> print(f"加载了 {len(documents)} 个文档")
    """
    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._local_readers = local_readers
        self._global_readers = global_readers

    def load_data(self, input_files: Optional[List[str]] = None, metadatas: Optional[Dict] = None,
                  *, split_image_nodes: bool = False) -> List[DocNode]:
        """Load and process documents from the specified input files.

This method reads documents from the input files using the configured file readers (both local and global), processes them into document nodes, and optionally separates image nodes from text nodes.

Args:
    input_files (Optional[List[str]]): A list of file paths to read. If None, uses the files specified during initialization.
    metadatas (Optional[Dict]): Additional metadata to associate with the loaded documents.
    split_image_nodes (bool): Whether to separate image nodes from text nodes. If True, returns a tuple of (text_nodes, image_nodes). If False, returns all nodes together.

**Returns:**

- Union[List[DocNode], Tuple[List[DocNode], List[ImageDocNode]]]: If split_image_nodes is False, returns a list of all document nodes. If True, returns a tuple containing text nodes and image nodes separately.
"""
        input_files = input_files or self._input_files
        file_readers = self._local_readers.copy()
        for key, func in self._global_readers.items():
            if key not in file_readers: file_readers[key] = func
        LOG.info(f'DirectoryReader loads data, input files: {input_files}')
        reader = SimpleDirectoryReader(input_files=input_files, file_extractor=file_readers, metadatas=metadatas)
        nodes: List[DocNode] = []
        image_nodes: List[ImageDocNode] = []
        for doc in reader():
            doc._group = LAZY_IMAGE_GROUP if isinstance(doc, ImageDocNode) else LAZY_ROOT_NAME
            if not split_image_nodes or not isinstance(doc, ImageDocNode):
                nodes.append(doc)
            else:
                image_nodes.append(doc)
        if not nodes and not image_nodes:
            LOG.warning(
                f'No nodes load from path {input_files}, please check your data path.'
            )
        LOG.info('DirectoryReader loads data done!')
        return (nodes, image_nodes) if split_image_nodes else nodes
