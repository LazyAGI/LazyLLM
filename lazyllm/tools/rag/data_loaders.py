from typing import List, Optional, Dict, Callable
from .store import DocNode, LAZY_ROOT_NAME
from lazyllm import LOG
from .dataReader import SimpleDirectoryReader

class DirectoryReader:
    def __init__(self, input_files: List[str], readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._readers = readers

    def load_data(self, input_files: Optional[List[str]] = None) -> List[DocNode]:
        input_files = input_files or self._input_files

        LOG.info(f"DirectoryReader loads data, input files: {input_files}")
        reader = SimpleDirectoryReader(input_files=input_files, file_extractor=self._readers)
        nodes = [doc for doc in reader() if setattr(doc, 'group', LAZY_ROOT_NAME) is None]
        if not nodes:
            LOG.warning(
                f"No nodes load from path {self.input_files}, please check your data path."
            )
        return nodes

    def register_file_reader(self, readers: Dict[str, Callable]):
        self._readers = readers
