from typing import List, Optional
from .store import DocNode, LAZY_ROOT_NAME
from lazyllm import LOG
from .dataReader import SimpleDirectoryReader

class DirectoryReader:
    def __init__(self, input_files: List[str]) -> None:
        self._input_files = input_files

    def load_data(self, input_files: Optional[List[str]] = None) -> List[DocNode]:
        input_files = input_files or self._input_files

        LOG.info(f"DirectoryReader loads data, input files: {input_files}")
        reader = SimpleDirectoryReader(input_files=input_files)
        nodes = [doc for doc in reader.load_data() if setattr(doc, 'group', LAZY_ROOT_NAME) is None]
        if not nodes:
            LOG.warning(
                f"No nodes load from path {self.input_files}, please check your data path."
            )
        return nodes
