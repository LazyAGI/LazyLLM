from typing import Union
from ..dataReader import SimpleDirectoryReader

class FileReader(object):

    def __call__(self, input_files: Union[str, list[str]]):
        if len(input_files) == 0:
            return []
        if isinstance(input_files, str):
            input_files = [input_files]
        nodes = SimpleDirectoryReader(input_files=input_files)._load_data()
        txt = [node.get_text() for node in nodes]
        return "\n".join(txt)
