from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem
import importlib
from lazyllm.thirdparty import pandas as pd

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class PandasCSVReader(LazyLLMReaderBase):
    def __init__(self, concat_rows: bool = True, col_joiner: str = ", ", row_joiner: str = "\n",
                 pandas_config: Optional[Dict] = None, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config or {}

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        text_list = df.apply(lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1).tolist()

        if self._concat_rows: return [DocNode(text=(self._row_joiner).join(text_list), metadata=extra_info or {})]
        else: return [DocNode(text=text, metadata=extra_info or {}) for text in text_list]

class PandasExcelReader(LazyLLMReaderBase):
    def __init__(self, concat_rows: bool = True, sheet_name: Optional[str] = None,
                 pandas_config: Optional[Dict] = None, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        self._pandas_config = pandas_config or {}

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        openpyxl_spec = importlib.util.find_spec("openpyxl")
        if openpyxl_spec is not None: pass
        else: raise ImportError("Please install openpyxl to read Excel files. "
                                "You can install it with `pip install openpyxl`")

        if not isinstance(file, Path): file = Path(file)
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        documents = []
        if isinstance(dfs, pd.DataFrame):
            df = dfs.fillna("")
            text_list = (df.astype(str).apply(lambda row: " ".join(row.values), axis=1).tolist())

            if self._concat_rows: documents.append(DocNode(text="\n".join(text_list), metadata=extra_info or {}))
            else: documents.extend([DocNode(text=text, metadata=extra_info or {}) for text in text_list])
        else:
            for df in dfs.values():
                df = df.fillna("")
                text_list = (df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist())

                if self._concat_rows: documents.append(DocNode(text="\n".join(text_list), metadata=extra_info or {}))
                else: documents.extend([DocNode(text=text, global_metadata=extra_info) for text in text_list])

        return documents
