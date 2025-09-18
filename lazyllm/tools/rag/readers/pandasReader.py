from pathlib import Path
from typing import Dict, List, Optional
from lazyllm.thirdparty import fsspec
import importlib
from lazyllm.thirdparty import pandas as pd

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class PandasCSVReader(LazyLLMReaderBase):
    """Reader for parsing CSV files using pandas.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single text block. Default is True.
    col_joiner (str): String used to join column values.
    row_joiner (str): String used to join rows.
    pandas_config (Optional[Dict]): Optional config for pandas.read_csv.
    return_trace (bool): Whether to return the processing trace.
"""
    def __init__(self, concat_rows: bool = True, col_joiner: str = ', ', row_joiner: str = '\n',
                 pandas_config: Optional[Dict] = None, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config or {}

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        text_list = df.apply(lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1).tolist()

        if self._concat_rows: return [DocNode(text=(self._row_joiner).join(text_list))]
        else: return [DocNode(text=text) for text in text_list]

class PandasExcelReader(LazyLLMReaderBase):
    """Reader for extracting text content from Excel (.xlsx) files.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single block.
    sheet_name (Optional[str]): Name of the sheet to read. If None, all sheets will be read.
    pandas_config (Optional[Dict]): Optional config for pandas.read_excel.
    return_trace (bool): Whether to return the processing trace.
"""
    def __init__(self, concat_rows: bool = True, sheet_name: Optional[str] = None,
                 pandas_config: Optional[Dict] = None, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        self._pandas_config = pandas_config or {}

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        openpyxl_spec = importlib.util.find_spec('openpyxl')
        if openpyxl_spec is not None: pass
        else: raise ImportError('Please install openpyxl to read Excel files. '
                                'You can install it with `pip install openpyxl`')

        if not isinstance(file, Path): file = Path(file)
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        documents = []
        if isinstance(dfs, pd.DataFrame):
            df = dfs.fillna('')
            text_list = (df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist())

            if self._concat_rows: documents.append(DocNode(text='\n'.join(text_list)))
            else: documents.extend([DocNode(text=text) for text in text_list])
        else:
            for df in dfs.values():
                df = df.fillna('')
                text_list = (df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist())

                if self._concat_rows: documents.append(DocNode(text='\n'.join(text_list)))
                else: documents.extend([DocNode(text=text) for text in text_list])

        return documents
