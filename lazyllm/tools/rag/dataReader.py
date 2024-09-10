import os
import mimetypes
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from functools import reduce
from itertools import repeat
from typing import Dict, Type, Optional, List, Callable
from pathlib import Path, PurePosixPath
from fsspec import AbstractFileSystem
from lazyllm import ModuleBase, LOG
from .store import DocNode
from .readers import (ReaderBase, PDFReader, DocxReader, HWPReader, PPTXReader, ImageReader, IPYNBReader,
                      EpubReader, MarkdownReader, MboxReader, PandasCSVReader, PandasExcelReader, VideoAudioReader,
                      get_default_fs, is_default_fs)

def _file_timestamp_format(timestamp: float, include_time: bool = False) -> Optional[str]:
    try:
        if include_time:
            return datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except Exception:
        return None

class _DefaultFileMetadataFunc:
    def __init__(self, fs: Optional[AbstractFileSystem] = None):
        self._fs = fs or get_default_fs()

    def __call__(self, file_path: str) -> Dict:
        stat_result = self._fs.stat(file_path)

        try:
            file_name = os.path.basename(str(stat_result['name']))
        except Exception:
            file_name = os.path.basename(file_path)

        creation_date = _file_timestamp_format(stat_result.get("created"))
        last_modified_date = _file_timestamp_format(stat_result.get("mtime"))
        last_accessed_date = _file_timestamp_format(stat_result.get("atime"))
        default_meta = {
            "file_path": file_path,
            "file_name": file_name,
            "file_type": mimetypes.guess_type(file_path)[0],
            "file_size": stat_result.get("size"),
            "creation_date": creation_date,
            "last_modified_date": last_modified_date,
            "last_accessed_date": last_accessed_date,
        }

        return {meta_key: meta_value for meta_key, meta_value in default_meta.items() if meta_value is not None}

class SimpleDirectoryReader(ModuleBase):
    _registered_file_reader_cls: Dict[str, Type[ReaderBase]] = {
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".hwp": HWPReader,
        ".pptx": PPTXReader,
        ".ppt": PPTXReader,
        ".pptm": PPTXReader,
        ".gif": ImageReader,
        ".jpeg": ImageReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".webp": ImageReader,
        ".ipynb": IPYNBReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".csv": PandasCSVReader,
        ".xls": PandasExcelReader,
        ".xlsx": PandasExcelReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
    }

    def __init__(self, input_dir: Optional[str] = None, input_files: Optional[List] = None,
                 exclude: Optional[List] = None, exclude_hidden: bool = True, recursive: bool = False,
                 encoding: str = "utf-8", filename_as_id: bool = False, required_exts: Optional[List[str]] = None,
                 fs: Optional[AbstractFileSystem] = None, file_metadata: Optional[Callable[[str], Dict]] = None,
                 num_files_limit: Optional[int] = None, return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)

        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")

        self._fs = fs or get_default_fs()
        self._encoding = encoding

        self._exclude = exclude
        self._recursive = recursive
        self._exclude_hidden = exclude_hidden
        self._required_exts = required_exts
        self._num_files_limit = num_files_limit
        _Path = Path if is_default_fs(self._fs) else PurePosixPath

        if input_files:
            self._input_files = []
            for path in input_files:
                if not self._fs.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = _Path(path)
                self._input_files.append(input_file)
        elif input_dir:
            if not self._fs.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            self._input_dir = _Path(input_dir)
            self._input_files = self._add_files(self._input_dir)

        self._file_metadata = file_metadata or _DefaultFileMetadataFunc(self._fs)
        self._filename_as_id = filename_as_id

    def _add_files(self, input_dir: Path) -> List[Path]:  # noqa: C901
        all_files = set()
        rejected_files = set()
        rejected_dirs = set()
        _Path = Path if is_default_fs(self._fs) else PurePosixPath

        if self._exclude is not None:
            for excluded_pattern in self._exclude:
                if self._recursive:
                    excluded_glob = _Path(input_dir) / _Path("**") / excluded_pattern
                else:
                    excluded_glob = _Path(input_dir) / excluded_pattern
                for file in self._fs.glob(str(excluded_glob)):
                    if self._fs.isdir(file):
                        rejected_dirs.add(_Path(file))
                    else:
                        rejected_files.add(_Path(file))

        file_refs: List[str] = []
        if self._recursive:
            file_refs = self._fs.glob(str(input_dir) + "/**/*")
        else:
            file_refs = self._fs.glob(str(input_dir) + "/*")

        for ref in file_refs:
            ref = _Path(ref)
            is_dir = self._fs.isdir(ref)
            skip_hidden = self._exclude_hidden and self._is_hidden(ref)
            skip_bad_exts = (self._required_exts is not None and ref.suffix not in self._required_exts)
            skip_excluded = ref in rejected_files
            if not skip_excluded:
                if is_dir:
                    ref_parent_dir = ref
                else:
                    ref_parent_dir = self._fs._parent(ref)
                for rejected_dir in rejected_dirs:
                    if str(ref_parent_dir).startswith(str(rejected_dir)):
                        skip_excluded = True
                        LOG.warning(f"Skipping {ref} because it in parent dir "
                                    f"{ref_parent_dir} which is in {rejected_dir}.")
                        break

            if is_dir or skip_hidden or skip_bad_exts or skip_excluded:
                continue
            else:
                all_files.add(ref)

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            raise ValueError(f"No files found in {input_dir}.")
        if self._num_files_limit is not None and self._num_files_limit > 0:
            new_input_files = new_input_files[0: self._num_files_limit]

        LOG.debug(f"[SimpleDirectoryReader] Total files add: {len(new_input_files)}")

        return new_input_files

    def _is_hidden(self, path: Path) -> bool:
        return any(part.startswith(".") and part not in [".", ".."] for part in path.parts)

    def _exclude_metadata(self, documents: List[DocNode]) -> List[DocNode]:
        for doc in documents:
            doc.excluded_embed_metadata_keys.extend(
                ["file_name", "file_type", "file_size", "creation_date",
                 "last_modified_date", "last_accessed_date"])
            doc.excluded_llm_metadata_keys.extend(
                ["file_name", "file_type", "file_size", "creation_date",
                 "last_modified_date", "last_accessed_date"])
        return documents

    @staticmethod
    def load_file(input_file: Path, file_metadata: Callable[[str], Dict], filename_as_id: bool = False,
                  encoding: str = "utf-8", fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        file_reader_cls = SimpleDirectoryReader._registered_file_reader_cls
        file_reader_suffix = list(file_reader_cls.keys())
        metadata: Optional[dict] = None
        documents: List[DocNode] = []

        if file_metadata is not None:
            metadata = file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()
        if file_suffix in file_reader_suffix:
            reader = file_reader_cls[file_suffix]()

            kwargs = {"extra_info": metadata}
            if fs and not is_default_fs(fs):
                kwargs['fs'] = fs
            docs = reader.load_data(input_file, **kwargs)

            if filename_as_id:
                for i, doc in enumerate(docs):
                    doc.uid = f"{input_file!s}_index_{i}"

            documents.extend(docs)
        else:
            fs = fs or get_default_fs()
            with fs.open(input_file, encoding=encoding) as f:
                data = f.read().decode(encoding)

            doc = DocNode(text=data, metadata=metadata or {})
            if filename_as_id:
                doc.uid = str(input_file)

            documents.append(doc)

        return documents

    def load_data(self, show_progress: bool = False, num_workers: Optional[int] = None,
                  fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        documents = []

        fs = fs or self._fs
        process_file = self._input_files

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                LOG.warning("Specified num_workers exceed number of CPUs in the system. "
                            "Setting `num_workers` down to the maximum CPU count.")
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(SimpleDirectoryReader.load_file,
                                    zip(process_file, repeat(self._file_metadata), repeat(self._filename_as_id),
                                        repeat(self._encoding), repeat(self._fs)))
                documents = reduce(lambda x, y: x + y, results)
        else:
            if show_progress:
                process_file = tqdm(self._input_files, desc="Loading files", unit="file")
            for input_file in process_file:
                documents.extend(
                    SimpleDirectoryReader.load_file(input_file=input_file, file_metadata=self._file_metadata,
                                                    filename_as_id=self._filename_as_id, encoding=self._encoding,
                                                    fs=self._fs))

        return self._exclude_metadata(documents)

    @classmethod
    def register_file_reader(cls, target_class: ReaderBase):
        clazz = target_class.__name__.casefold().split("reader")[0]
        cls._registered_file_reader_cls["." + clazz] = target_class
        return target_class

    @classmethod
    def get_registry(cls):
        return cls._registered_file_reader_cls
