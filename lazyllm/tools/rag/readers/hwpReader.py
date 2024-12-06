from fsspec import AbstractFileSystem
from pathlib import Path
import struct
from typing import Optional, Dict, List, Any
import zlib

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode
from lazyllm import LOG

class HWPReader(LazyLLMReaderBase):
    def __init__(self, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._FILE_HEADER_SECTION = "FileHeader"
        self._HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self._SECTION_NAME_LENGTH = len("Section")
        self._BODYTEXT_SECTION = "BodyText"
        self._HWP_TEXT_TAGS = [67]
        self._text = ""

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        try:
            import olefile
        except ImportError:
            raise ImportError("olefile is required to read hwp files: `pip install olefile`")

        if fs:
            LOG.warning("fs was specified but HWPReader doesn't support loading from "
                        "fsspec filesystems. Will load from local filesystem instead.")

        if not isinstance(file, Path): file = Path(file)

        load_file = olefile.OleFileIO(file)
        file_dir = load_file.listdir()
        if self._is_valid(file_dir) is False: raise Exception("Not Valid HwpFile")

        result_text = self._get_text(load_file, file_dir)
        return [DocNode(text=result_text, global_metadata=extra_info)]

    def _is_valid(self, dirs: List[str]) -> bool:
        if [self._FILE_HEADER_SECTION] not in dirs: return False
        return [self._HWP_SUMMARY_SECTION] in dirs

    def _text_to_docnode(self, text: str, extra_info: Optional[Dict] = None) -> DocNode:
        return DocNode(text=text, metadata=extra_info or {})

    def _get_text(self, load_file: Any, file_dirs: List[str]) -> str:
        sections = self._get_body_sections(file_dirs)
        text = ""
        for section in sections:
            text += self._get_text_from_section(load_file, section)
            text += "\n"

        self._text = text
        return self._text

    def _get_body_sections(self, dirs: List[str]) -> List[str]:
        m = []
        for d in dirs:
            if d[0] == self._BODYTEXT_SECTION:
                m.append(int(d[1][self._SECTION_NAME_LENGTH:]))

        return ["BodyText/Section" + str(x) for x in sorted(m)]

    def _is_compressed(self, load_file: Any) -> bool:
        header = load_file.openstream("FileHeader")
        header_data = header.read()
        return (header_data[36] & 1) == 1

    def _get_text_from_section(self, load_file: Any, section: str) -> str:
        bodytext = load_file.openstream(section)
        data = bodytext.read()

        unpacked_data = (zlib.decompress(data, -15) if self._is_compressed(load_file) else data)
        size = len(unpacked_data)

        i = 0
        text = ""
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3FF
            (header >> 10) & 0x3FF
            rec_len = (header >> 20) & 0xFFF

            if rec_type in self._HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4: i + 4 + rec_len]
                text += rec_data.decode("utf-16")
                text += "\n"

            i += 4 + rec_len
        return text
