from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode
from lazyllm import LOG

class MboxReader(LazyLLMReaderBase):
    DEFAULT_MESSAGE_FORMAT: str = (
        "Date: {_date}\n"
        "From: {_from}\n"
        "To: {_to}\n"
        "Subject: {_subject}\n"
        "Content: {_content}"
    )

    def __init__(self, max_count: int = 0, message_format: str = DEFAULT_MESSAGE_FORMAT,
                 return_trace: bool = True) -> None:
        try:
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError("`BeautifulSoup` package not found: `pip install beautifulsoup4`")

        super().__init__(return_trace=return_trace)
        self._max_count = max_count
        self._message_format = message_format

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        import mailbox
        from email.parser import BytesParser
        from email.policy import default
        from bs4 import BeautifulSoup

        if fs:
            LOG.warning("fs was specified but MboxReader doesn't support loading from "
                        "fsspec filesystems. Will load from local filesystem instead.")

        i = 0
        results: List[str] = []
        bytes_parser = BytesParser(policy=default).parse
        mbox = mailbox.mbox(file, factory=bytes_parser)

        for _, _msg in enumerate(mbox):
            try:
                msg: mailbox.mboxMessage = _msg
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get("Content-Disposition"))
                        if ctype == "text/plain" and "attachment" not in cdispo:
                            content = part.get_payload(decode=True)
                            break
                else:
                    content = msg.get_payload(decode=True)

                soup = BeautifulSoup(content)
                stripped_content = " ".join(soup.get_text().split())
                msg_string = self._message_format.format(_date=msg["date"], _from=msg["from"], _to=msg["to"],
                                                         _subject=msg["subject"], _content=stripped_content)
                results.append(msg_string)
            except Exception as e:
                LOG.warning(f"Failed to parse message:\n{_msg}\n with exception {e}")

            i += 1
            if self._max_count > 0 and i >= self._max_count: break
        return [DocNode(text=result, global_metadata=extra_info) for result in results]
