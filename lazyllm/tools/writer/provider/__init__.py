from .base import WriterProviderBase
from .feishu import FeishuWriterProvider
from .notion import NotionWriterProvider
from .registry import (
    get_writer_provider,
    match_writer_provider,
    register_writer_provider,
)


register_writer_provider(FeishuWriterProvider)
register_writer_provider(NotionWriterProvider)


__all__ = [
    'FeishuWriterProvider',
    'NotionWriterProvider',
    'WriterProviderBase',
    'get_writer_provider',
    'match_writer_provider',
    'register_writer_provider',
]
