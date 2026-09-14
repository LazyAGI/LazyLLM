from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderCapability,
    WriterProviderCapabilityError,
    WriterProviderDocument,
    WriterProviderRevisionError,
    WriterProviderWriteMode,
    WriterProviderWriteOutcomeError,
    is_ambiguous_write_error,
)
from .feishu import FeishuWriterProvider
from .github import GitHubWriterProvider
from .googledrive import GoogleDriveWriterProvider
from .notion import NotionWriterProvider
from .wechat import WeChatWriterProvider
from .obsidian import ObsidianWriterProvider
from .registry import (
    get_writer_provider,
    list_writer_providers,
    match_writer_provider,
    register_writer_provider,
    resolve_writer_create_target,
)


register_writer_provider(FeishuWriterProvider)
register_writer_provider(GitHubWriterProvider)
register_writer_provider(GoogleDriveWriterProvider)
register_writer_provider(NotionWriterProvider)
register_writer_provider(WeChatWriterProvider)
register_writer_provider(ObsidianWriterProvider)


__all__ = [
    'FeishuWriterProvider',
    'GitHubWriterProvider',
    'GoogleDriveWriterProvider',
    'NotionWriterProvider',
    'WeChatWriterProvider',
    'ObsidianWriterProvider',
    'WriterProviderBase',
    'WriterProviderCapabilities',
    'WriterProviderCapability',
    'WriterProviderCapabilityError',
    'WriterProviderDocument',
    'WriterProviderRevisionError',
    'WriterProviderWriteMode',
    'WriterProviderWriteOutcomeError',
    'is_ambiguous_write_error',
    'get_writer_provider',
    'list_writer_providers',
    'match_writer_provider',
    'register_writer_provider',
    'resolve_writer_create_target',
]
