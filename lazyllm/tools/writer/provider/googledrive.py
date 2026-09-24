from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from ...fs.client import FS
from ..data_models.task import TargetDocument
from ..data_models.writer_ir import WriterStage
from .base import WriterProviderBase, WriterProviderCapabilities


_FILE_ID = re.compile(r'[A-Za-z0-9_-]+')
_DOCUMENT_PATH = re.compile(r'/(?:document|spreadsheets|presentation|file)/d/([A-Za-z0-9_-]+)(?:/.*)?')
_EXPORT_FORMATS = {
    'application/vnd.google-apps.document': 'text',
    'application/vnd.google-apps.spreadsheet': 'csv',
}


class GoogleDriveWriterProvider(WriterProviderBase):
    provider = 'googledrive'
    capabilities = WriterProviderCapabilities(load=True)

    @classmethod
    def matches(cls, locator: str) -> bool:
        value = str(locator or '').strip()
        if value.lower().startswith('googledrive:/'):
            return True
        try:
            parsed = urlsplit(value)
            return (parsed.scheme == 'https' and parsed.hostname in {'drive.google.com', 'docs.google.com'}
                    and parsed.username is None and parsed.password is None and parsed.port in {None, 443})
        except ValueError:
            return False

    def resolve(self, locator: str) -> TargetDocument:
        value = str(locator or '').strip()
        if not self.matches(value):
            raise ValueError('Invalid Google Drive document locator.')
        if value.lower().startswith('googledrive:/'):
            file_id = value.split(':', 1)[1].lstrip('/')
        else:
            parsed = urlsplit(value)
            match = _DOCUMENT_PATH.fullmatch(parsed.path)
            if match:
                file_id = match[1]
            elif parsed.path in {'/open', '/uc'}:
                ids = parse_qs(parsed.query).get('id', [])
                file_id = ids[0] if len(ids) == 1 else ''
            else:
                file_id = ''
        if not _FILE_ID.fullmatch(file_id):
            raise ValueError('Google Drive locator must identify one file.')
        return TargetDocument(
            adapter=self.provider, doc_id=file_id, uri=f'googledrive:/{file_id}',
            meta={'browser_url': f'https://drive.google.com/file/d/{file_id}/view'},
        )

    def load_document(self, target: TargetDocument, *, stage: WriterStage = 'final') -> dict:
        if target.adapter and target.adapter != self.provider:
            raise ValueError('Google Drive target adapter does not match provider.')
        resolved = self.resolve(target.uri or f'googledrive:/{target.doc_id or ""}')
        if target.doc_id and target.doc_id != resolved.doc_id:
            raise ValueError('Google Drive target ID does not match locator.')
        path = f'/{resolved.doc_id}'
        fs = FS._get_or_create_fs(self.provider, None, path)
        metadata = fs.info(path)
        if metadata.get('name') != resolved.doc_id:
            raise ValueError('Google Drive metadata does not match requested file.')
        mime = str(metadata.get('mime_type') or '').split(';', 1)[0].lower()
        if metadata.get('type') == 'directory':
            raise ValueError('Google Drive document loader cannot read a folder.')
        content_format = _EXPORT_FORMATS.get(mime)
        if mime.startswith('text/') or mime == 'application/json':
            content_format = {
                'text/markdown': 'markdown', 'text/x-markdown': 'markdown', 'text/csv': 'csv',
            }.get(mime, 'text')
        if not content_format:
            raise NotImplementedError(f'Google Drive document loader cannot read {mime or "unknown MIME type"}.')
        content = fs.read_file(path)
        if not isinstance(content, str):
            raise TypeError('Google Drive text read must return a string.')
        warnings = []
        if mime in _EXPORT_FORMATS:
            warnings.append('Google Workspace text export does not preserve all document formatting or media.')
        if mime == 'application/vnd.google-apps.spreadsheet':
            warnings.append('CSV export covers one worksheet; this result is not the complete workbook.')
        if content_format != 'markdown':
            # Keep plain text and CSV literal, including embedded Markdown fences.
            fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', content)), default=0))
            content = f'{fence}{content_format}\n{content}' + ('' if content.endswith('\n') else '\n') + f'{fence}\n'
        resolved.title = str(metadata.get('title') or target.title or '') or None
        resolved.meta = {
            **target.meta, **resolved.meta,
            'browser_url': metadata.get('web_url') or resolved.meta['browser_url'],
            'mime_type': mime, 'content_format': content_format,
            'modified_time': metadata.get('mtime'),
            'stage': stage,
        }
        return {
            'representation': 'markdown', 'source_document': content,
            'target_document': resolved, 'provider': self.provider,
            'resource_warnings': warnings,
        }
