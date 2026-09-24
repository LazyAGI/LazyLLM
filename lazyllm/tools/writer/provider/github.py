from __future__ import annotations

import hashlib
import html
import mimetypes
import os
import posixpath
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

from lazyllm.common import ThreadPoolExecutor

from ...fs.supplier.github import GitHubFSError, GitHubRepoFS, GitHubWikiFS
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.task import InputResource, TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderDocument,
    WriterProviderWriteMode,
)

_GITHUB_REPO_URL_RE = re.compile(
    r'^https?://(?:www\.)?github\.com/[^/]+/[^/]+/blob/.+\.(?:md|markdown)(?:[?#].*)?$',
    re.IGNORECASE,
)
_GITHUB_WIKI_URL_RE = re.compile(
    r'^https?://(?:www\.)?github\.com/[^/]+/[^/]+/wiki/[^?#]+',
    re.IGNORECASE,
)
_MARKDOWN_H1_RE = re.compile(r'^#\s+(.+?)\s*$', re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(
    r'(?P<image>!)?\[[^\]]*\]\(\s*(?P<url><[^>]+>|[^\s)]+)(?:\s+["\'][^)]*["\'])?\s*\)',
)
_MARKDOWN_LINKED_IMAGE_RE = re.compile(
    r'\[(?P<image>!\[[^\]]*\]\(\s*(?:<[^>]+>|[^\s)]+)'
    r'(?:\s+["\'][^)]*["\'])?\s*\))\]'
    r'\(\s*(?:<[^>]+>|[^\s)]+)(?:\s+["\'][^)]*["\'])?\s*\)',
)
_HTML_IMAGE_RE = re.compile(
    r'<img\b[^>]*?\bsrc=["\'](?P<url>[^"\']+)["\']',
    re.IGNORECASE,
)
_HTML_LINKED_IMAGE_RE = re.compile(
    r'(?:<a\b(?P<link_attrs>[^>]*)>\s*)?'
    r'<img\b(?P<image_attrs>[^>]*)/?>'
    r'(?:\s*</a>)?',
    re.IGNORECASE | re.DOTALL,
)
_HTML_ATTRIBUTE_RE = re.compile(
    r'(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*'
    r'(?:(?P<quote>["\'])(?P<quoted>.*?)\2|(?P<unquoted>[^\s>]+))',
    re.DOTALL,
)
_HTML_SUB_RE = re.compile(r'<sub\b[^>]*>(?P<body>.*?)</sub>', re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_HTML_IMAGE_LAYOUT_PATTERNS = (
    re.compile(r'<table\b[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<div\b[^>]*>.*?<img\b.*?</div>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<p\b[^>]*>.*?<img\b.*?</p>', re.IGNORECASE | re.DOTALL),
)
_GITHUB_IMAGE_LAYOUT_META_KEY = 'github_writer_image_layouts'
_GITHUB_CODE_FENCE_META_KEY = 'github_writer_code_fences'
_GITHUB_MEDIA_ALIASES_META_KEY = 'github_writer_media_aliases'
_WRITER_MARKDOWN_CODE_LANGUAGES = frozenset({
    'bash',
    'css',
    'html',
    'javascript',
    'json',
    'markdown',
    'python',
    'sql',
    'text',
    'typescript',
    'yaml',
})
_CODE_FENCE_OPEN_RE = re.compile(
    r'^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})'
    r'(?P<info>[^\r\n]*)(?P<newline>\r?\n|$)',
)
_WRITER_USER_ANCHOR_LINE_RE = re.compile(
    r''' {0,3}<a[ \t]+id=(?P<quote>["'])block-user-[^\s"'<>]+(?P=quote)'''
    r'[ \t]*(?:>[ \t]*</a>|/>)[ \t]*(?:\r?\n|$)',
)
_MAX_ASSET_BYTES = 20 * 1024 * 1024
_MAX_WRITE_ASSET_BYTES = 50 * 1024 * 1024


def _target_type_from_locator(locator: str) -> str:
    value = str(locator or '').strip().lower()
    if value.startswith('githubwiki:') or _GITHUB_WIKI_URL_RE.match(value):
        return 'wiki'
    if value.startswith('githubrepo:') or _GITHUB_REPO_URL_RE.match(value):
        return 'repository'
    return ''


def _image_suffix(data: bytes) -> str:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if data.startswith(b'\xff\xd8\xff'):
        return '.jpg'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return '.gif'
    if data.startswith(b'BM'):
        return '.bmp'
    if data.startswith((b'II*\x00', b'MM\x00*')):
        return '.tiff'
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    prefix = data.lstrip(b'\xef\xbb\xbf\x00\x09\x0a\x0d\x20')[:4096].lower()
    if b'<svg' in prefix and b'<!doctype' not in prefix and b'<!entity' not in prefix:
        return '.svg'
    raise GitHubFSError(
        'GITHUB_ASSET_INVALID',
        'GitHub Writer assets must be supported image files.',
    )


def _html_attributes(value: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in _HTML_ATTRIBUTE_RE.finditer(value):
        attributes[match.group('name').lower()] = html.unescape(
            match.group('quoted') if match.group('quote') else match.group('unquoted'),
        )
    return attributes


def _markdown_image_alt(value: str) -> str:
    return str(value or '').replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')


def _html_image_layout_body(source: str) -> str:
    images: list[str] = []
    for match in _HTML_LINKED_IMAGE_RE.finditer(source):
        image_attributes = _html_attributes(match.group('image_attrs') or '')
        source_reference = str(image_attributes.get('src') or '').strip()
        if not source_reference:
            continue
        alt = str(image_attributes.get('alt') or '').strip()
        if not alt:
            alt = PurePosixPath(unquote(urlparse(source_reference).path)).stem or 'image'
        images.append(f'![{_markdown_image_alt(alt)}]({source_reference})')
    if not images:
        return ''
    captions = [
        html.unescape(_HTML_TAG_RE.sub(' ', match.group('body'))).strip()
        for match in _HTML_SUB_RE.finditer(source)
    ]
    captions = [' '.join(caption.split()) for caption in captions if caption]
    blocks: list[str] = []
    for index, image in enumerate(images):
        blocks.append(image)
        if index < len(captions):
            blocks.append(f'*{captions[index]}*')
    return '\n\n'.join(blocks)


def _normalize_html_image_layouts(markdown: str) -> tuple[str, list[dict[str, str]]]:
    normalized = markdown
    layouts: list[dict[str, str]] = []
    for pattern in _HTML_IMAGE_LAYOUT_PATTERNS:
        def replace(match: re.Match) -> str:
            source = match.group(0)
            body = _html_image_layout_body(source)
            if not body:
                return source
            layout_id = hashlib.sha256(
                f'{len(layouts)}:{source}'.encode(),
            ).hexdigest()[:16]
            layouts.append({
                'id': layout_id,
                'source': source,
                'display': body,
                'body': body,
            })
            return body

        normalized = pattern.sub(replace, normalized)

    def replace_linked_image(match: re.Match) -> str:
        source = match.group(0)
        body = match.group('image')
        layout_id = hashlib.sha256(
            f'{len(layouts)}:{source}'.encode(),
        ).hexdigest()[:16]
        layouts.append({
            'id': layout_id,
            'source': source,
            'display': body,
            'body': body,
        })
        return body

    normalized = _MARKDOWN_LINKED_IMAGE_RE.sub(replace_linked_image, normalized)
    return normalized, layouts


def _normalize_code_fences(markdown: str) -> tuple[str, list[dict[str, str]]]:
    lines = markdown.splitlines(keepends=True)
    normalized: list[str] = []
    layouts: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        opening = _CODE_FENCE_OPEN_RE.match(lines[index])
        if opening is None:
            normalized.append(lines[index])
            index += 1
            continue

        fence = opening.group('fence')
        closing_re = re.compile(
            rf'^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\r?\n|$)',
        )
        end = index + 1
        while end < len(lines) and closing_re.match(lines[end]) is None:
            end += 1
        if end < len(lines):
            end += 1

        info = opening.group('info')
        info_match = re.match(
            r'(?P<leading>[ \t]*)(?P<language>\S+)(?P<rest>.*)', info,
        )
        if info_match is None or info_match.group('language') in _WRITER_MARKDOWN_CODE_LANGUAGES:
            normalized.extend(lines[index:end])
            index = end
            continue

        display_info = (
            f"{info_match.group('leading')}text{info_match.group('rest')}"
        )
        display_opening = (
            f"{opening.group('indent')}{fence}{display_info}{opening.group('newline')}"
        )
        source = ''.join(lines[index:end])
        display = display_opening + ''.join(lines[index + 1:end])
        layout_id = hashlib.sha256(
            f'{len(layouts)}:{source}'.encode(),
        ).hexdigest()[:16]
        layouts.append({
            'id': layout_id,
            'language': info_match.group('language'),
            'source': source,
            'display': display,
        })
        normalized.append(display)
        index = end
    return ''.join(normalized), layouts


def _strip_writer_user_anchors(markdown: str) -> str:
    '''Remove standalone editor anchors from published Markdown, preserving code blocks.'''
    output: list[str] = []
    closing_re: re.Pattern | None = None
    for line in markdown.splitlines(keepends=True):
        if closing_re is not None:
            output.append(line)
            if closing_re.match(line):
                closing_re = None
            continue
        opening = _CODE_FENCE_OPEN_RE.match(line)
        if opening is not None:
            fence = opening.group('fence')
            closing_re = re.compile(
                rf'^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\r?\n|$)',
            )
        elif _WRITER_USER_ANCHOR_LINE_RE.fullmatch(line):
            continue
        output.append(line)
    return ''.join(output)


def _image_references(markdown: str) -> set[str]:
    references = {
        str(match.group('url') or '').strip('<>')
        for match in _MARKDOWN_LINK_RE.finditer(markdown)
        if match.group('image')
    }
    references.update(
        str(match.group('url') or '').strip('<>')
        for match in _HTML_IMAGE_RE.finditer(markdown)
    )
    return {reference for reference in references if reference}


def _rewrite_image_references(
    markdown: str,
    replacements: Mapping[str, str],
) -> str:
    if not replacements:
        return markdown

    def replace_url(match: re.Match, *, image_only: bool) -> str:
        if image_only and not match.groupdict().get('image'):
            return match.group(0)
        token = str(match.group('url') or '')
        value = token.strip('<>')
        replacement = str(replacements.get(value) or '')
        if not replacement:
            return match.group(0)
        if token.startswith('<') and token.endswith('>'):
            replacement = f'<{replacement}>'
        start, end = match.span('url')
        offset = match.start()
        return (
            match.group(0)[:start - offset]
            + replacement
            + match.group(0)[end - offset:]
        )

    rewritten = _MARKDOWN_LINK_RE.sub(
        lambda match: replace_url(match, image_only=True),
        markdown,
    )
    return _HTML_IMAGE_RE.sub(
        lambda match: replace_url(match, image_only=False),
        rewritten,
    )


def _matches_writer_preview(reference: str, digest: str) -> bool:
    if not digest or not re.fullmatch(r'[0-9a-f]{64}', digest):
        return False
    path = unquote(urlparse(reference).path).lower()
    return '/writer-preview-assets/' in path and digest in path


def _matches_materialized_asset(reference: str, digest: str) -> bool:
    if not digest or not re.fullmatch(r'[0-9a-f]{64}', digest):
        return False
    path = unquote(urlparse(reference).path).lower()
    match = re.search(
        r'(?:^|/)_?assets/[0-9a-f]{2}/(?P<digest>[0-9a-f]{64})\.[^/]+$',
        path,
    )
    return bool(match and match.group('digest') == digest)


def _unresolved_writer_preview_references(markdown: str) -> list[str]:
    return sorted({
        reference
        for reference in _image_references(markdown)
        if unquote(urlparse(reference).path).startswith('/static-files/')
    })


class GitHubWriterProvider(WriterProviderBase):
    '''Keep GitHub repository and Wiki Writer documents as native Markdown.'''

    provider = 'github'
    capabilities = WriterProviderCapabilities(
        load=True,
        create=True,
        replace=True,
        append=True,
        revision_check=True,
        media=True,
    )

    @classmethod
    def matches(cls, locator: str) -> bool:
        value = str(locator or '').strip()
        return bool(
            value.lower().startswith(('githubrepo:/', 'githubwiki:/'))
            or _GITHUB_REPO_URL_RE.match(value)
            or _GITHUB_WIKI_URL_RE.match(value)
        )

    @staticmethod
    def _fs(target_type: str, *, skip_instance_cache: bool = False):
        if target_type == 'repository':
            return GitHubRepoFS(dynamic_auth=True, skip_instance_cache=skip_instance_cache)
        if target_type == 'wiki':
            return GitHubWikiFS(dynamic_auth=True)
        raise ValueError(f'Unsupported GitHub target type: {target_type!r}.')

    def resolve(self, locator: str) -> TargetDocument:
        value = str(locator or '').strip()
        target_type = _target_type_from_locator(value)
        if not target_type:
            raise ValueError(f'Invalid GitHub Writer document locator: {locator!r}.')
        resolved = self._fs(target_type).resolve_target(value)
        return self._target_from_resolved(resolved)

    @staticmethod
    def _target_from_resolved(resolved: Mapping[str, object]) -> TargetDocument:
        meta = {
            key: value
            for key, value in resolved.items()
            if key not in {'doc_id', 'uri', 'title'} and value is not None
        }
        return TargetDocument(
            doc_id=str(resolved.get('doc_id') or '') or None,
            uri=str(resolved.get('uri') or '') or None,
            adapter='github',
            title=str(resolved.get('title') or '') or None,
            meta=meta,
        )

    @staticmethod
    def _locator(target: TargetDocument) -> str:
        locator = str(target.uri or target.meta.get('browser_url') or '').strip()
        if not locator:
            raise ValueError('GitHub target_document must provide uri or browser_url.')
        return locator

    @staticmethod
    def _target_type(target: TargetDocument) -> str:
        target_type = str(target.meta.get('target_type') or '').strip().lower()
        return target_type or _target_type_from_locator(str(target.uri or ''))

    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
        resource_cache_dir: str = '',
    ) -> dict:
        del stage  # Markdown representation does not carry Writer IR stages.
        target_type = self._target_type(target)
        fs = self._fs(target_type)
        locator = self._locator(target)
        if target_type == 'wiki':
            with fs.read_session(locator) as (resolved, read_bytes):
                return self._load_resolved_document(target, resolved, read_bytes)
        resolved = fs.resolve_target(locator)

        def read_resource(uri: str) -> bytes:
            resource_fs = self._fs(target_type, skip_instance_cache=True)
            try:
                return resource_fs.read_bytes(uri)
            finally:
                resource_fs.close()

        return self._load_resolved_document(
            target, resolved, fs.read_bytes,
            read_resource_bytes=read_resource if resource_cache_dir else None,
            resource_cache_dir=resource_cache_dir,
        )

    def _convert_native_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
    ) -> WriterProviderDocument:
        if not isinstance(content, str):
            raise TypeError('GitHub Writer Provider accepts final Markdown, not WriterDocument IR.')
        source = self._writer_document(content, media_assets)
        return WriterProviderDocument(
            provider=self.provider,
            format='markdown',
            content=content,
            source_document=source,
        )

    def prepare_markdown_for_editor(
        self,
        markdown: str,
        target: TargetDocument,
    ) -> str:
        return self.normalize_code_fences_for_writer(markdown, target)

    def _load_resolved_document(
        self,
        target: TargetDocument,
        resolved: Mapping[str, object],
        read_bytes: Callable[[str], bytes],
        *,
        read_resource_bytes: Callable[[str], bytes] | None = None,
        resource_cache_dir: str = '',
    ) -> dict:
        resolved_target = self._merge_target(target, resolved)
        try:
            markdown = read_bytes(str(resolved['uri'])).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise GitHubFSError(
                'GITHUB_FORMAT_UNSUPPORTED',
                'GitHub Writer documents must be UTF-8 Markdown.',
            ) from exc
        resources, warnings = self._collect_referenced_resources(
            markdown, resolved_target, read_resource_bytes or read_bytes,
            resource_cache_dir=resource_cache_dir,
        )
        writer_markdown, image_layouts = _normalize_html_image_layouts(markdown)
        if image_layouts:
            resolved_target.meta[_GITHUB_IMAGE_LAYOUT_META_KEY] = image_layouts
        else:
            resolved_target.meta.pop(_GITHUB_IMAGE_LAYOUT_META_KEY, None)
        writer_markdown = self.normalize_code_fences_for_writer(
            writer_markdown, resolved_target, replace_existing=True,
        )
        return {
            'representation': 'markdown',
            'source_document': writer_markdown,
            'target_document': resolved_target,
            'provider': self.provider,
            'block_count': 1,
            'input_resources': resources,
            'resource_warnings': warnings,
        }

    @staticmethod
    def normalize_code_fences_for_writer(
        markdown: str,
        target: TargetDocument,
        *,
        replace_existing: bool = False,
    ) -> str:
        normalized, layouts = _normalize_code_fences(markdown)
        existing = [] if replace_existing else target.meta.get(_GITHUB_CODE_FENCE_META_KEY)
        merged = [item for item in existing or [] if isinstance(item, Mapping)]
        existing_counts: dict[tuple[str, str], int] = {}
        for item in merged:
            identity = (str(item.get('source') or ''), str(item.get('display') or ''))
            existing_counts[identity] = existing_counts.get(identity, 0) + 1
        incoming_counts: dict[tuple[str, str], int] = {}
        for layout in layouts:
            identity = (layout['source'], layout['display'])
            incoming_counts[identity] = incoming_counts.get(identity, 0) + 1
            if incoming_counts[identity] > existing_counts.get(identity, 0):
                merged.append(layout)
        if merged:
            target.meta[_GITHUB_CODE_FENCE_META_KEY] = merged
        elif replace_existing:
            target.meta.pop(_GITHUB_CODE_FENCE_META_KEY, None)
        return normalized

    def _collect_referenced_resources(
        self,
        markdown: str,
        target: TargetDocument,
        read_bytes: Callable[[str], bytes],
        *,
        resource_cache_dir: str = '',
    ) -> tuple[list[InputResource], list[str]]:
        candidates: list[str] = []
        candidates.extend(
            match.group('url').strip('<>')
            for match in _MARKDOWN_LINK_RE.finditer(markdown)
            if match.group('image')
        )
        candidates.extend(
            match.group('url').strip()
            for match in _HTML_IMAGE_RE.finditer(markdown)
        )
        references: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_url in candidates:
            uri = self._referenced_uri(raw_url, target)
            if not uri or uri in seen:
                continue
            seen.add(uri)
            references.append((raw_url, uri))
        cache_root = Path(resource_cache_dir).expanduser().resolve() if resource_cache_dir else None

        def collect_resource(raw_url: str, uri: str) -> tuple[InputResource | None, list[str]]:
            mime_type = mimetypes.guess_type(unquote(urlparse(uri).path))[0]
            try:
                payload = read_bytes(uri)
            except Exception as exc:  # noqa: BLE001 - one resource failure becomes a warning.
                code = getattr(exc, 'code', type(exc).__name__)
                return None, [f'{raw_url}: {code}']
            resource = InputResource(
                resource_type='image',
                uri=uri,
                mime_type=mime_type,
                title=PurePosixPath(unquote(urlparse(uri).path)).name or None,
                summary=None,
                meta={
                    'provider': self.provider,
                    'role': 'background',
                    'referenced_from': target.uri,
                    'source_reference': raw_url,
                    'size': len(payload),
                },
            )
            if cache_root is not None:
                cache_path = cache_root / hashlib.sha256(uri.encode()).hexdigest()
                try:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(payload)
                except OSError as exc:
                    return resource, [f'{raw_url}: CACHE_WRITE_FAILED ({type(exc).__name__})']
                resource.meta['github_resource_cache'] = {
                    'uri': uri, 'path': str(cache_path),
                    'sha256': hashlib.sha256(payload).hexdigest(),
                }
            return resource, []

        if cache_root is not None and self._target_type(target) == 'repository' and len(references) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(references))) as executor:
                futures = [
                    executor.submit(collect_resource, raw_url, uri)
                    for raw_url, uri in references
                ]
                results = [future.result() for future in futures]
        else:
            results = [collect_resource(raw_url, uri) for raw_url, uri in references]
        resources: list[InputResource] = []
        warnings: list[str] = []
        for resource, resource_warnings in results:
            warnings.extend(resource_warnings)
            if resource is not None:
                resource.resource_id = f'github-resource-{len(resources)}'
                resources.append(resource)
        return resources, warnings

    def _referenced_uri(self, raw_url: str, target: TargetDocument) -> str:
        value = unquote(str(raw_url or '').strip())
        if not value or value.startswith(('#', 'data:', 'mailto:', 'javascript:')):
            return ''
        target_type = self._target_type(target)
        owner = str(target.meta.get('owner') or '')
        repo = str(target.meta.get('repo') or '')
        document_path = str(target.meta.get('path') or '')
        if not owner or not repo or not document_path:
            return ''
        parsed = urlparse(value)
        if parsed.scheme in ('http', 'https'):
            resource = self._absolute_resource_path(parsed, target, target_type, owner, repo)
            if resource is None:
                return ''
            owner, repo, resource_path = resource
        else:
            relative_path = unquote(parsed.path)
            if relative_path.startswith('/'):
                resource_path = relative_path.lstrip('/')
            else:
                resource_path = posixpath.normpath(
                    posixpath.join(posixpath.dirname(document_path), relative_path),
                )
            if resource_path == '..' or resource_path.startswith('../'):
                return ''
        if target_type == 'wiki':
            return f'githubwiki:/{owner}/{repo}/{quote(resource_path, safe="/")}'
        ref = str(target.meta.get('ref') or '')
        return (
            f'githubrepo:/{owner}/{repo}/{quote(resource_path, safe="/")}'
            f'?ref={quote(ref, safe="")}'
        )

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        title = str(title or '').strip()
        if not title:
            raise ValueError('title is required')
        parent_uri = str(parent_uri or '').strip()
        target_type = _target_type_from_locator(parent_uri)
        if not target_type:
            raise ValueError('GitHub create_document requires a repository or Wiki parent URI.')
        created = self._fs(target_type).create_document(
            title,
            parent_uri,
            operation_id=hashlib.sha256(
                f'create:{parent_uri}:{title}'.encode(),
            ).hexdigest()[:20],
        )
        return self._target_from_resolved(created)

    def _resolve_create_target(self, parent_uri: str) -> TargetDocument:
        parent_uri = str(parent_uri or '').strip()
        if GitHubRepoFS.matches_create_parent(parent_uri):
            fs = GitHubRepoFS(dynamic_auth=True)
        elif GitHubWikiFS.matches_create_parent(parent_uri):
            fs = GitHubWikiFS(dynamic_auth=True)
        else:
            raise ValueError(
                'GitHub document creation requires a repository root, tree directory, or Wiki root URL.',
            )
        planned = fs.resolve_create_parent(parent_uri)
        return self._target_from_resolved(planned)

    def write_document(
        self,
        converted: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        if converted.provider != self.provider or converted.format != 'markdown':
            raise ValueError('GitHub write_document requires converted GitHub Markdown.')
        if not isinstance(converted.content, str):
            raise TypeError('Converted GitHub content must be a Markdown string.')
        content = converted.content
        if mode == 'append':
            loaded = self.load_document(target)
            current = str(loaded['source_document'])
            target = TargetDocument.model_validate(loaded['target_document'])
            separator = '' if not current or current.endswith('\n') else '\n'
            content = f'{current}{separator}{content}'
        target_type = self._target_type(target)
        fs = self._fs(target_type)
        if target.meta.get('create_pending'):
            heading = _MARKDOWN_H1_RE.search(content)
            title = (
                str(target.title or '').strip()
                or (heading.group(1).strip() if heading else '')
                or 'untitled'
            )
            resolved = fs.resolve_create_target(target.meta, title)
            planned_target = self._merge_target(target, resolved)
            target.doc_id = planned_target.doc_id
            target.uri = planned_target.uri
            target.title = planned_target.title
            target.adapter = planned_target.adapter
            target.meta = planned_target.meta
            target.meta.setdefault(
                'commit_message', f'Create {target.meta.get("path")} via LazyMind',
            )
            target.meta.setdefault(
                'pull_request_title', f'Create {target.meta.get("path")}',
            )
        if not target.meta.get('revision'):
            refreshed = fs.resolve_target(self._locator(target))
            target = self._merge_target(target, refreshed)
        markdown = _strip_writer_user_anchors(content)
        if media_assets is not None:
            markdown = self._restore_imported_media_references(markdown, media_assets)
        markdown = self._restore_html_image_layouts(markdown, target)
        markdown = self._restore_code_fences(markdown, target)
        markdown, files = self._materialize_media(markdown, target, media_assets)
        if _unresolved_writer_preview_references(markdown):
            raise GitHubFSError(
                'GITHUB_ASSET_INVALID',
                'GitHub Writer output contains unresolved local image previews.',
            )
        publish_mode = str(target.meta.get('publish_mode') or target.meta.get('write_mode') or '').strip()
        if not publish_mode:
            publish_mode = 'direct' if target_type == 'wiki' else 'pull_request'
        requested_operation_id = str(target.meta.get('operation_id') or '').strip()
        last_operation_id = str(target.meta.get('last_operation_id') or '').strip()
        result = fs.apply_document_patch(
            self._locator(target),
            markdown,
            files=files,
            expected_revision=str(target.meta.get('revision') or ''),
            operation_id=(
                requested_operation_id
                if requested_operation_id != last_operation_id else ''
            ),
            publish_mode=publish_mode,
            commit_message=str(target.meta.get('commit_message') or ''),
            pull_request_title=str(target.meta.get('pull_request_title') or ''),
            work_branch=str(target.meta.get('work_branch') or ''),
            base_ref=str(target.meta.get('base_ref') or ''),
        )
        self._apply_write_result(target, result)
        return {
            **result,
            'doc_id': str(result.get('doc_id') or target.doc_id or ''),
            'adapter': self.provider,
            'locator': str(result.get('uri') or target.uri or ''),
            'block_count': 1,
            'warnings': list(result.get('warnings') or []),
            'persisted_document': content,
            'representation': 'markdown',
        }

    @staticmethod
    def _merge_target(target: TargetDocument, resolved: Mapping[str, object]) -> TargetDocument:
        merged = target.model_copy(deep=True)
        refreshed = GitHubWriterProvider._target_from_resolved(resolved)
        merged.doc_id = refreshed.doc_id
        merged.uri = refreshed.uri
        merged.title = refreshed.title or merged.title
        merged.adapter = 'github'
        merged.meta = {**merged.meta, **refreshed.meta}
        return merged

    @staticmethod
    def _apply_write_result(target: TargetDocument, result: Mapping[str, object]) -> None:
        target.doc_id = str(result.get('doc_id') or target.doc_id or '') or None
        target.uri = str(result.get('uri') or target.uri or '') or None
        target.adapter = 'github'
        target.meta = {
            **target.meta,
            **{
                key: value
                for key, value in result.items()
                if key not in {'success', 'provider', 'doc_id', 'uri', 'warnings'}
            },
        }
        operation_id = str(result.get('operation_id') or '').strip()
        if operation_id:
            target.meta['last_operation_id'] = operation_id

    @staticmethod
    def _imported_media_source_reference(asset: object) -> str:
        meta = asset.meta if isinstance(getattr(asset, 'meta', None), Mapping) else {}
        source_reference = str(meta.get('source_reference') or '').strip()
        if not source_reference and meta.get('origin') == 'markdown':
            original_uri = str(getattr(asset, 'uri', '') or '').strip()
            if urlparse(original_uri).scheme in {'http', 'https'}:
                source_reference = original_uri
        return source_reference

    def _materialize_media(
        self,
        markdown: str,
        target: TargetDocument,
        media_assets: MediaAssetLibrary | None,
    ) -> tuple[str, dict[str, bytes]]:
        if media_assets is None:
            return markdown, {}
        document_path = str(target.meta.get('path') or '')
        document_dir = posixpath.dirname(document_path)
        target_type = self._target_type(target)
        asset_dir = '_assets' if target_type == 'wiki' else 'assets'
        files: dict[str, bytes] = {}
        media_aliases: dict[str, str] = {}
        rewritten = self._restore_imported_media_references(markdown, media_assets)
        total_size = 0
        for asset_id, asset in media_assets.assets.items():
            references = _image_references(rewritten)
            source_reference = self._imported_media_source_reference(asset)
            if source_reference and source_reference in references:
                continue
            local_path, matched, data = self._match_media_asset(asset_id, asset, references)
            if not matched:
                continue
            if not local_path or not os.path.isfile(local_path):
                raise GitHubFSError(
                    'GITHUB_ASSET_INVALID',
                    f'Media asset {asset_id!r} has no readable local_path.',
                )
            if data is None:
                data = Path(local_path).read_bytes()
            if not data or len(data) > _MAX_ASSET_BYTES:
                raise GitHubFSError(
                    'GITHUB_ASSET_TOO_LARGE',
                    f'Media asset {asset_id!r} must be between 1 byte and 20 MB.',
                )
            digest = hashlib.sha256(data).hexdigest()
            suffix = _image_suffix(data)
            filename = f'{digest[:2]}/{digest}{suffix}'
            relative_target = posixpath.join(document_dir, asset_dir, filename)
            if relative_target not in files:
                total_size += len(data)
                if total_size > _MAX_WRITE_ASSET_BYTES:
                    raise GitHubFSError(
                        'GITHUB_ASSET_TOO_LARGE',
                        'GitHub Writer assets exceed the 50 MB per-write limit.',
                    )
                files[relative_target] = data
            link = posixpath.relpath(relative_target, document_dir or '.')
            repository_reference = quote(link, safe='/._-')
            media_aliases[repository_reference] = asset_id
            rewritten = _rewrite_image_references(
                rewritten,
                {reference: repository_reference for reference in matched},
            )
        if media_aliases:
            target.meta[_GITHUB_MEDIA_ALIASES_META_KEY] = media_aliases
        else:
            target.meta.pop(_GITHUB_MEDIA_ALIASES_META_KEY, None)
        return rewritten, files

    @staticmethod
    def _restore_html_image_layouts(
        markdown: str,
        target: TargetDocument,
    ) -> str:
        layouts = target.meta.get(_GITHUB_IMAGE_LAYOUT_META_KEY)
        if not isinstance(layouts, list):
            return markdown
        restored = markdown
        for layout in layouts:
            if not isinstance(layout, Mapping):
                continue
            layout_id = str(layout.get('id') or '').strip()
            source = str(layout.get('source') or '')
            display = str(layout.get('display') or '')
            body = str(layout.get('body') or '')
            if not source:
                continue
            if layout_id:
                marker = re.compile(
                    re.escape(f'<!-- lazyllm-github-image-layout:{layout_id}:start -->')
                    + r'.*?'
                    + re.escape(f'<!-- lazyllm-github-image-layout:{layout_id}:end -->'),
                    re.DOTALL,
                )
                restored, count = marker.subn(
                    lambda _, source=source: source,
                    restored,
                    count=1,
                )
                if count:
                    continue
            for candidate in (display, body):
                if candidate and candidate in restored:
                    restored = restored.replace(candidate, source, 1)
                    break
        return restored

    @staticmethod
    def _restore_code_fences(
        markdown: str,
        target: TargetDocument,
    ) -> str:
        layouts = target.meta.get(_GITHUB_CODE_FENCE_META_KEY)
        if not isinstance(layouts, list):
            return markdown
        restored = markdown
        for layout in layouts:
            if not isinstance(layout, Mapping):
                continue
            source = str(layout.get('source') or '')
            display = str(layout.get('display') or '')
            if source and display and display in restored:
                restored = restored.replace(display, source, 1)
        return restored

    @staticmethod
    def _restore_imported_media_references(
        markdown: str,
        media_assets: MediaAssetLibrary,
    ) -> str:
        replacements, digest_sources = GitHubWriterProvider._imported_media_replacements(media_assets)
        if not replacements:
            return markdown

        def replace_url(match: re.Match, *, image_only: bool) -> str:
            if image_only and not match.groupdict().get('image'):
                return match.group(0)
            token = str(match.group('url') or '')
            value = token.strip('<>')
            replacement = replacements.get(value)
            if not replacement:
                matched_sources = {
                    source_reference
                    for digest, sources in digest_sources.items()
                    if _matches_writer_preview(value, digest)
                    or _matches_materialized_asset(value, digest)
                    for source_reference in sources
                }
                if len(matched_sources) == 1:
                    replacement = matched_sources.pop()
            if not replacement:
                return match.group(0)
            if token.startswith('<') and token.endswith('>'):
                replacement = f'<{replacement}>'
            start, end = match.span('url')
            offset = match.start()
            return (
                match.group(0)[:start - offset]
                + replacement
                + match.group(0)[end - offset:]
            )

        restored = _MARKDOWN_LINK_RE.sub(
            lambda match: replace_url(match, image_only=True),
            markdown,
        )
        return _HTML_IMAGE_RE.sub(
            lambda match: replace_url(match, image_only=False),
            restored,
        )

    @staticmethod
    def _match_media_asset(asset_id, asset, references):
        local_path = str(asset.local_path or '').strip()
        meta = asset.meta if isinstance(asset.meta, Mapping) else {}
        candidates = {
            f'asset://{asset_id}',
            local_path,
            str(asset.uri or '').strip(),
            str(meta.get('preview_reference') or '').strip(),
        }
        candidates.discard('')
        matched = references.intersection(candidates)
        digest = str(meta.get('sha256') or '').strip().lower()
        data: bytes | None = None
        if not matched and local_path and os.path.isfile(local_path):
            data = Path(local_path).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
        if not matched and digest:
            matched = {
                reference
                for reference in references
                if _matches_writer_preview(reference, digest)
                or _matches_materialized_asset(reference, digest)
            }
        return local_path, matched, data

    @staticmethod
    def _imported_media_replacements(media_assets):
        replacements: dict[str, str] = {}
        digest_sources: dict[str, set[str]] = {}
        for asset in media_assets.assets.values():
            meta = asset.meta if isinstance(asset.meta, Mapping) else {}
            source_reference = GitHubWriterProvider._imported_media_source_reference(asset)
            if not source_reference:
                continue
            digest = str(meta.get('sha256') or '').strip().lower()
            if re.fullmatch(r'[0-9a-f]{64}', digest):
                digest_sources.setdefault(digest, set()).add(source_reference)
            for candidate in (
                str(asset.local_path or '').strip(),
                str(meta.get('preview_reference') or '').strip(),
            ):
                if candidate:
                    replacements[candidate] = source_reference
        return replacements, digest_sources

    @staticmethod
    def _absolute_resource_path(parsed, target, target_type, owner, repo):
        if target_type != 'repository':
            return None
        hostname = (parsed.hostname or '').lower()
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if hostname in {'github.com', 'www.github.com'}:
            if len(parts) < 5 or parts[0] != owner or parts[1] != repo or parts[2] != 'blob':
                return None
            # Referenced browser URLs are normally generated with the same ref
            # as the source document, so remove that exact prefix without
            # guessing another branch boundary.
            ref = str(target.meta.get('ref') or '')
            tail = '/'.join(parts[3:])
            if not ref or not tail.startswith(ref + '/'):
                return None
            resource_path = tail[len(ref) + 1:]
        elif hostname == 'raw.githubusercontent.com':
            if len(parts) < 4:
                return None
            owner, repo, ref = parts[:3]
            resource_path = '/'.join(parts[3:])
        else:
            return None
        return owner, repo, resource_path


__all__ = ['GitHubWriterProvider']
