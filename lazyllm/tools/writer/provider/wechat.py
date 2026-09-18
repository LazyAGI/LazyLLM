from __future__ import annotations

import json
import mimetypes
from collections.abc import Callable, Iterable
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

import requests

import lazyllm

from ..adapter.wechat import WeChatWriterAdapter
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchSet
from ..data_models.task import InputResource, TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..tools.revision_tools import apply_patch_to_ir
from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderCapabilityError,
    WriterProviderDocument,
    WriterProviderRevisionError,
    WriterProviderWriteMode,
)

_MP_HOME = 'https://mp.weixin.qq.com/'
_DRAFT_TITLE_NOT_FOUND = (
    '未在当前默认公众号的草稿箱中找到匹配文章，'
    '请输入草稿文章的准确完整标题。'
)
_ARTICLE_FIELDS = {
    'article_type', 'title', 'author', 'digest', 'content',
    'content_source_url', 'thumb_media_id', 'show_cover_pic',
    'need_open_comment', 'only_fans_can_comment',
}


def wechat_placeholder_cover_png() -> bytes:
    return files(__package__).joinpath(
        'assets/wechat-placeholder-cover.png',
    ).read_bytes()


_WECHAT_COVER_SIZE = (900, 383)
_WECHAT_DRAFT_REVISION_TERMS = ('微信', '公众号', '草稿箱')
_WECHAT_REVISION_INTENT_TERMS = (
    '修改', '改写', '重写', '润色', '编辑', '修订', '优化',
    '删除', '删掉', '移除', '替换', '更换', '调整',
    '扩写', '续写', '新增', '添加', '插入', '合并', '重排', '增强',
)


def _document_text(document: WriterDocument) -> str:
    return '\n'.join(
        block.content
        for block in document.iter_blocks()
        if block.content
    )


def prepare_wechat_cover(
    target: TargetDocument,
    document: WriterDocument | str,
    root: Path,
    *,
    model_available: Callable[[str], bool] | None = None,
    generator: Callable[..., dict[str, Any]] | None = None,
) -> TargetDocument:
    '''Prepare a WeChat cover for a new article target.

    Image generation is injected by the host application so this provider remains
    independent of application-layer model tooling.
    '''
    if (
        target.adapter != 'wechat'
        or target.doc_id
        or target.meta.get('thumb_media_id')
    ):
        return target
    if isinstance(document, WriterDocument):
        binding = document.provider_binding
        if binding.get('provider') == 'wechat' and binding.get('document_id'):
            return target
        title = document.title or target.title or '未命名文档'
        body = _document_text(document)
    else:
        title = target.title or '未命名文档'
        body = str(document)

    from PIL import Image, ImageOps

    cover_path = root / 'wechat-cover.png'
    try:
        if model_available is not None and not model_available('image_generator'):
            raise RuntimeError('image_generator is not configured')
        if generator is None:
            raise RuntimeError('WeChat cover generation service is not configured')
        result = generator(
            '为微信公众号文章生成一张专业、简洁、无文字、无水印的横版封面图。\n'
            f'文章标题：{title}\n文章内容摘要：{body[:1000]}',
            image_size='1024x1024',
            batch_size=1,
        )
        generated_path = Path(str(result.get('local_path') or ''))
        if not generated_path.is_file():
            raise ValueError('image_generator returned no usable local image')
        with Image.open(generated_path) as source:
            cover = ImageOps.fit(
                source.convert('RGB'),
                _WECHAT_COVER_SIZE,
                method=Image.Resampling.LANCZOS,
            )
            cover.save(cover_path, format='PNG')
    except Exception as exc:  # noqa: BLE001 - a valid cover is still required.
        lazyllm.LOG.warning(
            'WeChat cover generation failed; using placeholder cover: %s', exc,
        )
        cover_path.write_bytes(wechat_placeholder_cover_png())

    prepared = target.model_copy(deep=True)
    prepared.meta['cover_path'] = str(cover_path)
    return prepared


class WeChatClient:
    '''Call the WeChat Official Account draft and media APIs.'''

    api_base = 'https://api.weixin.qq.com/cgi-bin'

    def __init__(self, access_token: str, *, timeout: float = 20.0):
        token = str(access_token or '').strip()
        if not token:
            raise ValueError('WeChat Official Account stable token is required.')
        self.access_token = token
        self.timeout = timeout

    def add_draft(self, article: dict[str, Any]) -> str:
        '''Create a draft containing one article and return its media ID.'''
        payload = self._request_json('POST', '/draft/add', json={'articles': [article]})
        media_id = str(payload.get('media_id') or '').strip()
        if not media_id:
            raise RuntimeError('WeChat draft/add returned no media_id.')
        return media_id

    def update_draft(self, media_id: str, article: dict[str, Any], *, index: int = 0) -> None:
        '''Replace one article in an existing draft.'''
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError('WeChat draft article index must be a non-negative integer.')
        self._request_json('POST', '/draft/update', json={
            'media_id': media_id,
            'index': index,
            'articles': article,
        })

    def get_draft(self, media_id: str) -> dict[str, Any]:
        '''Return one draft by media ID.'''
        media_id = str(media_id or '').strip()
        if not media_id:
            raise ValueError('WeChat draft media_id is required.')
        return self._request_json('POST', '/draft/get', json={'media_id': media_id})

    def batch_get_drafts(
        self,
        offset: int = 0,
        count: int = 20,
        *,
        no_content: bool = False,
    ) -> dict[str, Any]:
        '''Return one page of drafts.'''
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError('WeChat draft offset must be a non-negative integer.')
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 20:
            raise ValueError('WeChat draft count must be between 1 and 20.')
        return self._request_json('POST', '/draft/batchget', json={
            'offset': offset,
            'count': count,
            'no_content': 1 if no_content else 0,
        })

    def upload_body_image(self, path: Path) -> str:
        '''Upload an article body image and return its hosted URL.'''
        data, mime = self._image_file(path, body=True)
        payload = self._request_json(
            'POST', '/media/uploadimg',
            files={'media': (path.name, data, mime)},
        )
        url = str(payload.get('url') or '').strip()
        if not url:
            raise RuntimeError('WeChat media/uploadimg returned no URL.')
        return url

    def upload_cover(self, filename: str, data: bytes, mime: str) -> str:
        '''Upload permanent cover bytes and return their media ID.'''
        payload = self._request_json(
            'POST', '/material/add_material',
            params={'type': 'image'},
            files={'media': (filename, data, mime)},
        )
        media_id = str(payload.get('media_id') or '').strip()
        if not media_id:
            raise RuntimeError('WeChat material/add_material returned no media_id.')
        return media_id

    def upload_cover_file(self, path: Path) -> str:
        '''Upload a cover image file and return its media ID.'''
        data, mime = self._image_file(path, body=False)
        return self.upload_cover(path.name, data, mime)

    @staticmethod
    def _image_file(path: Path, *, body: bool) -> tuple[bytes, str]:
        if not path.is_file():
            raise ValueError(f'Image file does not exist: {path}.')
        mime = mimetypes.guess_type(path.name)[0] or ''
        allowed = {'image/jpeg', 'image/png'} if body else {
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
        }
        if mime not in allowed:
            scope = 'body' if body else 'cover'
            raise ValueError(f'WeChat {scope} image {path.name!r} has unsupported type {mime or "unknown"}.')
        data = path.read_bytes()
        if body and len(data) > 1024 * 1024:
            raise ValueError(f'WeChat body image {path.name!r} exceeds 1 MB.')
        return data, mime

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        params = {**kwargs.pop('params', {}), 'access_token': self.access_token}
        if 'json' in kwargs:
            kwargs['data'] = json.dumps(
                kwargs.pop('json'), ensure_ascii=False, separators=(',', ':'),
            ).encode('utf-8')
            kwargs.setdefault('headers', {})['Content-Type'] = 'application/json; charset=utf-8'
        try:
            response = requests.request(
                method,
                f'{self.api_base}{path}',
                params=params,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            body = str(getattr(getattr(exc, 'response', None), 'text', '') or '').strip()
            detail = f'HTTP {status}' if status is not None else str(exc)
            if body:
                detail += f'; response={body[:1000]}'
            lazyllm.LOG.error(f'WeChat API {path} HTTP failure: {detail}')
            raise RuntimeError(f'WeChat API {path} request failed: {detail}') from exc
        try:
            payload = json.loads(response.content.decode('utf-8-sig'))
        except ValueError as exc:
            body = str(getattr(response, 'text', '') or '').strip()
            detail = f'response={body[:1000]}' if body else str(exc)
            raise RuntimeError(f'WeChat API {path} returned invalid JSON: {detail}') from exc
        if not isinstance(payload, dict):
            raise TypeError(f'WeChat API {path} returned an invalid response.')
        errcode = payload.get('errcode')
        if errcode not in (None, 0, '0'):
            errmsg = str(payload.get('errmsg') or 'unknown error')
            lazyllm.LOG.error(
                f'WeChat API {path} returned error: errcode={errcode}; '
                f'errmsg={errmsg}; payload={json.dumps(payload, ensure_ascii=False)[:1000]}',
            )
            raise RuntimeError(f'WeChat API {path} failed ({errcode}): {errmsg}')
        return payload


class WeChatWriterProvider(WriterProviderBase):
    provider = 'wechat'
    capabilities = WriterProviderCapabilities(
        load=True,
        create=True,
        replace=True,
        patch=True,
        revision_check=True,
        media=True,
    )

    @classmethod
    def matches(cls, locator: str) -> bool:
        request = str(locator or '')
        return (
            all(term in request for term in _WECHAT_DRAFT_REVISION_TERMS)
            and any(term in request for term in _WECHAT_REVISION_INTENT_TERMS)
        )

    def resolve(self, locator: str) -> TargetDocument:
        request = str(locator or '')
        if not self.matches(request):
            raise ValueError(f'Invalid WeChat draft request: {locator!r}.')
        for draft in self.list_drafts():
            media_id = str(draft.get('media_id') or '').strip()
            if not media_id:
                continue
            for article_index, article in enumerate(self._news_items(draft)):
                title = str(article.get('title') or '').strip()
                if title and title in request:
                    return TargetDocument(
                        doc_id=media_id,
                        adapter=self.provider,
                        title=title,
                        meta={
                            'article_index': article_index,
                            'browser_url': _MP_HOME,
                        },
                    )
        raise ValueError(_DRAFT_TITLE_NOT_FOUND)

    def list_drafts(self, *, page_size: int = 20) -> list[dict[str, Any]]:
        '''Return drafts in WeChat's order, including article titles.'''
        client = WeChatClient(self._access_token())
        offset = 0
        drafts: list[dict[str, Any]] = []
        while True:
            payload = client.batch_get_drafts(offset, page_size, no_content=True)
            items = payload.get('item') or []
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                raise RuntimeError('WeChat draft/batchget returned an invalid item list.')
            drafts.extend(items)
            total = payload.get('total_count')
            if not items or (isinstance(total, int) and len(drafts) >= total) or len(items) < page_size:
                return drafts
            offset += len(items)

    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
    ) -> dict:
        target = self._normalize_target(target)
        media_id = str(target.doc_id or '').strip()
        if not media_id:
            raise ValueError('WeChat draft target requires a media_id.')
        article_index = self._article_index(target)
        client = WeChatClient(self._access_token())
        payload = client.get_draft(media_id)
        articles = self._news_items(payload)
        if article_index >= len(articles):
            raise ValueError(
                f'WeChat draft {media_id!r} has {len(articles)} articles; '
                f'article index {article_index} is out of range.')
        article = articles[article_index]
        html = str(article.get('content') or '')
        if not html:
            raise ValueError(f'WeChat draft article {media_id!r}[{article_index}] has no content.')
        update_time = payload.get('update_time')
        if update_time is None:
            update_time = payload.get('updateTime')
        external_id = f'{media_id}:article-{article_index}'
        document = WeChatWriterAdapter().html_to_ir(
            html,
            external_document_id=external_id,
            stage=stage,
            title=str(article.get('title') or ''),
            revision=str(update_time) if update_time is not None else None,
        )
        document.provider_binding.update({
            'document_id': media_id,
            'article_index': article_index,
            'media_id': media_id,
            'thumb_media_id': str(article.get('thumb_media_id') or ''),
            'browser_url': _MP_HOME,
        })
        document.metadata['wechat_article'] = {
            key: value for key, value in article.items() if key in _ARTICLE_FIELDS and key != 'content'
        }
        document.metadata['wechat_media_id'] = media_id
        document.metadata['wechat_article_index'] = article_index
        resolved_target = target.model_copy(deep=True)
        resolved_target.doc_id = media_id
        resolved_target.uri = None
        resolved_target.adapter = self.provider
        resolved_target.title = str(article.get('title') or target.title or '')
        resolved_target.meta.update({
            'article_index': article_index,
            'thumb_media_id': str(article.get('thumb_media_id') or ''),
            'browser_url': _MP_HOME,
        })
        return {
            'doc_id': media_id,
            'provider': self.provider,
            'adapter': self.provider,
            'locator': '',
            'target_document': resolved_target,
            'source_document': document,
            'block_count': sum(1 for _ in document.iter_blocks()),
            'representation': 'ir',
        }

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        if not str(title or '').strip():
            raise ValueError('title is required')
        return TargetDocument(
            adapter=self.provider,
            title=title.strip(),
            meta={'browser_url': _MP_HOME},
        )

    def document_image_resources(
        self,
        document: WriterDocument,
    ) -> tuple[list[InputResource], list[str]]:
        resources: list[InputResource] = []
        warnings: list[str] = []
        for block in (item for item in document.iter_blocks() if item.type == 'image'):
            reference = next((
                ref for ref in block.references
                if ref.get('type') == 'wechat_image'
                and str(ref.get('url') or '').strip()
            ), None)
            url = str(reference.get('url') or '').strip() if reference else ''
            if not url:
                warnings.append(f'WeChat image block {block.node_id!r} has no image URL.')
                continue
            resources.append(InputResource(
                resource_id=f'wechat-image-{block.node_id}',
                resource_type='image',
                uri=url,
                title=block.content or f'WeChat image {block.node_id}',
                summary=block.content or None,
                meta={
                    'provider': self.provider,
                    'provider_block_id': block.node_id,
                    'source_type': 'input_resource',
                    'origin': 'source_document',
                    'caption': block.content or None,
                },
            ))
        return resources, warnings

    def _convert_native_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
        template: str | None = None,
    ) -> WriterProviderDocument:
        document = self._writer_document(content, media_assets)
        media_references: dict[str, str] = {}
        for block in document.iter_blocks():
            if block.type != 'image' or WeChatWriterAdapter.can_reuse_raw(block):
                continue
            reference = next((
                item for item in block.references
                if item.get('type') == 'media_asset' and item.get('id')
            ), None)
            asset_id = str((reference or {}).get('id') or '')
            asset = media_assets.assets.get(asset_id) if media_assets and asset_id else None
            url = str(
                (reference or {}).get('url')
                or (reference or {}).get('path')
                or (asset.uri if asset else '')
                or (asset.local_path if asset else '')
            ).strip()
            if not asset_id or not url:
                raise ValueError(f'Image block {block.node_id!r} media is unavailable.')
            media_references[asset_id] = url
        html = WeChatWriterAdapter().document_to_html(
            document, media_references, template=template,
        )
        return WriterProviderDocument(
            provider=self.provider,
            format='html',
            content=html,
            source_document=document,
            media_references=media_references,
        )

    def convert_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
        output_format: str = 'native',
        template: str | None = None,
    ) -> WriterProviderDocument:
        if output_format != 'native':
            return super().convert_document(
                content,
                target=target,
                media_assets=media_assets,
                output_format=output_format,
            )
        return self._convert_native_document(
            content,
            target=target,
            media_assets=media_assets,
            template=template,
        )

    def convert_document_with_template(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
        template: str | None = None,
    ) -> WriterProviderDocument:
        return self.convert_document(
            content,
            target=target,
            media_assets=media_assets,
            template=template,
        )

    def write_document(
        self,
        converted: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        if converted.provider != self.provider or converted.format != 'html':
            raise ValueError('WeChat write_document requires converted WeChat HTML.')
        if mode != 'replace':
            raise WriterProviderCapabilityError(self.provider, 'append')
        target = self._normalize_target(target)
        document = converted.source_document.model_copy(deep=True)
        title = document.title.strip() or str(target.title or '').strip()
        if not title:
            raise ValueError('WeChat draft title is required.')

        client = WeChatClient(self._access_token())
        image_assets = list(self._document_image_assets(document, media_assets))
        image_urls = {
            asset_id: client.upload_body_image(path) for asset_id, path in image_assets
        }
        html = str(converted.content)
        for asset_id, uploaded_url in image_urls.items():
            source_url = converted.media_references.get(asset_id, '')
            if source_url:
                html = html.replace(escape(source_url, quote=True), uploaded_url)
        existing_binding = document.provider_binding if (
            document.provider_binding.get('provider') == self.provider
        ) else {}
        media_id = str(target.doc_id or existing_binding.get('document_id') or '').strip()
        article_index = self._article_index(target, document=document)
        thumb_media_id = str(
            target.meta.get('thumb_media_id')
            or existing_binding.get('thumb_media_id')
            or ''
        ).strip()
        thumb_media_id = self._resolve_cover(client, target, media_id, article_index, thumb_media_id)

        previous_article = document.metadata.get('wechat_article')
        article = {
            key: value for key, value in (previous_article.items() if isinstance(previous_article, dict) else [])
            if key in _ARTICLE_FIELDS
        }
        article.update({
            'article_type': article.get('article_type') or 'news',
            'title': title,
            'author': str(document.metadata.get('author', article.get('author', '')) or '')[:16],
            'digest': str(document.metadata.get('digest', article.get('digest', '')) or '')[:120],
            'content': html,
            'content_source_url': str(document.metadata.get(
                'content_source_url', article.get('content_source_url', '')) or ''),
            'thumb_media_id': thumb_media_id,
            'need_open_comment': int(document.metadata.get(
                'need_open_comment', article.get('need_open_comment', 0)) or 0),
            'only_fans_can_comment': int(document.metadata.get(
                'only_fans_can_comment', article.get('only_fans_can_comment', 0)) or 0),
        })
        if media_id:
            client.update_draft(media_id, article, index=article_index)
        else:
            media_id = client.add_draft(article)

        document.title = title
        document.stage = 'final'
        document.ui_editable = True
        document.provider_binding = {
            'provider': self.provider,
            'document_id': media_id,
            'browser_url': _MP_HOME,
            'thumb_media_id': thumb_media_id,
            'article_index': article_index,
            'media_id': media_id,
        }
        document.metadata['wechat_article'] = {
            key: value for key, value in article.items() if key in _ARTICLE_FIELDS and key != 'content'
        }
        document.metadata['wechat_media_id'] = media_id
        document.metadata['wechat_article_index'] = article_index
        return {
            'doc_id': media_id,
            'adapter': self.provider,
            'locator': '',
            'block_count': sum(1 for _ in document.iter_blocks()),
            'persisted_document': document,
            'representation': 'ir',
        }

    def apply_patch_to_document(
        self,
        patch_set: PatchSet,
        source_document: WriterDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        target = self._normalize_target(target)
        media_id = str(
            target.doc_id or source_document.provider_binding.get('document_id') or ''
        ).strip()
        if not media_id:
            raise ValueError('WeChat patch requires a bound draft media_id.')
        if source_document.revision is not None:
            baseline = WeChatClient(self._access_token()).get_draft(media_id)
            current_revision = baseline.get('update_time')
            if current_revision is None:
                current_revision = baseline.get('updateTime')
            if str(current_revision) != source_document.revision:
                raise WriterProviderRevisionError(
                    self.provider,
                    source_document.revision,
                    str(current_revision) if current_revision is not None else None,
                )
        revised, patch_result = apply_patch_to_ir(
            source_document, patch_set, media_assets=media_assets)
        write_result = self.replace_document(revised, target, media_assets=media_assets)
        persisted = write_result['persisted_document']
        patch_result.message = 'WeChat draft replaced successfully.'
        return {
            'patch_result': patch_result,
            'persisted_document': persisted,
            'provider': self.provider,
            'document_id': persisted.provider_binding['document_id'],
        }

    @staticmethod
    def _access_token() -> str:
        auth = lazyllm.globals.config['dynamic_tool_auth'] or {}
        token = auth.get(WeChatWriterProvider.provider)
        if isinstance(token, (list, tuple)):
            token = next((item for item in token if str(item or '').strip()), '')
        token = str(token or '').strip()
        if not token:
            raise ValueError('WeChat Official Account stable token is not configured.')
        return token

    def _normalize_target(self, target: TargetDocument) -> TargetDocument:
        return target

    @staticmethod
    def _article_index(
        target: TargetDocument,
        *,
        document: WriterDocument | None = None,
    ) -> int:
        value = target.meta.get('article_index')
        if value is None and document is not None:
            value = document.provider_binding.get('article_index')
            if value is None:
                value = document.metadata.get('wechat_article_index')
        if value is None:
            value = 0
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('WeChat draft article_index must be a non-negative integer.') from exc
        if index < 0 or isinstance(value, bool):
            raise ValueError('WeChat draft article_index must be a non-negative integer.')
        return index

    @staticmethod
    def _news_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: Any = payload.get('news_item')
        if candidates is None:
            content = payload.get('content')
            candidates = content.get('news_item') if isinstance(content, dict) else None
        if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
            raise RuntimeError('WeChat draft response contains no valid news_item list.')
        return candidates

    @staticmethod
    def _document_image_assets(
        document: WriterDocument,
        media_assets: MediaAssetLibrary | None,
    ) -> Iterable[tuple[str, Path]]:
        seen = set()
        for block in document.iter_blocks():
            if block.type != 'image':
                continue
            if WeChatWriterAdapter.can_reuse_raw(block):
                continue
            asset_id = next((
                str(ref.get('id')) for ref in block.references
                if ref.get('type') == 'media_asset' and ref.get('id')
            ), '')
            if not asset_id or asset_id in seen:
                continue
            asset = media_assets.assets.get(asset_id) if media_assets else None
            path = Path(asset.local_path) if asset and asset.local_path else None
            if asset is None or path is None or not path.is_file():
                raise ValueError(f'Image media asset {asset_id!r} is unavailable.')
            seen.add(asset_id)
            yield asset_id, path

    def _resolve_cover(self, client, target, media_id, article_index, thumb_media_id):
        if not thumb_media_id:
            if media_id:
                articles = self._news_items(client.get_draft(media_id))
                if article_index >= len(articles):
                    raise ValueError(
                        f'WeChat draft {media_id!r} has {len(articles)} articles; '
                        f'article index {article_index} is out of range.')
                thumb_media_id = str(
                    articles[article_index].get('thumb_media_id') or '').strip()
                if not thumb_media_id:
                    raise ValueError('Existing WeChat draft article has no reusable cover.')
            else:
                cover_path = str(target.meta.get('cover_path') or '').strip()
                if cover_path:
                    try:
                        thumb_media_id = client.upload_cover_file(Path(cover_path))
                    except (OSError, ValueError) as exc:
                        lazyllm.LOG.warning(
                            'Cannot use WeChat cover %r; using the placeholder cover: %s',
                            cover_path, exc,
                        )
                if not thumb_media_id:
                    thumb_media_id = client.upload_cover(
                        'lazymind-cover.png', wechat_placeholder_cover_png(), 'image/png')
        return thumb_media_id


__all__ = [
    'WeChatClient',
    'WeChatWriterProvider',
    'prepare_wechat_cover',
]
