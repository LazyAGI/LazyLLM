import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops
import pytest

from lazyllm.tools.writer.adapter.wechat import WeChatWriterAdapter
from lazyllm.tools.writer.data_models.multimodal import MediaAsset, MediaAssetLibrary
from lazyllm.tools.writer.data_models.revision import PatchHunk, PatchSet
from lazyllm.tools.writer.data_models.task import TargetDocument
from lazyllm.tools.writer.data_models.writer_ir import (
    WriterBlock,
    WriterDocument,
    WriterSpan,
)
from lazyllm.tools.writer.provider.wechat import (
    WeChatClient,
    WeChatWriterProvider,
)
from lazyllm.tools.writer.provider import WriterProviderRevisionError, match_writer_provider
from lazyllm.tools.writer.templates.wechat import get_wechat_template, list_wechat_templates
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.utils.artifact import deserialize_artifact_json
from lazyllm.tools.writer.utils.serialization import parse_document_markdown


def _patch_wechat_client(monkeypatch, client):
    monkeypatch.setattr('lazyllm.tools.writer.provider.wechat.WeChatClient', client)
    monkeypatch.setattr(
        WeChatWriterProvider,
        '_access_token',
        staticmethod(lambda: 'stable-token'),
    )


def _table_block() -> WriterBlock:
    table = parse_document_markdown(
        '| 项目 | 值 |\n| --- | --- |\n| 中文 | 正常 |',
        'table-document',
    ).blocks[0]
    table.node_id = 'table'
    return table


def test_wechat_provider_matches_and_resolves_prompt(monkeypatch):
    request = '请修改微信公众号草稿箱中的《目标文章》'
    monkeypatch.setattr(
        WeChatWriterProvider,
        'list_drafts',
        lambda self: [{
            'media_id': 'draft-1',
            'content': {'news_item': [{'title': '目标文章'}]},
        }],
    )

    provider = match_writer_provider(request)

    assert isinstance(provider, WeChatWriterProvider)
    assert provider.resolve(request) == TargetDocument(
        doc_id='draft-1',
        adapter='wechat',
        title='目标文章',
        meta={
            'article_index': 0,
            'browser_url': 'https://mp.weixin.qq.com/',
        },
    )


def test_wechat_templates_render_distinct_html_and_validate_ids():
    document = WriterDocument(
        document_id='template-document',
        title='模板测试',
        blocks=[
            WriterBlock(node_id='heading', type='heading', content='标题', numbering={'level': 1}),
            WriterBlock(node_id='paragraph', type='paragraph', content='正文'),
            WriterBlock(node_id='bullet', type='list_item', content='无序列表', numbering={'ordered': False}),
            WriterBlock(
                node_id='number', type='list_item', content='有序列表',
                numbering={'ordered': True, 'number': [2]},
            ),
            WriterBlock(node_id='quote', type='quote', content='引用'),
            WriterBlock(node_id='code', type='code', content='print(1)'),
            WriterBlock(
                node_id='reference', type='paragraph', content='参见标题',
                spans=[
                    WriterSpan(text='参见'),
                    WriterSpan(
                        text='标题',
                        style={'link': {'type': 'internal_ref', 'target_node_id': 'heading'}},
                    ),
                ],
            ),
            _table_block(),
        ],
    )
    adapter = WeChatWriterAdapter()

    clean_html = adapter.document_to_html(document, template='clean')
    structured_html = adapter.document_to_html(document, template='structured')
    editorial_html = adapter.document_to_html(document, template='editorial')

    assert len({clean_html, structured_html, editorial_html}) == 3
    assert 'background-color:#edf7f3;border-left:5px solid #2f806d' in clean_html
    assert clean_html.startswith(
        '<section style="color:#33373d;font-size:16px;line-height:1.9;margin:0 0 16px">'
    )
    assert '<p>正文</p>' in clean_html
    assert 'background-color:#2563eb;border-bottom:3px solid #1746aa' in structured_html
    assert 'background-color:#faf2ec;border-left:3px solid #b9795f' in editorial_html
    for html in (clean_html, structured_html, editorial_html):
        assert '<ul' not in html and '<ol' not in html and '<li' not in html
        assert '•' not in html and '::marker' not in html
        assert '<span style=' in html and '>&#8203;</span>无序列表</p>' in html
        assert '>2</span>有序列表</p>' in html
        assert ' id=' not in html
        assert 'href="#block-' not in html
        assert '参见标题' in html
        assert 'border-collapse:collapse' in html
        assert 'border:1px solid #222222' in html
        assert 'padding:6px 8px' in html
    assert [item['id'] for item in list_wechat_templates()] == [
        'clean', 'structured', 'editorial',
    ]
    assert get_wechat_template('wechat.structured').template_id == 'structured'
    assert (
        'background-color:#2563eb;border-radius:50%;display:inline-block;height:7px;'
        'margin-left:-23px;margin-right:16px;vertical-align:middle;width:7px'
    ) in structured_html
    assert (
        'background-color:#2563eb;border-radius:3px;color:#ffffff;display:inline-block;'
        'font-size:12px;font-weight:700;height:22px;line-height:22px;margin-left:-28px;'
        'margin-right:6px;min-width:22px;text-align:center;vertical-align:middle'
    ) in structured_html
    with pytest.raises(ValueError, match='Unknown WeChat template'):
        get_wechat_template('not-found')


def test_wechat_client_uses_draft_api_payloads(monkeypatch):
    requests = []

    class Response:
        content = json.dumps(
            {'news_item': [{'title': 'MacOS系统操作指南'}]},
            ensure_ascii=False,
        ).encode('utf-8')

        def raise_for_status(self):
            return None

    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        return Response()

    monkeypatch.setattr(
        'lazyllm.tools.writer.provider.wechat.requests.request',
        request,
    )
    client = WeChatClient('stable-token')

    draft = client.get_draft('media-1')
    client.batch_get_drafts(20, 3, no_content=True)
    client.update_draft('media-1', {'title': '目标文章'}, index=2)

    assert draft['news_item'][0]['title'] == 'MacOS系统操作指南'
    assert [item[0:2] for item in requests] == [
        ('POST', 'https://api.weixin.qq.com/cgi-bin/draft/get'),
        ('POST', 'https://api.weixin.qq.com/cgi-bin/draft/batchget'),
        ('POST', 'https://api.weixin.qq.com/cgi-bin/draft/update'),
    ]
    assert requests[0][2]['params'] == {'access_token': 'stable-token'}
    assert json.loads(requests[0][2]['data']) == {'media_id': 'media-1'}
    assert json.loads(requests[1][2]['data']) == {
        'offset': 20, 'count': 3, 'no_content': 1,
    }
    assert json.loads(requests[2][2]['data']) == {
        'media_id': 'media-1', 'index': 2,
        'articles': {'title': '目标文章'},
    }


def test_wechat_draft_create_then_update(monkeypatch, tmp_path: Path):
    calls = {}

    class Client:
        def __init__(self, token):
            calls['token'] = token

        def upload_cover(self, filename, data, mime):
            calls['cover'] = (filename, data, mime)
            return 'cover-media'

        def add_draft(self, article):
            calls['created'] = article
            return 'draft-media'

        def upload_body_image(self, path):
            calls['body_image'] = path
            return 'https://mmbiz.qpic.cn/body.png'

        def update_draft(self, media_id, article, *, index=0):
            calls['updated'] = (media_id, index, article)

    _patch_wechat_client(monkeypatch, Client)
    provider = WeChatWriterProvider()
    created = provider.replace_document(
        WriterDocument(
            document_id='writer-1',
            title='公众号测试文章',
            stage='final',
            blocks=[WriterBlock(node_id='intro', type='paragraph', content='初稿正文')],
        ),
        TargetDocument(adapter=provider.provider, title='公众号测试文章'),
    )

    assert calls['token'] == 'stable-token'
    assert calls['cover'][0] == 'lazymind-cover.png'
    assert calls['cover'][1].startswith(b'\x89PNG')
    with Image.open(BytesIO(calls['cover'][1])) as cover:
        assert cover.size == (900, 383)
        background = Image.new('RGB', cover.size, 'white')
        text_bounds = ImageChops.difference(cover.convert('RGB'), background).getbbox()
        assert text_bounds is not None
        center = ((text_bounds[0] + text_bounds[2]) / 2, (text_bounds[1] + text_bounds[3]) / 2)
        assert center == (450, 191.5)
    assert calls['created']['content'] == '<p>初稿正文</p>'
    persisted = created['persisted_document']
    assert persisted.provider_binding['document_id'] == 'draft-media'

    image_path = tmp_path / 'body.png'
    image_path.write_bytes(b'png')
    persisted.blocks = [
        WriterBlock(
            node_id='heading', type='heading', content='章节',
            numbering={'level': 1, 'ordered': True},
        ),
        WriterBlock(
            node_id='paragraph', type='paragraph', content='粗体和链接',
            spans=[WriterSpan(text='粗体', style={'bold': True}), WriterSpan(text='和链接')],
            references=[{'type': 'link', 'url': 'https://example.com', 'start': 3, 'end': 5}],
        ),
        _table_block(),
        WriterBlock(node_id='unsafe', type='paragraph', content='<script>'),
        WriterBlock(
            node_id='image', type='image', content='配图',
            references=[{'type': 'media_asset', 'id': 'asset-1'}],
        ),
    ]
    media = MediaAssetLibrary(
        library_id='library',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1', asset_type='image', source_type='web_search',
            local_path=str(image_path),
        )},
    )
    updated = provider.replace_document(
        persisted,
        TargetDocument(adapter=provider.provider),
        media_assets=media,
    )

    assert calls['body_image'] == image_path
    assert calls['updated'][0] == 'draft-media'
    assert calls['updated'][1] == 0
    article = calls['updated'][2]
    assert article['thumb_media_id'] == 'cover-media'
    assert (
        '<h2 style="background-color:#edf7f3;border-left:5px solid #2f806d;'
        'border-radius:3px;color:#20584b;font-size:20px;font-weight:700;line-height:1.6;'
        'margin:26px 0 12px;padding:8px 12px">'
        '1. 章节</h2>'
    ) in article['content']
    assert '<strong>粗体</strong>和<a href="https://example.com">链接</a>' in article['content']
    assert ' id=' not in article['content']
    assert 'border:1px solid #222222' in article['content']
    assert '>中文</td>' in article['content']
    assert '&lt;script&gt;' in article['content'] and '<script>' not in article['content']
    assert 'https://mmbiz.qpic.cn/body.png' in article['content']
    assert updated['persisted_document'].provider_binding['document_id'] == 'draft-media'


def test_wechat_draft_uses_prepared_cover(monkeypatch, tmp_path: Path):
    calls = {}
    cover_path = tmp_path / 'cover.png'
    Image.new('RGB', (900, 383), 'red').save(cover_path)

    class Client:
        def __init__(self, token):
            assert token == 'stable-token'

        def upload_cover_file(self, path):
            calls['cover_path'] = path
            return 'generated-cover'

        def add_draft(self, article):
            calls['article'] = article
            return 'draft-media'

    _patch_wechat_client(monkeypatch, Client)

    WeChatWriterProvider().replace_document(
        WriterDocument(
            document_id='writer-cover',
            title='封面测试',
            stage='final',
            blocks=[WriterBlock(node_id='body', type='paragraph', content='正文')],
        ),
        TargetDocument(
            adapter='wechat',
            title='封面测试',
            meta={'cover_path': str(cover_path)},
        ),
    )

    assert calls['cover_path'] == cover_path
    assert calls['article']['thumb_media_id'] == 'generated-cover'


def test_existing_wechat_draft_fetches_and_reuses_cover(monkeypatch):
    calls = {}

    class Client:
        def __init__(self, token):
            assert token == 'stable-token'

        def get_draft(self, media_id):
            assert media_id == 'draft-media'
            return {'news_item': [{'thumb_media_id': 'original-cover'}]}

        def update_draft(self, media_id, article, *, index=0):
            calls['updated'] = (media_id, index, article)

        def upload_cover(self, *args, **kwargs):
            raise AssertionError('existing draft must not upload a replacement cover')

        def upload_cover_file(self, *args, **kwargs):
            raise AssertionError('existing draft must not upload a replacement cover')

    _patch_wechat_client(monkeypatch, Client)

    WeChatWriterProvider().replace_document(
        WriterDocument(
            document_id='writer-existing',
            title='已有草稿',
            stage='final',
            blocks=[WriterBlock(node_id='body', type='paragraph', content='修改后的正文')],
        ),
        TargetDocument(adapter='wechat', doc_id='draft-media'),
    )

    assert calls['updated'][0:2] == ('draft-media', 0)
    assert calls['updated'][2]['thumb_media_id'] == 'original-cover'


def test_wechat_draft_read_patch_write_preserves_untouched_html(monkeypatch):
    calls = {}
    source_html = (
        '<p style="color:red"><strong>保留样式</strong>&nbsp;原段落</p>'
        '<section data-component="video-card"><video src="https://video.example/v.mp4"></video></section>'
        '<section data-component="remote-image">'
        '<img src="https://mmbiz.qpic.cn/image.png" style="width:80%;" /></section>'
        '<table data-layout="custom"><tr><th>A</th><th>B</th></tr>'
        '<tr><td rowspan="2">1</td><td>2</td></tr><tr><td>3</td></tr></table>'
        '<p>需要修改</p>'
    )

    class Client:
        def __init__(self, token):
            calls['token'] = token

        def get_draft(self, media_id):
            assert media_id == 'media-1'
            return {
                'update_time': 123,
                'news_item': [
                    {
                        'title': '其他文章', 'content': '<p>其他正文</p>',
                        'thumb_media_id': 'other-cover',
                    },
                    {
                        'title': '目标文章', 'author': '作者', 'digest': '摘要',
                        'content': source_html, 'thumb_media_id': 'target-cover',
                        'show_cover_pic': 1, 'need_open_comment': 1,
                        'only_fans_can_comment': 1,
                    },
                ],
            }

        def update_draft(self, media_id, article, *, index=0):
            calls['update'] = (media_id, index, article)

    _patch_wechat_client(monkeypatch, Client)

    provider = WeChatWriterProvider()
    target = TargetDocument(
        adapter=provider.provider,
        doc_id='media-1',
        meta={'article_index': 1},
    )
    loaded = provider.load_document(target)
    document = loaded['source_document']
    assert loaded['target_document'].meta['article_index'] == 1
    assert document.provider_binding['article_index'] == 1
    assert [block.type for block in document.blocks] == [
        'paragraph', 'wechat_opaque', 'image', 'table', 'paragraph',
    ]
    table = document.blocks[3]
    assert [row.type for row in table.children] == [
        'table_row', 'table_row', 'table_row',
    ]
    assert [[cell.content for cell in row.children] for row in table.children] == [
        ['A', 'B'], ['1', '2'], ['3'],
    ]
    assert table.children[1].children[0].numbering['row_span'] == 2

    adapter = WeChatWriterAdapter()
    assert adapter.document_to_html(document) == source_html
    clean_html = adapter.document_to_html(document, template='clean')
    assert clean_html != source_html
    assert 'rowspan="2"' in clean_html

    target_block = document.blocks[-1].model_copy(deep=True)
    target_block.content = '修改后的正文'
    target_block.spans = []
    document.revision = None
    patch = PatchSet(
        patch_id='patch-media-1',
        target_doc_id=document.document_id,
        hunks=[PatchHunk(
            hunk_id='update-target',
            target_node_id=document.blocks[-1].node_id,
            modify_type='update',
            block=target_block,
        )],
    )
    result = provider.apply_patch_to_document(patch, document, target)

    media_id, index, article = calls['update']
    assert media_id == 'media-1'
    assert index == 1
    assert article['author'] == '作者'
    assert article['digest'] == '摘要'
    assert article['thumb_media_id'] == 'target-cover'
    assert article['show_cover_pic'] == 1
    assert article['need_open_comment'] == 1
    assert article['only_fans_can_comment'] == 1
    assert '<p>修改后的正文</p>' in article['content']
    assert (
        '<section data-component="video-card">'
        '<video src="https://video.example/v.mp4"></video></section>'
    ) in article['content']
    assert (
        '<section data-component="remote-image">'
        '<img src="https://mmbiz.qpic.cn/image.png" style="width:80%;" /></section>'
    ) in article['content']
    assert '<table data-layout="custom"><tr><th>A</th><th>B</th></tr>' in article['content']
    assert result['persisted_document'].provider_binding['article_index'] == 1


def test_wechat_patch_rejects_stale_remote_revision(monkeypatch):
    calls = {'reads': 0}

    class Client:
        def __init__(self, token):
            assert token == 'stable-token'

        def get_draft(self, media_id):
            calls['reads'] += 1
            return {
                'update_time': 124 if calls['reads'] > 1 else 123,
                'news_item': [{
                    'title': '目标文章',
                    'content': '<p>原始正文</p>',
                    'thumb_media_id': 'cover-1',
                }],
            }

        def update_draft(self, media_id, article, *, index=0):
            raise AssertionError('stale source must not be written')

    _patch_wechat_client(monkeypatch, Client)
    provider = WeChatWriterProvider()
    target = TargetDocument(adapter='wechat', doc_id='media-1')
    document = provider.load_document(target)['source_document']
    changed = document.blocks[0].model_copy(update={'content': '修改后的正文'})

    with pytest.raises(
        WriterProviderRevisionError, match='changed since it was loaded',
    ) as captured:
        provider.apply_patch_to_document(
            PatchSet(
                target_doc_id=document.document_id,
                hunks=[PatchHunk(
                    target_node_id=changed.node_id,
                    modify_type='update',
                    block=changed,
                )],
            ),
            document,
            target,
        )

    assert captured.value.code == 'REVISION_CONFLICT'
    assert captured.value.details == {
        'provider': 'wechat',
        'expected_revision': '123',
        'actual_revision': '124',
    }


def test_wechat_conversion_does_not_preempt_provider_content_limits():
    document = WriterDocument(
        document_id='long-document',
        title='长文章',
        blocks=[
            WriterBlock(node_id=f'paragraph-{index}', type='paragraph', content='内容' * 125)
            for index in range(81)
        ],
    )

    converted = WeChatWriterProvider().convert_document(document, template='clean')

    assert len(converted.content) > 20_000


def test_wechat_markdown_code_and_quote_render_without_markdown_markers():
    document = parse_document_markdown(
        '```python\nprint(1)\n```\n\n> 引用正文',
        document_id='markdown-article',
        stage='final',
    )

    html = WeChatWriterAdapter().document_to_html(document, template='clean')

    assert '<code>print(1)</code>' in html
    assert '<blockquote' in html and '>引用正文</blockquote>' in html
    assert '```' not in html and '&gt; 引用正文' not in html


def test_wechat_writeback_renumbers_unchanged_heading_after_delete():
    source_html = (
        '<h2>1. 第一章</h2>'
        '<h2>2. 第二章</h2>'
        '<h2>3. 第三章</h2>'
        '<h2>4. 第四章</h2>'
        '<h3>4.1. 常用设置</h3>'
        '<h3>4.2. 性能优化与磁盘管理</h3>'
        '<h3 style="color:red">4.3. 常见故障诊断与解决方案</h3>'
    )
    adapter = WeChatWriterAdapter()
    document = adapter.html_to_ir(
        source_html,
        external_document_id='media-numbering:article-0',
    )
    document.blocks = [
        block for block in document.blocks
        if block.content != '性能优化与磁盘管理'
    ]

    html = adapter.document_to_html(document)

    assert '4.2. 常见故障诊断与解决方案' in html
    assert '4.3. 常见故障诊断与解决方案' not in html
    assert '<h3>4.1. 常用设置</h3>' in html
    assert '<h3 style="color:red">4.2. 常见故障诊断与解决方案</h3>' in html


def test_wechat_opaque_block_cannot_be_modified():
    document = WeChatWriterAdapter().html_to_ir(
        '<section data-component="unsupported"><custom-card /></section>',
        external_document_id='media-opaque:article-0',
    )
    document.blocks[0].content = '误修改'
    try:
        WeChatWriterAdapter().document_to_html(document)
    except ValueError as exc:
        assert 'cannot be modified' in str(exc)
    else:
        raise AssertionError('modified opaque WeChat HTML should be rejected')


def test_wechat_draft_list_is_paginated(monkeypatch):
    calls = []

    class Client:
        def __init__(self, token):
            assert token == 'stable-token'

        def batch_get_drafts(self, offset, count, *, no_content=False):
            calls.append((offset, count, no_content))
            pages = {
                0: [{'media_id': 'media-1'}, {'media_id': 'media-2'}],
                2: [{'media_id': 'media-3'}],
            }
            return {'total_count': 3, 'item': pages.get(offset, [])}

    _patch_wechat_client(monkeypatch, Client)

    drafts = WeChatWriterProvider().list_drafts(page_size=2)
    assert [item['media_id'] for item in drafts] == ['media-1', 'media-2', 'media-3']
    assert calls == [(0, 2, True), (2, 2, True)]


def test_wechat_resource_tools_read_patch_and_persist(monkeypatch, tmp_path: Path):
    calls = {}

    class Client:
        def __init__(self, token):
            assert token == 'stable-token'

        def get_draft(self, media_id):
            assert media_id == 'media-resource'
            return {
                'update_time': 456,
                'news_item': [{
                    'title': '编排测试',
                    'content': '<p>原始内容</p><section data-raw="1"><custom-card /></section>',
                    'thumb_media_id': 'cover-resource',
                }],
            }

        def update_draft(self, media_id, article, *, index=0):
            calls['update'] = (media_id, index, article)

    _patch_wechat_client(monkeypatch, Client)

    target = TargetDocument(
        adapter='wechat',
        doc_id='media-resource',
        meta={'article_index': 0},
    )
    resources = WriterResourceTools(llm=None, artifact_store=str(tmp_path))
    loaded = resources.document_to_docir(target.model_dump())
    source_path = loaded['metadata']['artifact_paths']['document']
    source = deserialize_artifact_json(
        Path(source_path).read_text(encoding='utf-8'), WriterDocument,
    )
    changed = source.blocks[0].model_copy(deep=True)
    changed.content = '修改后的内容'
    changed.spans = []
    patch = PatchSet(
        patch_id='resource-patch',
        target_doc_id=source.document_id,
        hunks=[PatchHunk(
            hunk_id='update-resource',
            target_node_id=changed.node_id,
            modify_type='update',
            block=changed,
        )],
    )
    result = resources.apply_patch_to_document(
        patch.model_dump(), source.model_dump(), target.model_dump(),
    )
    persisted_path = result['metadata']['artifact_paths']['persisted_document']
    persisted = deserialize_artifact_json(
        Path(persisted_path).read_text(encoding='utf-8'), WriterDocument,
    )

    assert calls['update'][0:2] == ('media-resource', 0)
    assert '<p>修改后的内容</p>' in calls['update'][2]['content']
    assert '<section data-raw="1"><custom-card /></section>' in calls['update'][2]['content']
    assert persisted.provider_binding['document_id'] == 'media-resource'
