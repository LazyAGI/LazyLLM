import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.fs.supplier.feishu import FeishuFS
from lazyllm.tools.writer.data_models import (
    MediaAssetLibrary,
    WriterBlock,
    WriterDocument,
    WritingTask,
)
from lazyllm.tools.writer.tools.multimodal_tools import WriterMultimodalTools
from lazyllm.tools.writer.utils import load_artifact_json


_PNG_BYTES = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
)


def test_collect_available_media_downloads_markdown_images(tmp_path):
    pytest.importorskip('mistune', minversion='3.0.0')
    markdown = tmp_path / 'source.md'
    markdown.write_text(
        '# 标题\n\n'
        '![用户原图](https://cdn.example.com/original.png "原图")\n\n'
        '![本地图片](./images/local.png)\n',
        encoding='utf-8',
    )
    tool = WriterMultimodalTools(artifact_store=str(tmp_path / 'media-store'))
    task = WritingTask(task_id='task-markdown', query='复用原图写文档', task_type='write')

    with patch.object(tool, '_download_external_image', return_value=_PNG_BYTES) as download:
        result = tool.collect_available_media(task=task, source_document=str(markdown))

    library = load_artifact_json(result['artifact_path'], MediaAssetLibrary)
    resources = load_artifact_json(
        result['metadata']['artifact_paths']['profile_input_resources'],
        validate_schema=False,
    )
    assert len(library.assets) == 1
    asset = next(iter(library.assets.values()))
    download.assert_called_once_with('https://cdn.example.com/original.png')
    assert asset.uri == 'https://cdn.example.com/original.png'
    assert asset.meta['origin'] == 'markdown'
    assert Path(asset.local_path).is_file()
    assert any(resource.get('uri') == asset.uri for resource in resources)
    assert result['metadata']['warnings'] == []


def test_external_image_download_streams_valid_image_bytes():
    response = MagicMock(
        status_code=200,
        headers={'Content-Type': 'image/png', 'Content-Length': str(len(_PNG_BYTES))},
    )
    response.iter_content.return_value = [_PNG_BYTES[:20], _PNG_BYTES[20:]]
    tool = WriterMultimodalTools()
    url = 'https://cdn.example.com/original.png'

    with patch.object(tool, '_validate_remote_url') as validate, patch(
        'lazyllm.tools.writer.tools.multimodal_tools.requests.get',
        return_value=response,
    ) as get:
        assert tool._download_external_image(url) == _PNG_BYTES

    validate.assert_called_once_with(url)
    get.assert_called_once_with(
        url,
        timeout=(5, 30),
        stream=True,
        allow_redirects=False,
        headers={'Accept': 'image/*', 'User-Agent': 'LazyLLM-Writer/1.0'},
    )
    response.close.assert_called_once_with()


def test_external_image_download_rejects_non_public_hosts():
    tool = WriterMultimodalTools()
    address_info = [(None, None, None, None, ('127.0.0.1', 443))]

    with patch(
        'lazyllm.tools.writer.tools.multimodal_tools.config',
        {'allow_internal_network': False},
    ), patch(
        'lazyllm.tools.writer.tools.multimodal_tools.socket.getaddrinfo',
        return_value=address_info,
    ), pytest.raises(ValueError, match='non-public image host'):
        tool._validate_remote_url('https://internal.example.com/image.png')


def test_collect_available_media_downloads_feishu_source_images(tmp_path):
    source = WriterDocument(
        document_id='feishu-doc-doc-1',
        stage='final',
        provider_binding={
            'provider': 'feishu',
            'document_id': 'doc-1',
            'uri': 'https://company.feishu.cn/docx/doc-1',
        },
        blocks=[WriterBlock(
            node_id='image-1',
            type='image',
            content='产品原图',
            stage='final',
            provider_binding={'provider': 'feishu', 'block_id': 'block-1'},
            provider_payload={'raw_block': {
                'block_id': 'block-1',
                'block_type': 27,
                'image': {'token': 'media-token-1'},
            }},
            editable=False,
        )],
    )
    fs = MagicMock()
    fs.download_media.return_value = _PNG_BYTES
    tool = WriterMultimodalTools(artifact_store=str(tmp_path / 'media-store'))
    task = WritingTask(task_id='task-feishu', query='复用飞书原图', task_type='write')

    with patch(
        'lazyllm.tools.fs.client.FS._parse',
        return_value=('feishu', None, '~docx/doc-1'),
    ), patch(
        'lazyllm.tools.fs.client.FS._get_or_create_fs',
        return_value=fs,
    ):
        result = tool.collect_available_media(task=task, source_document=source)

    library = load_artifact_json(result['artifact_path'], MediaAssetLibrary)
    asset = next(iter(library.assets.values()))
    fs.download_media.assert_called_once_with('media-token-1')
    assert asset.caption == '产品原图'
    assert asset.meta['provider_block_id'] == 'block-1'
    assert asset.meta['origin'] == 'source_document'
    assert 'provider_media_token' not in asset.meta
    assert Path(asset.local_path).is_file()
    assert result['metadata']['warnings'] == []


def test_feishu_download_media_uses_authenticated_media_endpoint():
    fs = object.__new__(FeishuFS)
    fs._base_url = 'https://open.feishu.cn/open-apis'
    fs._request = MagicMock(return_value=SimpleNamespace(content=_PNG_BYTES))

    assert fs.download_media('token/with space') == _PNG_BYTES
    fs._request.assert_called_once_with(
        'GET',
        'https://open.feishu.cn/open-apis/drive/v1/medias/token%2Fwith%20space/download',
    )
