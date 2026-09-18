import pytest

from lazyllm.tools.writer.data_models import TargetDocument, WriterBlock, WriterDocument, WriterSpan
from lazyllm.tools.writer.provider import (
    FeishuWriterProvider, GitHubWriterProvider, NotionWriterProvider,
    WeChatWriterProvider, WriterProviderBase,
)
from lazyllm.tools.writer.utils import export as writer_export


@pytest.mark.parametrize('provider_class', [
    FeishuWriterProvider, GitHubWriterProvider, NotionWriterProvider, WeChatWriterProvider,
])
def test_portable_formats_bypass_native_conversion_and_cannot_be_written(provider_class, monkeypatch):
    provider = provider_class()

    def unexpected(*args, **kwargs):
        pytest.fail('Portable conversion must not call the platform converter')

    monkeypatch.setattr(provider, '_convert_native_document', unexpected)
    for output_format in ('markdown', 'latex', 'text'):
        converted = provider.convert_document('# Hello\n\nWorld', output_format=output_format)
        assert converted.provider == ''
        assert converted.format == output_format
        assert 'Hello' in converted.content and 'World' in converted.content
        with pytest.raises(ValueError):
            provider.write_document(converted, TargetDocument(adapter=provider.provider, doc_id='unused'))


def test_latex_copy_uses_shared_markdown_to_latex_conversion(monkeypatch):
    calls = []

    def convert(content, source_format, target_format):
        calls.append((content, source_format, target_format))
        return '\\documentclass{article}\n'

    monkeypatch.setattr(writer_export, 'convert_writer_content', convert)
    result = WriterProviderBase.convert_common_document(
        '# Report\n\nCost is 50% & $x_1$.\n', output_format='latex',
    ).content

    assert result == '\\documentclass{article}\n'
    assert calls == [('# Report\n\nCost is 50% & $x_1$.', 'markdown', 'latex')]


def test_markdown_keeps_source_math_and_plain_text_removes_markup():
    source = '# Title\n\n**Bold** $x_1$ [Link](https://example.com)\n'
    markdown = WriterProviderBase.convert_common_document(source, output_format='markdown').content
    assert '**Bold** $x_1$' in markdown
    plain = WriterProviderBase.convert_common_document(source, output_format='text').content
    assert 'Bold x_1 Link (https://example.com)' in plain
    assert '**' not in plain and '# Title' not in plain


def test_ir_conversion_does_not_mutate_source_and_keeps_media_locator():
    document = WriterDocument(document_id='doc', title='Title', blocks=[
        WriterBlock(node_id='heading', type='heading', content='Section', numbering={'level': 1}),
        WriterBlock(node_id='image', type='image', content='Diagram', references=[
            {'type': 'media_asset', 'id': 'asset', 'url': 'https://example.com/image.png'},
        ]),
    ])
    before = document.model_dump()
    for output_format in ('markdown', 'latex', 'text'):
        result = WriterProviderBase.convert_common_document(document, output_format=output_format)
        assert 'Section' in result.content
        if output_format == 'latex':
            assert 'assets/image.png' in result.content
        else:
            assert 'https://example.com/image.png' in result.content
    assert document.model_dump() == before


def test_unknown_format_and_unrepresentable_block_fail_explicitly():
    with pytest.raises(ValueError, match='Unsupported Writer output format'):
        WriterProviderBase.convert_common_document('body', output_format='pdf')
    document = WriterDocument(document_id='doc', blocks=[
        WriterBlock(node_id='embed', type='embed', provider_payload={'secret': 'native-only'}),
    ])
    with pytest.raises(ValueError, match='no portable content'):
        WriterProviderBase.convert_common_document(document, output_format='markdown')


def test_ir_math_is_not_escaped_as_ordinary_markdown_text():
    document = WriterDocument(document_id='doc', title='Title', blocks=[
        WriterBlock(node_id='p', type='paragraph', content=r'Formula $x_1 + \alpha$ costs 50%'),
    ])
    result = WriterProviderBase.convert_common_document(document, output_format='latex')
    assert r'\(x_1 + \alpha\)' in result.content
    assert r'50\%' in result.content


def test_structured_table_exports_each_cell_once_with_rich_text_and_math(monkeypatch):
    document = WriterDocument(document_id='doc', blocks=[WriterBlock(
        node_id='table', type='table', children=[
            WriterBlock(node_id='row-1', type='table_row', children=[
                WriterBlock(
                    node_id='cell-1', type='table_cell', content='Metric',
                    numbering={'header': True},
                ),
                WriterBlock(
                    node_id='cell-2', type='table_cell', content='Value',
                    numbering={'header': True},
                ),
            ]),
            WriterBlock(node_id='row-2', type='table_row', children=[
                WriterBlock(
                    node_id='cell-3', type='table_cell', content='DAU',
                    spans=[WriterSpan(text='DAU', style={'bold': True})],
                ),
                WriterBlock(node_id='cell-4', type='table_cell', content=r'$x_1$'),
            ]),
        ],
    )])

    markdown = writer_export.export_writer_document(document, 'markdown')
    plain = writer_export.export_writer_document(document, 'text')
    captured = []
    monkeypatch.setattr(
        writer_export,
        'convert_writer_content',
        lambda source, source_format, target_format: captured.append(source) or 'latex',
    )
    latex = writer_export.export_writer_document(document, 'latex')

    assert markdown.count('DAU') == 1 and '**DAU**' in markdown
    assert markdown.count('$x_1$') == 1
    assert plain.count('DAU') == 1 and plain.count('x_1') == 1
    assert latex == 'latex' and captured[0].count('DAU') == 1


def test_legacy_provider_subclass_can_still_override_convert_document():
    class LegacyProvider(WriterProviderBase):
        def convert_document(self, content, **kwargs):
            return content

        def write_document(self, *args, **kwargs):
            pass

        @classmethod
        def matches(cls, locator):
            return False

        def resolve(self, locator):
            pass

        def load_document(self, *args, **kwargs):
            pass

    assert LegacyProvider().convert_document('legacy') == 'legacy'


@pytest.mark.parametrize('output_format', ['markdown', 'text'])
def test_copy_removes_editor_anchors_and_extra_paragraph_spacing(output_format):
    source = '''# 故事

林远舟回到港口，章节见[后续内容](#block-sec-003)。

<a id="block-sec-003"></a>

## 不可名状的终焉

<a id="block-sec-003-001"></a>

### 城中的异变

码头上空无一人。



不，不完全如此。
'''
    result = WriterProviderBase.convert_common_document(source, output_format=output_format).content
    assert '<a ' not in result
    assert '#block-' not in result
    assert 'block-sec-' not in result
    assert '\n\n' not in result
    assert all(line.strip() for line in result.splitlines())
    assert '章节见后续内容。' in result
    assert '码头上空无一人。' in result


def test_markdown_copy_cleans_ir_anchors_without_changing_source():
    document = WriterDocument(document_id='doc', blocks=[
        WriterBlock(node_id='sec-003', type='heading', content='章节', numbering={'level': 1}),
        WriterBlock(node_id='p', type='paragraph', content='正文'),
    ])
    before = document.model_dump()
    result = WriterProviderBase.convert_common_document(document, output_format='markdown').content
    assert '<a ' not in result and '\n\n' not in result
    assert '章节' in result and '正文' in result
    assert document.model_dump() == before


def test_markdown_copy_preserves_code_examples_and_external_links():
    source = '''# Code

[外部链接](https://example.com/#block-section)

`<a id="block-example"></a>`

```markdown
<a id="block-in-code"></a>

[代码示例](#block-in-code)
```

[正文引用](#block-target)

<a id="block-target"></a>
## 目标
'''
    result = WriterProviderBase.convert_common_document(source, output_format='markdown').content
    assert '[外部链接](https://example.com/#block-section)' in result
    assert '`<a id="block-example"></a>`' in result
    assert '<a id="block-in-code"></a>\n\n[代码示例](#block-in-code)' in result
    assert '[正文引用](#block-target)' not in result
    assert '正文引用' in result
    assert '<a id="block-target"></a>' not in result


def test_native_conversion_keeps_original_platform_format():
    source = '# Native\n\n<a id="block-section"></a>\n\n## Section\n'
    assert GitHubWriterProvider().convert_document(source).content == source
