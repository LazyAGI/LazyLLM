import pytest

from lazyllm.tools.writer.utils.serialization import (
    apply_markdown_outline_instructions,
    parse_document_markdown,
    parse_markdown_outline_instructions,
)


@pytest.mark.parametrize('opening, inner, closing', [
    ('```text', '', '```'),
    ('````text', '```', '````'),
    ('```text', '```not-a-closing-fence', '```'),
    ('~~~~text', '~~~', '~~~~'),
])
def test_outline_examples_inside_code_remain_literal(opening, inner, closing):
    sidecar = '<!-- writer:outline {"target_chars":999} -->'
    code = '\n'.join([opening, inner, '<a id="block-fake"></a>', '## Fake', sidecar, closing])
    markdown = '\n'.join([
        '<a id="block-real"></a>', '## Real',
        '<!-- writer:outline {"target_chars":42} -->', '', code, '',
    ])
    assert parse_markdown_outline_instructions(markdown) == {'real': {'target_chars': 42, 'node_id': 'real'}}
    document = parse_document_markdown(markdown, document_id='fenced-outline')
    assert any(sidecar in block.content for block in document.iter_blocks() if block.type == 'code')
    restored = apply_markdown_outline_instructions(markdown, document)
    assert code in restored
    assert parse_markdown_outline_instructions(restored)['real']['target_chars'] == 42


def test_visible_sidecar_still_round_trips():
    markdown = '<a id="block-real"></a>\n## Real\n<!-- writer:outline {"target_chars":42} -->\n'
    document = parse_document_markdown(markdown, document_id='visible-outline')
    restored = apply_markdown_outline_instructions(markdown, document)
    assert restored.count('writer:outline') == 1
    assert parse_markdown_outline_instructions(restored)['real']['target_chars'] == 42
