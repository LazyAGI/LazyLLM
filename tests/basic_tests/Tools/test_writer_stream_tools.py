import json

import pytest

from lazyllm.tools.writer.data_models import ContentRef, WriterBlock
from lazyllm.tools.writer.data_models.planning import SectionInstruction
from lazyllm.tools.writer.tools.stream_tools import (
    DraftIRStream,
    DraftMarkdownStream,
    DraftPreviewStream,
    IRJSONMarkdownParser,
    MarkdownStreamNormalizer,
)
from lazyllm.tools.writer.utils import render_block_markdown


def _instruction(title='Section'):
    return SectionInstruction(
        instruction_id='instruction-1',
        content_ref=ContentRef(node_id='section-1'),
        section_title=title,
        section_goal='Cover the requested points.',
    )


def _section_block(*children, title='Section'):
    return WriterBlock(
        node_id='section-1',
        type='heading',
        content=title,
        children=list(children),
    )


def test_markdown_normalizer_removes_fragmented_leading_think_blocks():
    normalizer = MarkdownStreamNormalizer()

    assert normalizer.feed('  \n<TH') == []
    assert normalizer.feed('ink data-kind="reasoning">hidden') == []
    assert normalizer.feed('</think>\n<think>hidden again</think>\n  body ') == ['body']
    assert normalizer.feed('\n\nnext  ') == [' \n\nnext']
    assert normalizer.finish() == []
    assert normalizer.body == 'body \n\nnext'


def test_markdown_normalizer_treats_incomplete_think_prefix_as_body_at_finish():
    normalizer = MarkdownStreamNormalizer()

    assert normalizer.feed('  <thi') == []
    assert normalizer.finish() == ['<thi']
    assert normalizer.body == '<thi'


def test_draft_preview_stream_requires_consumption_and_finalizes_result():
    def call(sink):
        sink({'delta': 'first'})
        sink({'delta': 'second'})
        return 'response'

    stream = DraftPreviewStream(
        call=call,
        consume=lambda payload: [payload['delta'].upper()],
        finalize=lambda response: (['done'], {'response': response}),
        idle_timeout=1,
        initial_deltas=['start'],
        label='Test preview',
    )

    with pytest.raises(RuntimeError, match='fully consumed'):
        stream.result()

    assert list(stream) == ['start', 'FIRST', 'SECOND', 'done']
    assert stream.result() == {'response': 'response'}


def test_draft_preview_stream_preserves_worker_error_for_result():
    error = RuntimeError('worker failed')

    def call(_sink):
        raise error

    stream = DraftPreviewStream(
        call=call,
        consume=lambda _payload: [],
        finalize=lambda _response: ([], {}),
        idle_timeout=1,
    )

    with pytest.raises(RuntimeError, match='worker failed') as raised:
        list(stream)
    assert raised.value is error

    with pytest.raises(RuntimeError, match='worker failed') as raised_again:
        stream.result()
    assert raised_again.value is error


def test_draft_markdown_stream_filters_tags_and_normalizes_body():
    finalized = []

    def call(sink):
        sink({'tag': 'think', 'delta': 'provider reasoning'})
        sink({'tag': 'text', 'delta': '  <thi'})
        sink({'tag': 'text', 'delta': 'nk>inline reasoning</think>\n\n body '})
        sink({'tag': 'text', 'delta': '\nnext  '})
        return 'body \nnext'

    def finalize(body):
        finalized.append(body)
        return {'body': body}

    stream = DraftMarkdownStream(
        call=call,
        finalize=finalize,
        prefix='## Section\n\n',
        idle_timeout=1,
    )

    assert ''.join(stream) == '## Section\n\nbody \nnext\n'
    assert finalized == ['body \nnext']
    assert stream.result() == {'body': 'body \nnext'}


def test_draft_markdown_stream_rejects_response_mismatch():
    def call(sink):
        sink({'tag': 'text', 'delta': 'streamed body'})
        return 'different body'

    stream = DraftMarkdownStream(
        call=call,
        finalize=lambda body: {'body': body},
        prefix='',
        idle_timeout=1,
    )

    with pytest.raises(ValueError, match='does not match'):
        list(stream)
    with pytest.raises(ValueError, match='does not match'):
        stream.result()


def test_ir_json_parser_streams_content_with_prefixes_and_json_escapes():
    payload = {
        'node_id': 'section-1',
        'type': 'heading',
        'content': 'Section',
        'children': [
            {
                'node_id': 'paragraph-1',
                'content': '  alpha\n\u4e2d\U0001f600  ',
                'type': 'paragraph',
                'children': [],
            },
            {
                'node_id': 'heading-1',
                'type': 'heading',
                'content': 'Nested',
                'children': [],
            },
            {
                'node_id': 'item-1',
                'type': 'list_item',
                'content': 'First',
                'numbering': {'ordered': True},
                'children': [],
            },
        ],
    }
    raw = json.dumps(payload, ensure_ascii=True, separators=(',', ':'))
    block = WriterBlock.model_validate(payload)
    parser = IRJSONMarkdownParser(_instruction())
    deltas = [parser.prefix]
    first_body_offset = None

    for offset in range(0, len(raw), 7):
        current = parser.feed(raw[offset:offset + 7])
        if current and first_body_offset is None:
            first_body_offset = offset + 7
        deltas.extend(current)
    deltas.extend(parser.finish(block))

    assert first_body_offset is not None
    assert first_body_offset < len(raw)
    assert ''.join(deltas) == render_block_markdown(block, level=2).rstrip() + '\n'
    assert 'alpha\n\u4e2d\U0001f600' in ''.join(deltas)
    assert '\n\n### Nested\n\n1. First' in ''.join(deltas)


def test_ir_json_parser_buffers_non_streamable_parent_and_its_children():
    child = WriterBlock(
        node_id='image-1',
        type='image',
        content='Caption',
        children=[WriterBlock(
            node_id='paragraph-1',
            type='paragraph',
            content='Nested text',
        )],
    )
    block = _section_block(child)
    raw = block.model_dump_json(exclude_defaults=True)
    child_raw = child.model_dump_json(exclude_defaults=True)
    child_end = raw.index(child_raw) + len(child_raw)
    parser = IRJSONMarkdownParser(_instruction())

    assert parser.feed(raw[:child_end - 1]) == []
    buffered_delta = parser.feed(raw[child_end - 1:])
    final_delta = parser.finish(block)

    assert buffered_delta == ['Caption\n\nNested text']
    assert ''.join([parser.prefix, *buffered_delta, *final_delta]) == (
        render_block_markdown(block, level=2).rstrip() + '\n'
    )


@pytest.mark.parametrize('invalid_json', [
    '{"content":"\\x"}',
    '{"content":"\\uD83D"}',
])
def test_ir_json_parser_rejects_invalid_string_escapes(invalid_json):
    parser = IRJSONMarkdownParser(_instruction())

    with pytest.raises(ValueError):
        parser.feed(invalid_json)


def test_ir_json_parser_rejects_preview_that_differs_from_validated_block():
    streamed = _section_block(WriterBlock(
        node_id='paragraph-1',
        type='paragraph',
        content='Streamed body',
    ))
    validated = _section_block(WriterBlock(
        node_id='paragraph-1',
        type='paragraph',
        content='Validated body',
    ))
    parser = IRJSONMarkdownParser(_instruction())

    parser.feed(streamed.model_dump_json(exclude_defaults=True))

    with pytest.raises(ValueError, match='does not match'):
        parser.finish(validated)


def test_draft_ir_stream_validates_normalizes_and_finalizes_response():
    response = _section_block(WriterBlock(
        node_id='paragraph-1',
        type='paragraph',
        content='Draft body',
    ))
    raw = response.model_dump_json(exclude_defaults=True)
    normalized = []

    def call(sink):
        sink({'tag': 'think', 'delta': 'provider reasoning'})
        for offset in range(0, len(raw), 11):
            sink({'tag': 'text', 'delta': raw[offset:offset + 11]})
        return response.model_dump()

    def normalize(block):
        normalized.append(block)
        return block

    stream = DraftIRStream(
        call=call,
        normalize=normalize,
        finalize=lambda block: {'node_id': block.node_id},
        instruction=_instruction(),
        idle_timeout=1,
    )

    assert ''.join(stream) == render_block_markdown(response, level=2).rstrip() + '\n'
    assert normalized == [response]
    assert stream.result() == {'node_id': 'section-1'}
