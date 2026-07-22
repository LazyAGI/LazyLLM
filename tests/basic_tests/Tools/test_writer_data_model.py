import json
import os
import tempfile

from lazyllm.tools.writer.data_models import (
    ResourceProfile,
    WriterAuthoring,
    WriterBlock,
    WriterConstraints,
    WriterDocument,
    WriterSpan,
    WritingContext,
)
from lazyllm.tools.writer.tools.base import WriterToolBase
from lazyllm.tools.writer.utils import (
    ToolResult,
    load_artifact_json,
    save_artifact_json,
)


def _make_writer_document():
    paragraph = WriterBlock(
        node_id='block-1',
        type='paragraph',
        content='正文内容',
        spans=[WriterSpan(text='正文内容', style=['strong'])],
        stage='draft',
        provider_binding={'provider': 'feishu', 'block_id': 'external-1'},
        provider_payload={'raw_type': 'paragraph'},
    )
    section = WriterBlock(
        node_id='section-1',
        type='heading',
        content='第一章',
        stage='draft',
        numbering={'level': 1},
        authoring=WriterAuthoring(
            instruction_id='instruction-1',
            constraints=WriterConstraints(section_goal='介绍背景'),
        ),
        children=[paragraph],
    )
    return WriterDocument(
        document_id='document-1',
        stage='draft',
        title='测试文档',
        blocks=[section],
        revision='rev-1',
        metadata={'source': 'test'},
        provider_binding={'provider': 'feishu', 'document_id': 'external-doc-1'},
    )


def test_writer_document_roundtrip_preserves_nested_ir_fields():
    document = _make_writer_document()
    restored = WriterDocument.model_validate_json(document.model_dump_json())

    assert restored.document_id == document.document_id
    assert restored.provider_binding['document_id'] == 'external-doc-1'
    assert restored.blocks[0].children[0].spans[0].style == ['strong']
    assert restored.blocks[0].children[0].provider_payload == {'raw_type': 'paragraph'}


def test_writer_document_iter_blocks_and_block_by_id_traverse_depth_first():
    document = _make_writer_document()

    assert [block.node_id for block in document.iter_blocks()] == ['section-1', 'block-1']
    assert document.block_by_id('block-1') is document.blocks[0].children[0]


def test_artifact_envelope_fields():
    context = WritingContext(context_id='ctx-1')
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'context.json')
        context.save(path, created_by='test')
        with open(path, encoding='utf-8') as handle:
            raw = json.load(handle)

        assert {'schema', 'schema_version', 'data', 'meta'} <= raw.keys()
        assert raw['meta']['created_by'] == 'test'


def test_standalone_save_load_functions_support_models_and_lists():
    document = _make_writer_document()
    profiles = [{'resource_id': 'r1', 'resource_role': 'background'}]

    with tempfile.TemporaryDirectory() as d:
        document_path = os.path.join(d, 'document.json')
        profiles_path = os.path.join(d, 'profiles.json')

        returned_document_path = save_artifact_json(document, document_path)
        returned_profiles_path = save_artifact_json(
            profiles,
            profiles_path,
            schema_name='lazyllm.tools.writer.artifacts.resource_profiles',
        )

        restored = load_artifact_json(returned_document_path, WriterDocument)
        loaded_profiles = load_artifact_json(
            returned_profiles_path,
            expected_schema_name='lazyllm.tools.writer.artifacts.resource_profiles',
        )

        assert os.path.isabs(returned_document_path)
        assert restored.block_by_id('block-1').content == '正文内容'
        assert loaded_profiles == profiles


def test_writer_tool_base_save_artifacts_metadata():
    document = _make_writer_document()
    context = WritingContext(context_id='ctx-1')
    profiles = [ResourceProfile(resource_id='r1', resource_role='background')]

    with tempfile.TemporaryDirectory() as d:
        result = WriterToolBase(artifact_store=d)._save_artifacts(
            {
                'document': document,
                'writing_context': context,
                'resource_profiles': profiles,
            },
            step_name='create_document',
            primary_key='document',
            summary='Created document.',
            counts={'blocks': 2, 'resource_profiles': 1},
        )

        assert isinstance(result, ToolResult)
        assert result.artifact_path.endswith('document.json')
        assert result.context_path.endswith('writing_context.json')
        assert result.metadata['schema_names']['resource_profiles'] == (
            'lazyllm.tools.writer.artifacts.resource_profiles'
        )
        assert result.metadata['counts']['blocks'] == 2
