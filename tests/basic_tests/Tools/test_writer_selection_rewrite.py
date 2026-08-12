from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models.revision import (
    StringReplace,
    StringReplaceSet,
)
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import (
    MarkdownSelectionError,
    locate_markdown_paragraph,
)


def _context():
    return {'context_id': 'context-1', 'meta': {}}


def test_locator_maps_rendered_inline_text_to_source_paragraph():
    source = '# 标题\n\n这是 **重要** 的[段落](https://example.com)。\n'
    assert locate_markdown_paragraph(source, '这是 重要 的段落。') == (
        '这是 **重要** 的[段落](https://example.com)。'
    )


def test_duplicate_paragraph_returns_before_llm_call():
    tool = WriterRevisionTools(llm=MagicMock())
    with patch.object(tool, '_call_llm_structured') as call:
        with pytest.raises(MarkdownSelectionError) as exc:
            tool.build_selected_markdown_replace_set(
                '重复内容。\n\n重复内容。\n', '润色', '重复内容。', _context(),
            )
    assert exc.value.error_code == 'SELECTION_AMBIGUOUS'
    call.assert_not_called()


def test_selected_paragraph_reuses_string_replace_apply(tmp_path):
    source = '# 标题\n\n旧段落。\n\n后续段落。\n'
    tool = WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))
    with patch.object(
        tool,
        '_call_llm_structured',
        return_value=StringReplace(old_string='旧段落。', new_string='新段落。'),
    ):
        payload = tool.build_selected_markdown_replace_set(
            source, '改写', '旧段落。', _context(),
        )

    replace_set = StringReplaceSet.model_validate(payload)
    assert replace_set.replacements[0].old_string == '旧段落。'
    assert replace_set.replacements[0].new_string == '新段落。'
    result = WriterRevisionTools(artifact_store=str(tmp_path)).apply_string_replace(
        source, replace_set, _context(),
    )
    assert Path(result['revised_document_md']).read_text(encoding='utf-8') == (
        '# 标题\n\n新段落。\n\n后续段落。\n'
    )
