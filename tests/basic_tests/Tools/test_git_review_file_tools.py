import pytest

from lazyllm.tools.agent import ToolExecutionError
from lazyllm.tools.git.review.pre_analysis.agent_tools import _build_scoped_agent_tools
from lazyllm.tools.git.review.pre_analysis.arch_pipeline import _arch_collect_snippets


def test_scoped_file_tools_use_current_interfaces_and_keep_boundary(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    source = repo / 'module.py'
    source.write_text('header\ndef needle():\n    return 1\nneedle()\n')
    (repo / 'nested').mkdir()
    (repo / 'nested/note.txt').write_text('needle')
    outside = tmp_path / 'outside.txt'
    outside.write_text('private')
    (repo / 'link').symlink_to(outside)
    tools = {tool.__name__: tool for tool in _build_scoped_agent_tools(str(repo))}
    assert tools['read_file_scoped']('module.py', start_line=2, end_line=3)['content'] == (
        'def needle():\n    return 1\n')
    for path in [str(outside), '../outside.txt', 'link']:
        with pytest.raises(ToolExecutionError, match='outside the allowed root'):
            tools['read_file_scoped'](path)
    with pytest.raises(ToolExecutionError, match='outside the allowed root'):
        tools['list_dir_scoped']('..', recursive=True)
    assert 'nested/note.txt' in tools['list_dir_scoped']('.', recursive=True)['entries']
    result = tools['search_scoped']('needle', glob='*.py', max_results=1)
    assert len(result['results']) == 1 and result['truncated']
    assert result['results'][0]['path'] == str(source)
    assert tools['grep_callers']('needle', max_results=1)['results']
    snippets = _arch_collect_snippets(str(repo), {'search_hints': ['def needle']})
    assert 'def needle():' in snippets and 'return 1' in snippets
