import os
import tempfile


from lazyllm.tools import ToolManager
from lazyllm.tools.agent.file_tool import (read, write, ls, grep,
                                           move, remove)
from lazyllm.tools.agent.shell_tool import shell


class TestFileTool(object):
    def test_file_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'a.txt')
            res = write(path, 'hello\nworld', root=tmp)
            assert res['status'] == 'ok'

            res = read(path, root=tmp)
            assert res['status'] == 'ok'
            assert 'hello' in res['content']

            res = ls(tmp, root=tmp)
            assert res['status'] == 'ok'
            assert 'a.txt' in res['entries']

            res = grep('world', path=tmp, root=tmp)
            assert res['status'] == 'ok'
            assert any(item['path'].endswith('a.txt') for item in res['results'])

            dst = os.path.join(tmp, 'b.txt')
            res = move(path, dst, root=tmp)
            assert res['status'] == 'ok'

            res = remove(dst, root=tmp)
            assert res['status'] == 'ok'


class TestShellTool(object):
    def test_shell(self):
        res = shell('echo hello')
        assert res['status'] == 'ok'
        assert 'hello' in res['stdout']

    def test_shell_needs_approval(self):
        manager = ToolManager(['shell'])
        prepared = manager.prepare_tool_calls({
            'function': {'name': 'shell', 'arguments': {'cmd': 'echo approved'}},
        })
        result = manager.execute_prepared(prepared)
        assert result.results[0]['needs_approval'] is True
        assert result.records[0].reason == 'approval_required'

    def test_shell_executes_after_host_approval(self):
        manager = ToolManager(['shell'])
        prepared = manager.prepare_tool_calls({
            'function': {'name': 'shell', 'arguments': {'cmd': 'echo approved'}},
        })
        result = manager.execute_prepared(prepared, approved_indices=(0,))
        assert result.results[0]['ok'] is True
        assert 'approved' in result.results[0]['value']['stdout']


def test_read_windows_continue_through_long_unicode_lines(tmp_path):
    source = tmp_path / 'large.txt'
    content = ('中' * 12000 + '\nshort\n') * 5
    source.write_text(content)
    offset, chunks = 1, []
    while True:
        window = read(str(source), offset=offset, limit=2000, max_bytes=4096)
        assert len(window['content'].encode('utf-8')) <= 4096
        chunks.append(window['content'])
        if window['eof']:
            break
        assert window['next_offset'] > offset
        offset = window['next_offset']
    assert ''.join(chunks) == content


def test_filesystem_bounds_and_atomic_failures(tmp_path, monkeypatch):
    import pytest
    from lazyllm.tools.agent import file_tool
    target = tmp_path / 'file.txt'
    target.write_text('original')
    target.chmod(0o640)
    with pytest.raises(UnicodeEncodeError):
        write(str(target), '中文', encoding='ascii')
    assert target.read_text() == 'original'
    with pytest.raises(Exception):
        file_tool.edit(str(target), 'missing', 'new')
    assert target.read_text() == 'original'
    def fail_replace(*_):
        raise OSError('simulated replace failure')
    with monkeypatch.context() as patch:
        patch.setattr(file_tool.os, 'replace', fail_replace)
        with pytest.raises(OSError):
            write(str(target), 'new')
    assert target.read_text() == 'original'
    assert not list(tmp_path.glob('.lazyllm-*'))
    file_tool.edit(str(target), 'original', 'changed')
    assert target.read_text() == 'changed'
    assert target.stat().st_mode & 0o777 == 0o640
    for index in range(210):
        (tmp_path / f'{index}.txt').write_text('needle\n' * 3)
    listing = ls(str(tmp_path))
    assert len(listing['entries']) == 200 and listing['truncated']
    matches = grep('needle', str(tmp_path), max_results=1000)
    assert len(matches['results']) == 100 and matches['truncated']
    found = file_tool.glob('*.txt', str(tmp_path), max_results=1000)
    assert len(found['paths']) == 100 and found['truncated']
    monkeypatch.setattr(file_tool.shutil, 'which', lambda _: None)
    with pytest.raises(Exception, match='requires ripgrep'):
        grep('needle', str(tmp_path))


def test_shell_prepared_cwd_and_exclusive_schedule(tmp_path):
    manager = ToolManager(['shell'])
    batch = manager.prepare_tool_calls({'function': {'name': 'shell', 'arguments': {'cmd': 'pwd'}}},
                                       working_directory=str(tmp_path))
    assert batch[0].access.exclusive
    result = manager.execute_prepared(batch, approved_indices=(0,))
    assert result.results[0]['value']['cwd'] == str(tmp_path)
