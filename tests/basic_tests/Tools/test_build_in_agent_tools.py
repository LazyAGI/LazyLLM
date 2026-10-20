import io
import os
import tempfile
from contextlib import nullcontext


from lazyllm.tools import AuthorizationDecision, ToolManager
from lazyllm.tools.agent import ToolExecutionError
from lazyllm.tools.agent.file_tool import (read, write, ls, grep,
                                           move, remove)
from lazyllm.tools.agent.shell_tool import shell


class TestFileTool(object):
    def test_file_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'a.txt')
            res = write(path, 'hello\nworld')
            assert res['status'] == 'ok'

            res = read(path)
            assert res['status'] == 'ok'
            assert 'hello' in res['content']

            res = ls(tmp)
            assert res['status'] == 'ok'
            assert 'a.txt' in res['entries']

            res = grep('world', path=tmp)
            assert res['status'] == 'ok'
            assert any(item['path'].endswith('a.txt') for item in res['results'])

            dst = os.path.join(tmp, 'b.txt')
            res = move(path, dst)
            assert res['status'] == 'ok'

            res = remove(dst)
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
        }, authorization_policy=lambda _: AuthorizationDecision.ASK)
        result = manager.execute_prepared(prepared)
        assert result.results[0]['needs_approval'] is True
        assert result.records[0].reason == 'approval_required'

    def test_shell_executes_after_host_approval(self):
        manager = ToolManager(['shell'])
        prepared = manager.prepare_tool_calls({
            'function': {'name': 'shell', 'arguments': {'cmd': 'echo approved'}},
        }, authorization_policy=lambda _: AuthorizationDecision.ASK)
        result = manager.execute_prepared(prepared, approved_indices=(0,))
        assert result.results[0]['ok'] is True
        assert 'approved' in result.results[0]['value']['stdout']


def test_physical_lines_long_unicode_and_grep_read_pair(tmp_path):
    source = tmp_path / 'large.txt'
    source.write_bytes(('中' * 12000 + '\r\nshort needle\r\nlast').encode())
    first = read(str(source), limit=1)
    assert len(first['content'].encode()) <= 65536
    assert first['line_unit'] == 'physical'
    assert first['truncated_lines'] == [1]
    assert first['next_offset'] == 2 and not first['eof']
    found = grep('needle', str(source))['results'][0]
    assert found['line'] == '2'
    window = read(str(source), offset=int(found['line']))
    assert window['content'] == 'short needle\nlast'
    assert window['eof'] and not window['truncated']
    assert window['end_line'] == 3
    assert read(str(source), offset=4)['content'] == ''
    source.write_text('中' * 3000)
    last = read(str(source))
    assert last['eof'] and last['truncated'] and last['truncated_lines'] == [1]
    assert last['next_offset'] is None and len(last['content']) == 2000


def test_read_stops_before_next_line_exceeding_budget(tmp_path):
    source = tmp_path / 'budget.txt'
    source.write_text(('x' * 1999 + '\n') * 32 + '中' * 1000 + '\nend')
    first = read(str(source))
    assert first['end_line'] == 32 and first['next_offset'] == 33
    assert first['truncated_lines'] == []
    second = read(str(source), offset=33)
    assert second['content'] == '中' * 1000 + '\nend'
    assert second['eof']


def test_read_limit_only_probes_next_line(monkeypatch):
    from lazyllm.tools.agent import file_tool

    stream = io.StringIO('header\n' + 'x' * (8 * 1024 * 1024) + '\n')
    monkeypatch.setattr(file_tool, '_open_text', lambda *args, **kwargs: nullcontext(stream))
    result = read('/tmp/read-limit.txt', limit=1)
    assert result['content'] == 'header\n'
    assert result['end_line'] == 1 and result['next_offset'] == 2
    assert not result['eof'] and result['truncated']
    assert stream.tell() <= len('header\n') + 1


def test_read_limit_at_eof(tmp_path):
    source = tmp_path / 'one-line.txt'
    for content in ('header\n', 'header'):
        source.write_text(content)
        result = read(str(source), limit=1)
        assert result['content'] == content
        assert result['eof'] and not result['truncated']
        assert result['next_offset'] is None and result['end_line'] == 1


def test_grep_snippet_shows_late_match_and_skips_oversized_record(tmp_path):
    source = tmp_path / 'matches.txt'
    source.write_text('a' * 100000 + '中文needle' + 'z' * 3000 + '\n'
                      + 'x' * (17 * 1024 * 1024) + 'needle\n'
                      + 'later needle\n')
    result = grep('needle', str(source))
    assert [r['line'] for r in result['results']] == ['1', '3']
    assert all('needle' in r['text'] and len(r['text']) <= 1000 for r in result['results'])
    assert result['results'][0]['snippet_truncated']
    assert result['skipped_records'] == 1 and result['search_incomplete']
    assert not result['output_limit_reached']
    huge = grep('a{3000}', str(source))
    assert len(huge['results'][0]['text']) <= 1000
    assert 'match clipped' in huge['results'][0]['text']


def test_filesystem_bounds_and_atomic_failures(tmp_path, monkeypatch):
    import pytest
    from lazyllm.tools.agent import file_tool
    target = tmp_path / 'file.txt'
    target.write_text('original')
    target.chmod(0o640)
    with pytest.raises(UnicodeEncodeError):
        file_tool._atomic_write(str(target), '中文', 'ascii')
    assert target.read_text() == 'original'
    with pytest.raises(ToolExecutionError, match='Expected 1 matches, found 0'):
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
    matches = grep('needle', str(tmp_path))
    assert len(matches['results']) == 100 and matches['truncated']
    found = file_tool.glob('*.txt', str(tmp_path))
    assert len(found['paths']) == 100 and found['truncated']
    monkeypatch.setattr(file_tool.shutil, 'which', lambda _: None)
    with pytest.raises(Exception, match='requires ripgrep'):
        grep('needle', str(tmp_path))


def test_shell_prepared_cwd_and_exclusive_schedule(tmp_path):
    manager = ToolManager(['shell'])
    batch = manager.prepare_tool_calls({'function': {'name': 'shell', 'arguments': {'cmd': 'pwd'}}},
                                       working_directory=str(tmp_path))
    assert batch[0].access.exclusive
    result = manager.execute_prepared(batch)
    assert result.results[0]['value']['cwd'] == str(tmp_path)


def test_edit_byte_limit_and_bounded_read(tmp_path, monkeypatch):
    import pytest
    from contextlib import contextmanager
    from lazyllm.tools.agent import file_tool

    target = tmp_path / 'text'
    original = '中' * 21845 + 'x'
    target.write_text(original, encoding='utf-8')
    assert target.stat().st_size == 65536
    opened = file_tool._open_text
    reads = []

    @contextmanager
    def bounded(*args, **kwargs):
        with opened(*args, **kwargs) as stream:
            class Reader:
                def fileno(self):
                    return stream.fileno()

                def read(self, size=-1):
                    assert 0 < size <= 65537
                    reads.append(size)
                    return stream.read(size)
            yield Reader()

    monkeypatch.setattr(file_tool, '_open_text', bounded)
    file_tool.edit(str(target), 'x', 'y')
    assert reads and target.read_text() == original[:-1] + 'y'
    target.write_bytes(target.read_bytes() + b'z')
    before = target.read_bytes()
    reads.clear()
    with pytest.raises(ToolExecutionError, match='65536'):
        file_tool.edit(str(target), 'y', 'x')
    assert target.read_bytes() == before and not reads


def test_append_streams_bytes_and_preserves_original_on_failure(tmp_path, monkeypatch):
    import pytest
    from contextlib import contextmanager
    from lazyllm.tools.agent import file_tool

    target = tmp_path / 'large'
    with target.open('wb') as stream:
        stream.write(b'\xff')
        stream.truncate(100 * 1024 * 1024)
    target.chmod(0o640)
    original_size = target.stat().st_size
    opened = file_tool._open_text
    reads = []

    @contextmanager
    def bounded(*args, **kwargs):
        with opened(*args, **kwargs) as stream:
            class Reader:
                def read(self, size=-1):
                    assert 0 < size <= 1024 * 1024
                    reads.append(size)
                    return stream.read(size)
            yield Reader()

    monkeypatch.setattr(file_tool, '_open_text', bounded)
    size = file_tool._atomic_write(str(target), '中文', 'utf-8', mode='append')
    assert size == 6 and len(reads) > 1
    assert target.stat().st_size == original_size + 6
    assert target.stat().st_mode & 0o777 == 0o640
    with target.open('rb') as stream:
        assert stream.read(1) == b'\xff'
        stream.seek(-6, os.SEEK_END)
        assert stream.read() == '中文'.encode()

    def fail_replace(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(file_tool.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replace failed'):
        file_tool._atomic_write(str(target), 'lost', 'utf-8', mode='append')
    assert target.stat().st_size == original_size + 6
    assert not list(tmp_path.glob('.lazyllm-*'))
    link = tmp_path / 'link'
    link.symlink_to(target)
    with pytest.raises(ToolExecutionError, match='regular file'):
        file_tool._atomic_write(str(link), 'x', 'utf-8', mode='append')


def test_windows_cross_drive_roots_are_tool_errors(monkeypatch):
    import ntpath
    import pytest
    from lazyllm.tools.agent import file_tool
    monkeypatch.setattr(file_tool, '_resolve_path', lambda path: path)
    monkeypatch.setattr(file_tool.os.path, 'commonpath', ntpath.commonpath)
    file_tool._check_root(r'C:\workspace\file', r'C:\workspace')
    for path in [r'D:\file', r'\\server\share\file']:
        with pytest.raises(ToolExecutionError, match='outside the allowed root'):
            file_tool._check_root(path, r'C:\workspace')


def test_shell_bounded_head_tail_and_byte_counts(tmp_path):
    import shlex
    import sys
    script = tmp_path / 'output.py'
    script.write_text("import os\nfor fd in (1, 2):\n    os.write(fd, b'H' * 65536 + b'M' * 300000 + b'T' * 65536)\n")
    result = shell(f'{shlex.quote(sys.executable)} {shlex.quote(str(script))}')
    for name in ('stdout', 'stderr'):
        assert result[name + '_bytes'] == 431072
        assert result[name + '_truncated']
        assert result[name].startswith('H' * 65536)
        assert result[name].endswith('T' * 65536)
        assert len(result[name]) < 131200
    result = shell('printf hello; printf error >&2; exit 7')
    assert result['stdout'] == 'hello' and result['stderr'] == 'error'
    assert result['exit_code'] == 7
    assert result['stdout_bytes'] == 5 and not result['stdout_truncated']


def test_shell_timeout_includes_inherited_pipes(tmp_path):
    import pytest
    import time
    for command in ['sleep 10', 'sleep 10 & exit 0']:
        started = time.monotonic()
        with pytest.raises(ToolExecutionError, match='timed out'):
            shell(command, timeout=0.2)
        assert time.monotonic() - started < 3
