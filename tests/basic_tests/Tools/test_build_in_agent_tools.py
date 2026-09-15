import os
import tempfile


from lazyllm.tools import ToolManager
from lazyllm.tools.agent.file_tool import (read, write, ls, grep,
                                           move, remove)
from lazyllm.tools.agent.shell_tool import shell_tool
from lazyllm.tools.agent.download_tool import download_file


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
    def test_shell_tool(self):
        res = shell_tool('echo hello')
        assert res['status'] == 'ok'
        assert 'hello' in res['stdout']

    def test_shell_tool_needs_approval(self):
        manager = ToolManager(['shell_tool'])
        prepared = manager.prepare_tool_calls({
            'function': {'name': 'shell_tool', 'arguments': {'cmd': 'echo approved'}},
        })
        result = manager.execute_prepared(prepared)
        assert result.results[0]['needs_approval'] is True
        assert result.records[0].reason == 'approval_required'

    def test_shell_tool_executes_after_host_approval(self):
        manager = ToolManager(['shell_tool'])
        prepared = manager.prepare_tool_calls({
            'function': {'name': 'shell_tool', 'arguments': {'cmd': 'echo approved'}},
        })
        result = manager.execute_prepared(prepared, approved_indices=(0,))
        assert result.results[0]['ok'] is True
        assert 'approved' in result.results[0]['value']['stdout']


class TestDownloadTool(object):
    def test_download_tool_needs_approval(self, monkeypatch):
        requested = []
        monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: requested.append(args))
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, 'a.txt')
            manager = ToolManager([download_file])
            prepared = manager.prepare_tool_calls({
                'function': {'name': 'download_file', 'arguments': {
                    'url': 'http://example.com/a.txt', 'dst': dst, 'root': tmp,
                }},
            })
            result = manager.execute_prepared(prepared)
            assert result.results[0]['needs_approval'] is True
            assert requested == []

    def test_download_tool(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def read():
                return b'hello download'

        monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: FakeResponse())
        with tempfile.TemporaryDirectory() as tmp:
            url = 'http://example.com/payload.txt'
            dst = os.path.join(tmp, 'out.txt')
            res = download_file(url, dst, root=tmp)
            assert res['status'] == 'ok'
            assert res['bytes'] > 0
