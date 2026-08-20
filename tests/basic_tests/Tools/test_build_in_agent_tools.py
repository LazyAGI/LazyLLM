import os
import tempfile

import pytest

from lazyllm.tools import ToolManager
from lazyllm.tools.agent import ToolPermissionError
from lazyllm.tools.agent.file_tool import (read_file, write_file, list_dir, search_in_files,
                                           move_file, delete_file)
from lazyllm.tools.agent.shell_tool import shell_tool
from lazyllm.tools.agent.download_tool import download_file


class TestFileTool(object):
    def test_file_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'a.txt')
            res = write_file(path, 'hello\nworld', root=tmp)
            assert res['status'] == 'ok'

            res = read_file(path, root=tmp)
            assert res['status'] == 'ok'
            assert 'hello' in res['content']

            res = list_dir(tmp, root=tmp)
            assert res['status'] == 'ok'
            assert 'a.txt' in res['entries']

            res = search_in_files('world', path=tmp, root=tmp)
            assert res['status'] == 'ok'
            assert any(item['path'].endswith('a.txt') for item in res['results'])

            dst = os.path.join(tmp, 'b.txt')
            res = move_file(path, dst, root=tmp, allow_unsafe=True)
            assert res['status'] == 'ok'

            res = delete_file(dst, root=tmp, allow_unsafe=True)
            assert res['status'] == 'ok'


class TestShellTool(object):
    def test_shell_tool(self):
        res = shell_tool('echo hello')
        assert res['status'] == 'ok'
        assert 'hello' in res['stdout']

    def test_shell_tool_needs_approval(self):
        with pytest.raises(ToolPermissionError, match='dangerous token'):
            shell_tool('rm -rf /tmp/does_not_exist')

    def test_shell_tool_permission_failure_is_structured(self):
        manager = ToolManager(['shell_tool'])
        call = {
            'function': {
                'name': 'shell_tool',
                'arguments': {'cmd': 'rm -rf /tmp/does_not_exist'},
            },
        }

        result = manager(call)[0]

        assert result['error']['category'] == 'PERMISSION_ERROR'
        assert result['error']['code'] == 'SHELL_COMMAND_REQUIRES_APPROVAL'


class TestDownloadTool(object):
    def test_download_tool_needs_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, 'a.txt')
            with pytest.raises(ToolPermissionError, match='requires approval'):
                download_file('http://example.com/a.txt', dst, root=tmp)

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
            res = download_file(url, dst, root=tmp, allow_unsafe=True)
            assert res['status'] == 'ok'
            assert res['bytes'] > 0
