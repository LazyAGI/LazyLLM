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
