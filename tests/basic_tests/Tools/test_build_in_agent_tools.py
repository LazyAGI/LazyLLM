import os
import tempfile
import threading
import http.server
import socketserver
from functools import partial

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
        res = shell_tool('rm -rf /tmp/does_not_exist')
        assert res['status'] == 'needs_approval'


class TestDownloadTool(object):
    def test_download_tool_needs_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, 'a.txt')
            res = download_file('http://example.com/a.txt', dst, root=tmp)
            assert res['status'] == 'needs_approval'

    def test_download_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, 'payload.txt')
            with open(src_path, 'w', encoding='utf-8') as f:
                f.write('hello download')

            handler = partial(http.server.SimpleHTTPRequestHandler, directory=tmp)
            httpd = socketserver.TCPServer(('127.0.0.1', 0), handler)
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{port}/payload.txt'
                dst = os.path.join(tmp, 'out.txt')
                res = download_file(url, dst, root=tmp, allow_unsafe=True)
                assert res['status'] == 'ok'
                assert res['bytes'] > 0
            finally:
                httpd.shutdown()
                httpd.server_close()
