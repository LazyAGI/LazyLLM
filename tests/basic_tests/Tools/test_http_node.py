from lazyllm.thirdparty import httpx
from lazyllm.tools.http_request.http_request import HttpRequest


class TestHTTPRequest(object):

    def test_http_request(self, monkeypatch):
        mock_resp = httpx.Response(200, json={'origin': '127.0.0.1'})
        monkeypatch.setattr(
            'lazyllm.tools.http_request.http_request.httpx.request',
            lambda **kwargs: mock_resp,
        )
        http_request = HttpRequest('get', 'https://httpbin.org/ip', api_key='', headers={}, params={}, body='')
        r = http_request()
        assert r['status_code'] == 200
        assert 'origin' in r['content']
