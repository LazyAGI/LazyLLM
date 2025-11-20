from lazyllm.tools.http_request.http_request import HttpRequest

class TestHTTPRequest(object):

    def test_http_request(self):
        http_request = HttpRequest('get', 'https://httpbin.org/ip', api_key='', headers={}, params={}, body='')
        r = http_request()
        assert r['status_code'] == 200
        assert 'origin' in r['content']
