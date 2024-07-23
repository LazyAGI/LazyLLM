from lazyllm.tools.http_request.http_request import HttpRequest

class TestHTTPRequest(object):

    def test_http_request(self):
        http_request = HttpRequest('get', 'https://httpbin.org/ip', API_Key='', headers={}, params={}, body='')
        r = http_request()
        assert 'status' in r
