import json
from lazyllm.tools.tools import Weather

class TestWeather(object):
    # skip
    def _test_weather(self):
        weather = Weather()
        res = weather('海淀')
        assert res['status_code'] == 200
        content = json.loads(res['content'])
        assert content['station']['city'] == '海淀'
