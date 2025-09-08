import json
from lazyllm.thirdparty import httpx
from lazyllm.tools.tools import HttpTool
from functools import lru_cache
from typing import Dict, Optional

# assumes the city name is unique
@lru_cache(maxsize=None)
def get_city2code():
    city2code = {}

    res = httpx.get('http://nmc.cn/rest/province')
    provinces = json.loads(res.text)
    for prov in provinces:
        url = f'http://nmc.cn/rest/province/{prov["code"]}'
        res = httpx.get(url)
        cities = json.loads(res.text)
        for c in cities:
            city2code[c['city']] = c['code']

    return city2code


class Weather(HttpTool):
    def __init__(self):
        self._city2code = get_city2code()
        url = 'http://www.nmc.cn/rest/real/{{city_code}}'
        super().__init__(method='GET', url=url)

    def forward(self, city_name: str) -> Optional[Dict]:
        city_code = self._city2code.get(city_name)
        if not city_code:
            return None

        return super().forward(city_code=city_code)
