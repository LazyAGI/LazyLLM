import json
import httpx
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request import HttpRequest
from functools import lru_cache
from typing import Dict

@lru_cache(maxsize=None)
def get_city_list():
    province_list = []

    res = httpx.get('http://nmc.cn/rest/province')
    provinces = json.loads(res.text)
    for prov in provinces:
        city_list = []

        url = f'http://nmc.cn/rest/province/{prov["code"]}'
        res = httpx.get(url)
        cities = json.loads(res.text)
        for c in cities:
            city_list.append({
                'code': c['code'],
                'name': c['city'],
            })

        province_list.append({
            'code': prov['code'],
            'name': prov['name'],
            'city_list': city_list,
        })

    return province_list


class Weather(ModuleBase):
    def __init__(self):
        super().__init__()


    @staticmethod
    def get_city_list():
        return get_city_list()


    def forward(self, cityid) -> Dict:
        url = f'http://www.nmc.cn/rest/real/{cityid}'
        req = HttpRequest(method='GET', url=url, params=None,
                          api_key='', headers=None, body=None)
        return req()


if __name__ == '__main__':
    weather = Weather()
    res = weather('fElIR')
