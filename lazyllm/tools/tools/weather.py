import json
import httpx
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
    """
Create a tool for querying weather.


Examples:
    
    from lazyllm.tools.tools import Weather
    
    weather = Weather()
    """
    def __init__(self):
        self._city2code = get_city2code()
        url = 'http://www.nmc.cn/rest/real/{{city_code}}'
        super().__init__(method='GET', url=url)

    def forward(self, city_name: str) -> Optional[Dict]:
        """
Query the weather of a specific city. The minimum input scope for cities is at the prefecture level, and for municipalities, it is at the district level. The input city or district name should not include the suffix "市" (city) or "区" (district). Refer to the examples below.

Args:
    city_name (str): The name of the city for which weather information is needed.


Examples:
    
    from lazyllm.tools.tools import Weather
    
    weather = Weather()
    res = weather('海淀')
    """
        city_code = self._city2code.get(city_name)
        if not city_code:
            return None

        return super().forward(city_code=city_code)
