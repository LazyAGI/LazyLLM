from typing import Literal
import requests

from lazyllm.agent_tmp.configs import WEATHER_URL, WEATHER_KEY
from lazyllm.agent_tmp.tools.base import BaseTool
from lazyllm.agent_tmp.tools.query_weather.utils import CityCodeMatcher


class QueryWeather(BaseTool):

    name = "query_weather"
    
    def __init__(self):
        self._city_code_matcher = CityCodeMatcher()
        if not WEATHER_KEY:
            raise ValueError('If you want to use the tool "QueryWeather", '
                             'please set "LAZYLLM_WEATHER_KEY" in the environment variable, '
                             'or set `lazyllm.agent.configs.WEATHER_KEY = "your key"`. '
                             '\nThe key can be obtained at https://lbs.amap.com/dev/key/app ')
        self._url = WEATHER_URL
        self._key = WEATHER_KEY

    def call(self,
            city_name:str,
            weather_type:Literal["base","all"] = "base",
            **kwargs
        ) -> str:
        """
        查询目标城市当前/未来的天气情况，数据来源是中国气象局

        Args:
            city_name (str): 要查询的城市名，支持中文和拼音，例如"北京"、"上海"、"guangzhou"等
            weather_type (Literal["base","all"]): 气象类型，可选值：base/all，base表示返回实况天气，all表示返回预报天气，默认值查询实况天气
        """
        
        citycode, full_city_name = self._city_code_matcher.get_adcode_and_fullname(city_name)
        if not citycode: return f"没有找到[{city_name}]的天气信息，请确认城市名是否正确"
        weather_type = weather_type if weather_type in ("base","all") else "base"

        response = requests.get(self._url, params={"key":self._key, "city":citycode, "extensions":weather_type, "output":"json"})
        # print(f"response->:{response.json()}\nstatus_code:{response.status_code}")
        if response.status_code != 200: return "查询天气失败，请稍后重试"

        return response.content.decode('utf-8')

query_weather = QueryWeather()