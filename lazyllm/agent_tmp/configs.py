
# 最大连续调用工具次数
MAX_CONSECUTIVE_TOOL_CALL_NUM = 5


# 查询天气
# 使用高德开放平台的天气查询API
# API文档: https://lbs.amap.com/api/webservice/guide/api/weatherinfo

import lazyllm
lazyllm.config.add("weather_key", str, "", "WEATHER_KEY")

WEATHER_KEY = lazyllm.config["weather_key"]
WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo?"