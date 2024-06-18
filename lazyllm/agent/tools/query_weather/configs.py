import lazyllm


# 查询天气
# 使用高德开放平台的天气查询API
# API文档: https://lbs.amap.com/api/webservice/guide/api/weatherinfo

WEATHER_KEY = lazyllm.config.getenv("weather_key", str, None)
WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo?"