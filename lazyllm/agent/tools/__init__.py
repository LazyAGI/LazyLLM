from .baseTool import BaseTool
from .toolManager import ToolManager
from .query_weather import QueryWeather
from .web_search import WebSearch

query_weather = QueryWeather()
web_search = WebSearch()

TOOL_LIST = [
    query_weather,
    web_search,
]

default_tool_manager = ToolManager(TOOL_LIST)