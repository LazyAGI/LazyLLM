from .base import BaseTool
from .query_weather import query_weather



TOOLS_MAP = {query_weather.name: query_weather}