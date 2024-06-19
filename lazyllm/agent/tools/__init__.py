from .base import BaseTool
from .query_weather import query_weather
from .tool_manager import ToolManager


TOOLS_MAP = {query_weather.name: query_weather}