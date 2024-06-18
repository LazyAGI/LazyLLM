from typing import List, Dict, Any, Union
import traceback
import json
import time
try:
    import json5
except ImportError:
    raise ImportError("Please install json5 using `pip install json5`. ")

from lazyllm import LOG as logger
from .baseTool import BaseTool


class ToolManager(object):
    def __init__(self, tools:List[BaseTool]):
        super().__init__()
        tools = [self._check_tool(tool) for tool in tools]
        self._tools_map:Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self._tools_desc_map:Dict[str, Dict] = {tool.name: tool.get_description() for tool in tools}

    def _check_tool(self, tool:BaseTool) -> BaseTool:
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Tool {tool} is not a subclass of BaseTool")
        if hasattr(tool, "call") and not callable(tool.call):
            raise ValueError(f"Tool {tool} does not have a callable 'call' method")
        if not tool.call.__doc__:
            raise ValueError(f"Tool {tool} does not have a docstring for its 'call' method")
        if hasattr(tool, "name"):
             tool.name = tool.__class__.__name__.lower()
        return tool

    def register_tool(self, override=False):
        def decorator(cls):
            tool_name = cls.name if hasattr(cls, "name") else cls.__name__.lower()
            if tool_name in self._tools_map:
                if override:
                    logger.warning(f'Tool {tool_name} is already registered, override it.')
                else:
                    raise ValueError(f'Tool {tool_name} is already registered.')

            cls.name = tool_name
            self._tools_map[tool_name] = cls
            self._tools_desc_map[tool_name] = cls.get_description()
            return cls
        return decorator
    
    def pop(self, tool_name:str, default=None) -> BaseTool:
        """
        Remove the tool with the given name from the manager.

        Args:
            tool_name (str): The name of the tool to remove.
            default (Any, optional): The default value to return if the tool is not found. Defaults to None.
        Returns:
            BaseTool: The removed tool.
        """
        return self._tools_map.pop(tool_name, default)
    
    def pop(self, tool_name:str) -> BaseTool:
        if tool_name not in self._tools_map:
            raise KeyError(f"Tool {tool_name} is not registered")
        return self._tools_map.pop(tool_name)

    def get(self, tool_name:str, default=None) -> Union[BaseTool, None]:
        """
        Get the tool with the given name from the manager.

        Args:
            tool_name (str): The name of the tool to get.
            default (Any, optional): The default value to return if the tool is not found. Defaults to None.
            
        Returns:
            Union[BaseTool, None]: The tool with the given name.
        """
        return self._tools_map.get(tool_name, default)
    
    def get(self, tool_name:str) -> BaseTool:
        if tool_name not in self._tools_map:
            raise KeyError(f"Tool {tool_name} is not registered")
        return self._tools_map[tool_name]
        

    @property
    def all_tools(self) -> List[str]:
        return list(self._tools_map.keys())
    
    @property
    def all_tools_description(self) -> List[Dict]:
        return list(self._tools_desc_map.values())
    
    def _call_tool(self, tool_name:str, tool_args:Dict[str, Any], **kwargs) -> Union[str, None]:
        """
        Call the tool with the given name.

        Args:
            tool_name (str): The name of the tool to call.
            tool_args (Dict[str, Any]): The arguments to pass to the tool.
        Returns:
            str: The output of the tool.
        """
        start_time = time.time()
        if tool_name not in self._tools_map:
            raise ValueError(f"Tool {tool_name} is not registered")
        try:
            tool_args = json5.loads(tool_args) if isinstance(tool_args, str) else tool_args
            intersection_keys = tool_args.keys() & kwargs.keys()
            poped_args = {k:tool_args.pop(k) for k in intersection_keys}
            if poped_args:
                logger.debug(f"The following args of tool {tool_name} are overridden: {poped_args}")
            tool_res = self._tools_map[tool_name](**tool_args, **kwargs)
            logger.debug(f"Tool {tool_name} takes {time.time() - start_time:.2f}s.")
            return tool_res if isinstance(tool_res, str) else json.dumps(tool_res, ensure_ascii=False, indent=4)
        except:
            logger.error(f"Error calling tool {tool_name}, args: {tool_args}, kwargs: {kwargs}.")
            logger.debug(traceback.format_exc())