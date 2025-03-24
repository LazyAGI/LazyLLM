import asyncio
import textwrap

from typing import Any, Callable, Dict, List, Literal, Optional, Union
from mcp import ClientSession
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent

# 定义非文本内容的联合类型
NonTextContent = Union[ImageContent, EmbeddedResource]


def get_type_annotation(prop: dict) -> str:
    """
    Returns the corresponding Python type annotation string based on the JSON Schema property.

    Args:
        prop: A dictionary describing the JSON Schema property.

    Returns:
        The type annotation string, such as "str", "int", "Literal['a', 'b']" etc.
    """
    if "enum" in prop:
        enum_values = prop["enum"]
        values_str = ", ".join(repr(val) for val in enum_values)
        return f"Literal[{values_str}]"
    
    t = prop.get("type")
    if t == "string":
        return "str"
    elif t == "integer":
        return "int"
    elif t == "number":
        return "float"
    elif t == "boolean":
        return "bool"
    elif t == "object":
        return "Dict[str, Any]"
    elif t == "array":
        return "List[Any]"
    else:
        return "Any"


def convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[Union[str, List[str]], Optional[List[NonTextContent]]]:
    """
    Separates and returns the text content and non-text content from a CallToolResult.

    Args:
        call_tool_result: The result returned after calling the tool.

    Returns:
        A tuple (tool_content, non_text_contents), where tool_content is either a single string or a list of strings,
        and non_text_contents is a list of non-text content; if there is no non-text content, returns None.
    """
    text_contents: List[TextContent] = []
    non_text_contents: List[NonTextContent] = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: Union[str, List[str]] = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    return tool_content, non_text_contents or None


class McpToolAdaptor:
    """
    Converts tools fetched from the MCP client for use by the LazyLLM agent.
    """

    def __init__(
        self,
        client: ClientSession,
        allowed_tools: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        self.allowed_tools = allowed_tools or []

    async def fetch_tools(self) -> List[Any]:
        """
        Asynchronously retrieves the list of tools returned by the MCP client and filters them based on allowed_tools.
        """
        response = await self.client.list_tools()
        tools = getattr(response, "tools", [])
        if self.allowed_tools:
            tools = [tool for tool in tools if tool.name in self.allowed_tools]
        return tools

    def _convert_to_lazyllm_tool(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any]
    ) -> Callable:
        """
        Converts a tool into a function. The generated function dynamically constructs its parameter signature based on the tool's JSON Schema and automatically calls client.call_tool when invoked.

        Args:
            tool_name: The tool name (also used as the generated function's name, must be a valid identifier).
            tool_description: The tool description.
            input_schema: The JSON Schema for the tool's input parameters.

        Returns:
            A callable function that packages the input parameters into a dictionary, calls client.call_tool, and transforms the result.
        """
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        param_list = []
        for param, prop in properties.items():
            type_annotation = get_type_annotation(prop)
            if param in required:
                param_list.append(f"{param}: {type_annotation}")
            else:
                if "default" in prop:
                    default_val = repr(prop["default"])
                elif "enum" in prop and prop["enum"]:
                    default_val = repr(prop["enum"][0])
                else:
                    default_val = "None"
                param_list.append(f"{param}: {type_annotation} = {default_val}")
        param_str = ", ".join(param_list)

        arg_items = ", ".join([f"'{param}': {param}" for param in properties.keys()])

        doc_lines = []
        if tool_description:
            doc_lines.append(tool_description)
        if "Args:" not in tool_description and "参数" not in tool_description:
            doc_lines.append("")
            doc_lines.append("Args:")
            if not properties:
                doc_lines.append("    No parameters.")
            for param, prop in properties.items():
                type_annotation = get_type_annotation(prop)
                param_desc = prop.get("description", f"type: {prop.get('type', 'Any')}")
                req_str = "required" if param in required else "optional"
                doc_lines.append(f"    {param} ({type_annotation}): {param_desc} ({req_str}).")
        full_docstring = "\n".join(doc_lines)

        func_code = textwrap.dedent(f"""\
def {tool_name}({param_str}):
    \"\"\"
    {full_docstring}
    \"\"\"
    import asyncio
    arguments = {{{arg_items}}}
    res = asyncio.run(client.call_tool(tool_name, arguments))
    return convert_call_tool_result(res)
""")

        local_ns: Dict[str, Any] = {
            "Dict": Dict,
            "Any": Any,
            "List": List,
            "Literal": Literal,
            "client": self.client,
            "tool_name": tool_name,
            "convert_call_tool_result": convert_call_tool_result,
        }
        exec(func_code, local_ns)
        generated_func = local_ns[tool_name]
        return generated_func

    async def atool_list(self) -> List[Callable]:
        tools_list = await self.fetch_tools()
        tools = []
        for tool in tools_list:
            fn = self._convert_to_lazyllm_tool(tool.name, tool.description, tool.inputSchema)
            tools.append(fn)
        return tools

    def tool_list(self) -> List[Callable]:
        """
        sync method of atool_list
        """
        return patch_sync(self.atool_list)()


def patch_sync(func_async: Callable) -> Callable:
    """
    Wraps an asynchronous function into a synchronous function. If called in an asynchronous context, it raises an exception 
    advising to use the asynchronous interface.

    Args:
        func_async: An asynchronous function.

    Returns:
        A function wrapped for synchronous invocation.
    """
    def patched_sync(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "In an asynchronous environment, synchronous calls are not supported. "
                "Please use the asynchronous interface (e.g., atool_list) instead."
            )
        return asyncio.run(func_async(*args, **kwargs))
    return patched_sync
