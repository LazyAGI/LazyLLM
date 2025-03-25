import asyncio
import textwrap

from typing import Any, Callable, Dict, List, Literal, Optional, Union
from mcp import ClientSession
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent

# Union type for non-text content
NonTextContent = Union[ImageContent, EmbeddedResource]


def _get_type_annotation(prop: dict) -> str:
    # Returns the corresponding Python type annotation string based on the JSON Schema property.
    if "enum" in prop:
        enum_values = prop["enum"]
        values_str = ", ".join(repr(val) for val in enum_values)
        return f"Literal[{values_str}]"
    
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "array": "List[Any]"
    }
    
    t = prop.get("type")
    return type_mapping.get(t, "Any")


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[Union[str, List[str]], Optional[List[NonTextContent]]]:
    # Separates and returns the text content and non-text content from a CallToolResult.
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
    def __init__(
        self,
        client: ClientSession,
        allowed_tools: Optional[List[str]] = None,
    ) -> None:
        self._client = client
        self._allowed_tools = allowed_tools or []

    async def fetch_tools(self) -> List[Any]:
        # Asynchronously retrieves the list of tools returned by the MCP client and filters them based on allowed_tools.
        response = await self._client.list_tools()
        tools = getattr(response, "tools", [])
        if self._allowed_tools:
            tools = [tool for tool in tools if tool.name in self._allowed_tools]
        return tools

    def _convert_to_lazyllm_tool(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any]
    ) -> Callable:
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        param_list = []
        for param, prop in properties.items():
            type_annotation = _get_type_annotation(prop)
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
                type_annotation = _get_type_annotation(prop)
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
            "client": self._client,
            "tool_name": tool_name,
            "convert_call_tool_result": _convert_call_tool_result,
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
