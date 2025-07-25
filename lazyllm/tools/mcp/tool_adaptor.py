import inspect
import asyncio

from typing import Any, Callable, Dict, List, Set
from lazyllm import LOG
from lazyllm.thirdparty import mcp

from .utils import run_async_in_new_loop, run_async_in_thread

type_mapping = {
    "string": ("str", str),
    "integer": ("int", int),
    "number": ("float", float),
    "boolean": ("bool", bool),
    "object": ("Dict[str, Any]", Dict[str, Any]),
    "array": ("List[Any]", List[Any])
}


def _generate_docstring(tool_desc: str, properties: Dict[str, Any], required: List[str]) -> str:
    doc_lines = []
    if tool_desc:
        doc_lines.append(tool_desc)
    if not tool_desc or "Args:" not in tool_desc:
        doc_lines.append("")
        doc_lines.append("Args:")
        if not properties:
            doc_lines.append("    No parameters.")

        for param, prop in properties.items():
            json_type = prop.get("type", "Any")
            doc_type = type_mapping.get(json_type, ("Any", Any))[0]
            param_desc = prop.get("description", f"type: {json_type}")
            if param in required:
                doc_lines.append(f"    {param} ({doc_type}): {param_desc}.")
            else:
                doc_lines.append(f"    {param} (Optional[{doc_type}]): {param_desc}.")
    return "\n".join(doc_lines)


def _handle_tool_result(result, tool_name: str) -> str:
    if not result.content or len(result.content) == 0:
        return "No data available for this request."
    try:
        text_contents: List[mcp.types.TextContent] = []
        image_contents: List[mcp.types.ImageContent] = []
        for content in result.content:
            if isinstance(content, mcp.types.TextContent):
                text_contents.append(content.text)
            elif isinstance(content, mcp.types.ImageContent):
                image_contents.append(content.data)
            else:
                LOG.warning(f"Unsupported content type: {type(content)}")
        res_str = "Tool call result:\n"
        if text_contents:
            res_str += "Received text message:\n" + "\n".join(text_contents)
        if image_contents:
            res_str += "Received image message:\n" + "\n".join(
                [f"Image: {image}" for image in image_contents]
            )
        return res_str
    except (IndexError, AttributeError) as e:
        LOG.error(f"Error processing content from MCP tool '{tool_name}': {e!s}")
        return f"Error processing content from MCP tool '{tool_name}': {e!s}"


def generate_lazyllm_tool(client, mcp_tool) -> Callable:
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description
    input_schema = mcp_tool.inputSchema
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Generate docstring
    func_desc = _generate_docstring(tool_desc, properties, required)

    annotations = {}
    defaults: Dict[str, Any] = {}
    func_params = []

    for param, prop in properties.items():
        json_type = prop.get("type", "Any")
        py_type = type_mapping.get(json_type, ("Any", Any))[1]
        annotations[param] = py_type

        if param not in required:
            defaults[param] = prop.get("default", None)
        func_params.append(param)

    # Define the function
    def dynamic_lazyllm_func(**kwargs):
        missing_params: Set[str] = set(required) - set(kwargs.keys())
        if missing_params:
            LOG.warning(f"Missing required parameters: {missing_params}")
            return f"Missing required parameters: {missing_params}"
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                result = run_async_in_thread(client.call_tool, tool_name, kwargs)
            else:
                result = run_async_in_new_loop(client.call_tool, tool_name, kwargs)
        except Exception as e:
            LOG.error(f"Failed to call MCP tool '{tool_name}': {e!s}")
            return f"Failed to call MCP tool '{tool_name}': {e!s}"
        return _handle_tool_result(result, tool_name)

    # Set function attributes
    dynamic_lazyllm_func.__name__ = tool_name
    dynamic_lazyllm_func.__doc__ = func_desc
    dynamic_lazyllm_func.__annotations__ = annotations

    sig = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name=param,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=defaults.get(param, inspect.Parameter.empty),
                annotation=annotations[param],
            )
            for param in func_params
        ]
    )
    dynamic_lazyllm_func.__signature__ = sig

    return dynamic_lazyllm_func
