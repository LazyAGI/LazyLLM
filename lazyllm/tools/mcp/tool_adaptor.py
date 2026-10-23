import inspect
import copy
import json
import asyncio
import re

from typing import Any, Callable, Dict, List, Set
from lazyllm import LOG
from lazyllm.thirdparty import mcp
from ..agent.toolsManager import fc_register

from .utils import run_async_in_new_loop, run_async_in_thread

type_mapping = {
    'string': ('str', str),
    'integer': ('int', int),
    'number': ('float', float),
    'boolean': ('bool', bool),
    'object': ('Dict[str, Any]', Dict[str, Any]),
    'array': ('List[Any]', List[Any])
}


def _generate_docstring(tool_desc: str, properties: Dict[str, Any], required: List[str]) -> str:
    doc_lines = []
    if tool_desc:
        doc_lines.append(tool_desc)
    if not tool_desc or 'Args:' not in tool_desc:
        doc_lines.append('')
        doc_lines.append('Args:')
        if not properties:
            doc_lines.append('    No parameters.')

        for param, prop in properties.items():
            json_type = prop.get('type', 'Any')
            # JSON Schema also accepts a type array, e.g. ['string', 'null'].
            # This is only a doc hint; the original schema remains authoritative.
            doc_type = type_mapping.get(json_type, ('Any', Any))[0] if isinstance(json_type, str) else 'Any'
            param_desc = prop.get('description', f'type: {json_type}')
            if param in required:
                doc_lines.append(f'    {param} ({doc_type}): {param_desc}.')
            else:
                doc_lines.append(f'    {param} (Optional[{doc_type}]): {param_desc}.')
    return '\n'.join(doc_lines)


def _handle_tool_result(result, tool_name: str, result_mode: str = 'text') -> str:
    if result_mode == 'structured':
        # Keep the protocol envelope intact even for errors with no text fallback.
        content = [item.model_dump(mode='json', by_alias=True, exclude_none=True)
                   for item in (result.content or [])]
        structured = getattr(result, 'structuredContent', None)
        if structured is None:
            structured = getattr(result, 'structured_content', None)
        is_error = getattr(result, 'isError', getattr(result, 'is_error', False))
        return json.dumps({'content': content, 'structuredContent': structured,
                           'isError': bool(is_error)}, ensure_ascii=False)
    if not result.content or len(result.content) == 0:
        return 'No data available for this request.'
    try:
        text_contents: List[mcp.types.TextContent] = []  # noqa NID003
        image_contents: List[mcp.types.ImageContent] = []  # noqa NID003
        for content in result.content:
            if isinstance(content, mcp.types.TextContent):
                text_contents.append(content.text)
            elif isinstance(content, mcp.types.ImageContent):
                image_contents.append(content.data)
            else:
                LOG.warning(f'Unsupported content type: {type(content)}')
        res_str = 'Tool call result:\n'
        if text_contents:
            res_str += 'Received text message:\n' + '\n'.join(text_contents)
        if image_contents:
            res_str += 'Received image message:\n' + '\n'.join(
                [f'Image: {image}' for image in image_contents]
            )
        return res_str
    except (IndexError, AttributeError) as e:
        LOG.error(f'Error processing content from MCP tool "{tool_name}": {e!s}')
        return f'Error processing content from MCP tool "{tool_name}": {e!s}'


class MCPToolNameError(ValueError):
    pass


def _validate_mcp_tool_names(tools):
    names = {}
    for tool in tools:
        name = tool.__name__
        wire_name = tool.__mcp_tool_name__
        if name in names:
            raise MCPToolNameError(f'MCP tool name collision: {names[name]} and {wire_name} map to {name}')
        names[name] = wire_name
    return tools


def _model_tool_name(server_name: str, tool_name: str, tool_name_prefix: str = '') -> str:
    if not isinstance(tool_name, str) or not tool_name:
        raise MCPToolNameError('MCP tool name must not be empty')
    if not server_name:
        name = re.sub(r'\W', '_', tool_name)
        return '_' + name if name[:1].isdigit() else name
    server = re.sub(r'[^a-z0-9_]+', '_', server_name.lower()).strip('_')
    tool = re.sub(r'[^A-Za-z0-9_]+', '_', tool_name.removeprefix(tool_name_prefix)).strip('_')
    name = f'mcp_{server}_{tool}'
    if not server or not tool or len(name) > 64:
        raise MCPToolNameError(f'MCP tool name is empty or exceeds 64 characters: {name}')
    return name


def generate_lazyllm_tool(client, mcp_tool) -> Callable:
    tool_name = mcp_tool.name
    exposed_tool_name = _model_tool_name(
        getattr(client, 'server_name', ''), tool_name, getattr(client, 'tool_name_prefix', ''))
    tool_desc = mcp_tool.description
    # MCP's Pydantic models used the JSON alias ``inputSchema`` in older
    # releases and the Pythonic ``input_schema`` attribute in newer ones.
    input_schema = getattr(mcp_tool, 'inputSchema', None)
    if input_schema is None:
        input_schema = getattr(mcp_tool, 'input_schema', {})
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])

    # Generate docstring
    func_desc = _generate_docstring(tool_desc, properties, required)

    annotations = {}
    defaults: Dict[str, Any] = {}
    func_params = []

    for param, prop in properties.items():
        # The MCP server validates its JSON Schema; Python coercion would lose
        # union constraints, explicit nulls, or exact identifier values.
        annotations[param] = Any

        if param not in required:
            defaults[param] = prop.get('default', None)
        func_params.append(param)

    # Define the function
    def dynamic_lazyllm_func(**kwargs):
        missing_params: Set[str] = set(required) - set(kwargs.keys())
        if missing_params:
            LOG.warning(f'Missing required parameters: {missing_params}')
            return f'Missing required parameters: {missing_params}'
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
            if getattr(client, 'result_mode', 'text') == 'structured':
                error_type = type(e).__name__
                LOG.error(f'Failed to call MCP tool "{tool_name}": {error_type}')
                return json.dumps({'content': [], 'structuredContent': {
                    'code': 'MCP_CALL_FAILED', 'error_type': error_type}, 'isError': True})
            LOG.error(f'Failed to call MCP tool "{tool_name}": {e!s}')
            return f'Failed to call MCP tool "{tool_name}": {e!s}'
        return _handle_tool_result(result, tool_name, getattr(client, 'result_mode', 'text'))

    # Set function attributes
    # LazyLLM's registry interprets dots as registry-group separators. MCP
    # servers commonly namespace tools with dots, so expose a valid Python
    # identifier while retaining the original name in the call closure.
    dynamic_lazyllm_func.__name__ = exposed_tool_name
    dynamic_lazyllm_func.__mcp_tool_name__ = tool_name
    dynamic_lazyllm_func.__mcp_input_schema__ = copy.deepcopy(input_schema)
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

    server_id = getattr(client, 'server_id', '')
    metadata = {'tool_origin': server_id,
                'tool_identity': json.dumps([server_id, tool_name], ensure_ascii=False)} if server_id else {}
    return fc_register(tool_source='mcp', **metadata)(dynamic_lazyllm_func)
