import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, ModelFailureCode


class ClaudeChat(OnlineChatModuleBase):
    # Anthropic native Messages API (/v1/messages).
    # Differs from OpenAI: x-api-key header, system as top-level field,
    # max_tokens required, and SSE event types for streaming.

    _ANTHROPIC_VERSION = '2023-06-01'
    _DEFAULT_MAX_TOKENS = 4096
    _message_format = 'anthropic'
    PROVIDER_NAME = 'anthropic'
    RESPONSE_PROFILE = OnlineChatModuleBase.RESPONSE_PROFILE.extend(type_map={
        'authentication_error': ModelFailureCode.AUTHENTICATION_FAILED,
        'permission_error': ModelFailureCode.PERMISSION_DENIED,
        'not_found_error': ModelFailureCode.NOT_FOUND,
        'rate_limit_error': ModelFailureCode.RATE_LIMITED,
        'overloaded_error': ModelFailureCode.PROVIDER_OVERLOADED,
        'api_error': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    })
    _FINISH_REASONS = {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'tool_calls',
                       'max_tokens': 'length', 'refusal': 'content_filter'}

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.anthropic.com/v1/'
        model = model or 'claude-3-5-sonnet-20241022'
        super().__init__(api_key=api_key or self._default_api_key(),
                         base_url=base_url, model_name=model, stream=stream,
                         return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are Claude, an AI assistant made by Anthropic. You are helpful, harmless, and honest.'

    def _get_chat_url(self, url):
        if url.rstrip('/').endswith('v1/messages'):
            return url
        base = url.rstrip('/')
        if base.endswith('/v1'):
            return base + '/messages'
        return urljoin(url if url.endswith('/') else url + '/', 'v1/messages')

    @staticmethod
    def _get_header(api_key: str) -> dict:
        header = {'Content-Type': 'application/json',
                  'anthropic-version': ClaudeChat._ANTHROPIC_VERSION}
        if api_key:
            header['x-api-key'] = api_key
        return header

    @staticmethod
    def _convert_tools(tools: List[Dict]) -> List[Dict]:
        # Convert OpenAI-format tools to Anthropic format.
        # OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
        # Anthropic: [{"name": ..., "description": ..., "input_schema": {...}}]
        result = []
        for tool in tools:
            fn = tool.get('function', tool)
            result.append({
                'name': fn['name'],
                'description': fn.get('description', ''),
                'input_schema': fn.get('parameters', fn.get('input_schema', {'type': 'object', 'properties': {}})),
            })
        return result

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Agents build their history in OpenAI format, but the Messages API has no `tool` role and rejects
        # extra message fields (tool_calls, reasoning_content, ...), so tool loops fail with 400 after the
        # first call. Map tool_calls to `tool_use` blocks and group consecutive tool results into one user
        # message of `tool_result` blocks, as Anthropic expects.
        result = []
        for msg in messages:
            role = msg.get('role')
            if role == 'tool':
                content = msg.get('content')
                if content is None:
                    content = ''
                elif not isinstance(content, (str, list)):
                    content = json.dumps(content, ensure_ascii=False)
                block = {'type': 'tool_result', 'tool_use_id': msg.get('tool_call_id'), 'content': content}
                last = result[-1] if result else None
                if (last and last['role'] == 'user' and isinstance(last['content'], list)
                        and last['content'] and all(b.get('type') == 'tool_result' for b in last['content'])):
                    last['content'].append(block)
                else:
                    result.append({'role': 'user', 'content': [block]})
                continue
            content = msg.get('content')
            if role == 'assistant' and msg.get('tool_calls'):
                if isinstance(content, list):
                    blocks = list(content)
                else:
                    blocks = [{'type': 'text', 'text': content}] if content else []
                for call in msg['tool_calls']:
                    fn = call.get('function', {})
                    blocks.append({'type': 'tool_use', 'id': call.get('id'), 'name': fn.get('name'),
                                   'input': ClaudeChat._parse_tool_arguments(fn.get('arguments'))})
                result.append({'role': 'assistant', 'content': blocks})
                continue
            result.append({'role': role, 'content': '' if content is None else content})
        return result

    def _output_token_key(self, data):
        return 'max_tokens'

    def _prepare_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data.setdefault('max_tokens', self._DEFAULT_MAX_TOKENS)
        if data.get('tools'):
            data['tools'] = self._convert_tools(data['tools'])
        if isinstance(data.get('messages'), list):
            data['messages'] = self._convert_messages(data['messages'])
        return data

    def _convert_msg_format(self, msg: Dict[str, Any]):
        msg_type = msg.get('type', '')
        if msg_type == 'message':  # non-stream response
            text = ''.join(b.get('text', '') for b in msg.get('content', []) if b.get('type') == 'text')
            tool_calls = [
                {'id': b['id'], 'type': 'function',
                 'function': {'name': b['name'], 'arguments': json.dumps(b.get('input', {}), ensure_ascii=False)}}
                for b in msg.get('content', []) if b.get('type') == 'tool_use'
            ]
            message: Dict[str, Any] = {'role': 'assistant', 'content': text}
            if tool_calls:
                message['tool_calls'] = tool_calls
            usage = msg.get('usage', {})
            return {'choices': [{'message': message,
                                 'finish_reason': ClaudeChat._FINISH_REASONS.get(msg.get('stop_reason'), 'unknown')}],
                    'usage': {'prompt_tokens': usage.get('input_tokens', -1),
                              'completion_tokens': usage.get('output_tokens', -1)}}
        if msg_type == 'content_block_delta':
            delta_obj = msg.get('delta', {})
            if delta_obj.get('type') == 'text_delta':
                return {'choices': [{'delta': {'role': 'assistant', 'content': delta_obj.get('text', '')}}]}
            if delta_obj.get('type') == 'input_json_delta':
                # Partial tool input — carry as tool_calls delta with index
                partial = delta_obj.get('partial_json', '')
                return {'choices': [{'index': msg.get('index', 0),
                                     'delta': {'tool_calls': [{'function': {'arguments': partial}}]}}]}
        if msg_type == 'content_block_start':
            block = msg.get('content_block', {})
            if block.get('type') == 'tool_use':
                # Emit the tool call header (id + name) as the first delta
                return {'choices': [{'index': msg.get('index', 0),
                                     'delta': {'role': 'assistant', 'content': '',
                                               'tool_calls': [{'index': msg.get('index', 0),
                                                               'id': block['id'], 'type': 'function',
                                                               'function': {'name': block['name'], 'arguments': ''}}]}}]}
        if msg_type == 'message_start':
            usage = msg.get('message', {}).get('usage', {})
            return {'choices': [{'delta': {'role': 'assistant', 'content': ''}}],
                    'usage': {'prompt_tokens': usage.get('input_tokens', -1), 'completion_tokens': -1}}
        if msg_type == 'message_delta':
            usage = msg.get('usage', {})
            return {'choices': [{'delta': {'role': 'assistant', 'content': ''},
                                 'finish_reason': ClaudeChat._FINISH_REASONS.get(
                                     msg.get('delta', {}).get('stop_reason'), 'unknown')}],
                    'usage': {'prompt_tokens': -1, 'completion_tokens': usage.get('output_tokens', -1)}}
        return ''  # ping / content_block_stop / message_stop → filtered out

    @staticmethod
    def _merge_request_usage(existing, message):
        usage = dict(existing or {'prompt_tokens': -1, 'completion_tokens': -1})
        for key, value in (message.get('usage') or {}).items():
            if key in ('prompt_tokens', 'completion_tokens') and isinstance(value, (int, float)) and value >= 0:
                usage[key] = value
        usage['provider_usage'] = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens')}
        return usage

    @classmethod
    def _extract_usage(cls, frames):
        usage = cls._merge_request_usage(None, {})
        for frame in frames:
            usage = cls._merge_request_usage(usage, frame)
        return usage

    def _update_usage(self, message, state):
        state.usage = self._merge_request_usage(state.usage, message)

    def _parse_response_frame(self, msg: Union[str, bytes], stream_output: bool):
        parser = self._response_parser(stream_output)
        return parser.parse_response_frame(msg) if stream_output else parser.parse_json_payload(msg)
