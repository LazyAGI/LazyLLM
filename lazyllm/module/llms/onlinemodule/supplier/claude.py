import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, ModelFailureCode, ModelFailureOrigin


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

    def _output_token_key(self, data):
        return 'max_tokens'

    def _prepare_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data.setdefault('max_tokens', self._DEFAULT_MAX_TOKENS)
        if data.get('tools'):
            data['tools'] = self._convert_tools(data['tools'])
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
    def _extract_usage(frames):
        usage = {'prompt_tokens': -1, 'completion_tokens': -1}
        for frame in frames:
            for key, value in (frame.get('usage') or {}).items():
                if key in usage and isinstance(value, (int, float)) and value >= 0:
                    usage[key] = value
        return {**usage, 'provider_usage': dict(usage)}

    def _update_usage(self, message, state):
        usage = dict(state.usage or {'prompt_tokens': -1, 'completion_tokens': -1})
        for key, value in (message.get('usage') or {}).items():
            if key in ('prompt_tokens', 'completion_tokens') and isinstance(value, (int, float)) and value >= 0:
                usage[key] = value
        usage['provider_usage'] = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens')}
        state.usage = usage

    def _parse_response_frame(self, msg: Union[str, bytes], stream_output: bool):
        if stream_output:
            msg = self._response_parser(False).extract_sse_payload(msg)
            if msg is None:
                return None
        try:
            raw = json.loads(msg)
        except (ValueError, TypeError) as exc:
            raise self.RESPONSE_PROFILE.error('Invalid model response.', ModelFailureOrigin.PROTOCOL) from exc
        if not isinstance(raw, dict):
            raise self.RESPONSE_PROFILE.error('Invalid model response.', ModelFailureOrigin.PROTOCOL)
        fields = self.RESPONSE_PROFILE.extract_error(raw)
        if fields is not None:
            code, error_type = fields
            raise self.RESPONSE_PROFILE.error('Provider returned a model error.', ModelFailureOrigin.PROVIDER,
                                              provider_error_code=code, provider_error_type=error_type)
        message = self._convert_msg_format(raw)
        if stream_output:
            color = stream_output.get('color') if isinstance(stream_output, dict) else None
            for item in (message.get('choices', []) if isinstance(message, dict) else []):
                delta = item.get('delta', {})
                if (content := delta.get('content', '')) and not delta.get('tool_calls'):
                    self._stream_output(content, color)
        return message
