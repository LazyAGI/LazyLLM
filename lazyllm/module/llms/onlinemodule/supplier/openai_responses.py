'''OpenAI Responses transport for the existing LazyLLM chat contract.'''

import json
import re
from copy import deepcopy
from urllib.parse import urlsplit

from ..base.model_request import model_response, response_frames
from ..base.protocol_ir import ProtocolCapabilityError, RequestIR, ResponseIR, _PROVIDER_ITEMS
from ..base.provider_response import raise_for_http_error
from ..base.model_outcome import ModelFailureOrigin
from .openai import OpenAIChat


def _normalize_sampling_params(body, chat):
    reasoning = body.get('reasoning')
    effort = reasoning.get('effort') if isinstance(reasoning, dict) else None
    model = str(body.get('model') or '').lower()
    reasoning_model = bool(re.match(r'^(?:gpt-[56]|o[134])(?:[.-]|$)', model))
    if reasoning_model and effort != 'none':
        for key in ('temperature', 'top_p', 'top_logprobs'):
            body.pop(key, None)
        if isinstance(body.get('include'), list):
            body['include'] = [item for item in body['include'] if item != 'message.output_text.logprobs']
            if not body['include']:
                body.pop('include')
    elif chat.get('logprobs') is True:
        include = list(body.get('include') or [])
        if 'message.output_text.logprobs' not in include:
            include.append('message.output_text.logprobs')
        body['include'] = include


def _validate_responses_options(chat):
    if chat.get('background'):
        raise ProtocolCapabilityError('Responses background jobs require a polling lifecycle and are not supported')
    unsupported = {'stop', 'frequency_penalty', 'presence_penalty', 'seed', 'logit_bias',
                   'audio', 'modalities', 'prediction'}
    supplied = sorted(key for key in unsupported if chat.get(key) is not None)
    if supplied:
        raise ProtocolCapabilityError('Responses cannot translate chat options: ' + ', '.join(supplied))
    if chat.get('n') not in (None, 1):
        raise ProtocolCapabilityError('Responses supports n=1 only')
    if any(tool.get('type') != 'function' for tool in chat.get('tools') or []):
        raise ProtocolCapabilityError('Responses workflow adapter supports function tools only')


def _normalize_legacy_tools(chat):
    if chat.get('functions') is not None:
        if chat.get('tools') is not None:
            raise ProtocolCapabilityError('Specify tools or legacy functions, not both')
        chat['tools'] = [{'type': 'function', 'function': function} for function in chat['functions']]
    if chat.get('function_call') is not None:
        if chat.get('tool_choice') is not None:
            raise ProtocolCapabilityError('Specify tool_choice or legacy function_call, not both')
        choice = chat['function_call']
        chat['tool_choice'] = ({'type': 'function', 'function': choice} if isinstance(choice, dict) else choice)


def chat_to_responses_request(chat):
    '''Translate chat messages and function definitions without losing call IDs.'''
    chat = deepcopy(chat)
    _normalize_legacy_tools(chat)
    _validate_responses_options(chat)
    body = {key: value for key, value in chat.items() if key not in {
        'messages', 'tools', 'max_tokens', 'max_completion_tokens', 'reasoning_effort',
        'stream_options', 'response_format', 'n', 'functions', 'function_call', 'logprobs',
    }}
    request = RequestIR.from_payload(chat, protocol='completions')
    body['input'] = [item for message in request.messages for item in message.to_responses()]
    if chat.get('tools'):
        body['tools'] = [dict(tool.get('function') or {}, type='function') if tool.get('type') == 'function'
                         else tool for tool in chat['tools']]
    choice = chat.get('tool_choice')
    if isinstance(choice, dict) and choice.get('type') == 'function' and isinstance(choice.get('function'), dict):
        body['tool_choice'] = {'type': 'function', 'name': choice['function']['name']}
    limit = chat.get('max_completion_tokens', chat.get('max_tokens'))
    if limit is not None:
        body['max_output_tokens'] = limit
    if chat.get('reasoning_effort') is not None:
        body['reasoning'] = {**(body.get('reasoning') or {}), 'effort': chat['reasoning_effort']}
    _normalize_sampling_params(body, chat)
    if chat.get('response_format'):
        response_format = chat['response_format']
        if response_format.get('type') == 'json_schema':
            body['text'] = {**(body.get('text') or {}),
                            'format': {'type': 'json_schema', **response_format.get('json_schema', {})}}
        else:
            body['text'] = {**(body.get('text') or {}), 'format': response_format}
    return body


def responses_to_chat_chunk(response):
    return ResponseIR.from_responses(response).to_chat_chunk()


def _chat_frame(chunk):
    return 'data: ' + json.dumps(chunk, ensure_ascii=False)


def _protocol_error(message):
    return OpenAIChat.RESPONSE_PROFILE.error(message, ModelFailureOrigin.PROTOCOL)


def _response_json(payload):
    try:
        value = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise _protocol_error('Responses returned invalid JSON.') from exc
    if not isinstance(value, dict):
        raise _protocol_error('Responses returned a non-object payload.')
    return value


def _response_chunk(response):
    if not isinstance(response, dict):
        raise _protocol_error('Responses event is missing a response object.')
    try:
        return responses_to_chat_chunk(response)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise _protocol_error('Responses returned an invalid output item.') from exc


def _response_stream_as_chat(frames):  # noqa: C901
    # One ordered event loop owns stream state and terminal validation; splitting
    # event handlers across independent consumers would obscure replay ordering.
    # Keep deltas observable even when no terminal snapshot arrives. Calls are
    # accumulated by provider item identity; only terminal finishes execute them.
    emitted = {'content': '', 'refusal': '', 'reasoning_content': ''}
    calls = {}
    item_calls = {}

    def delta_frame(delta, finish=None, usage=None):
        chunk = {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        if usage is not None:
            chunk['usage'] = usage
        return _chat_frame(chunk)

    def text_delta(field, value, snapshot=False):
        if not isinstance(value, str):
            raise _protocol_error('Responses returned a non-text delta.')
        if snapshot:
            if not value.startswith(emitted[field]):
                raise _protocol_error('Responses snapshot disagrees with emitted text.')
            value = value[len(emitted[field]):]
        emitted[field] += value
        return {field: value} if value else {}

    def tool_delta(item):
        call_id = item.get('call_id')
        if not call_id or not item.get('name'):
            raise _protocol_error('Responses function call lacks its identity.')
        if item.get('id'):
            item_calls[item['id']] = call_id
        previous = calls.get(call_id)
        arguments = item.get('arguments', '')
        if not isinstance(arguments, str):
            raise _protocol_error('Responses function arguments must be a string.')
        if previous is None:
            previous = calls[call_id] = {'index': len(calls), 'arguments': '', 'name': item['name']}
            delta = {'index': previous['index'], 'id': call_id, 'type': 'function',
                     'function': {'name': item['name'], 'arguments': arguments}}
        else:
            if previous['name'] != item['name'] or not arguments.startswith(previous['arguments']):
                raise _protocol_error('Responses function snapshot disagrees with emitted arguments.')
            delta = {'index': previous['index'], 'function': {'arguments': arguments[len(previous['arguments']):]}}
        previous['arguments'] = arguments
        return {'tool_calls': [delta]}

    for raw in frames:
        try:
            line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        except UnicodeError as exc:
            raise _protocol_error('Responses returned non-UTF-8 data.') from exc
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if not payload or payload == '[DONE]':
            continue
        event = _response_json(payload)
        kind = event.get('type')
        if not isinstance(kind, str):
            raise _protocol_error('Responses event lacks a type.')
        fields = {'response.output_text.delta': 'content', 'response.refusal.delta': 'refusal',
                  'response.reasoning_summary_text.delta': 'reasoning_content',
                  'response.reasoning_text.delta': 'reasoning_content'}
        if kind in fields:
            yield delta_frame(text_delta(fields[kind], event.get('delta')))
        elif kind in ('response.output_item.added', 'response.output_item.done'):
            item = event.get('item')
            if not isinstance(item, dict):
                raise _protocol_error('Responses item event lacks an item.')
            if item.get('type') == 'function_call':
                yield delta_frame(tool_delta(item))
        elif kind == 'response.function_call_arguments.delta':
            call_id = item_calls.get(event.get('item_id'))
            if call_id is None:
                raise _protocol_error('Responses argument delta has no function call item.')
            call = calls[call_id]
            arguments = event.get('delta')
            if not isinstance(arguments, str):
                raise _protocol_error('Responses function argument delta must be a string.')
            call['arguments'] += arguments
            yield delta_frame({'tool_calls': [{'index': call['index'], 'function': {'arguments': arguments}}]})
        elif kind in ('response.completed', 'response.incomplete', 'response.failed'):
            response = event.get('response')
            complete = _response_chunk(response)
            choice = complete['choices'][0]
            message = choice['message']
            if kind == 'response.completed' and set(calls) - {
                    call.get('id') for call in message.get('tool_calls') or []}:
                raise _protocol_error('Responses terminal snapshot dropped a streamed function call.')
            for field in ('content', 'refusal'):
                delta = text_delta(field, message.get(field) or '', snapshot=True)
                if delta:
                    yield delta_frame(delta)
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                yield delta_frame(tool_delta({'call_id': call.get('id'), **function}))
            delta = {_PROVIDER_ITEMS: message[_PROVIDER_ITEMS]} if message.get(_PROVIDER_ITEMS) else {}
            # Failure payloads may carry billable usage and partial output.
            yield delta_frame(delta, None if kind == 'response.failed' else choice['finish_reason'],
                              complete.get('usage'))
            if kind == 'response.failed':
                yield _chat_frame({'error': response.get('error') or {'message': 'response failed'}})
            else:
                yield 'data: [DONE]'
            return
        elif kind == 'error':
            yield _chat_frame({'error': event.get('error') or event})
            return


class OpenAIResponsesChat(OpenAIChat):
    __lazyllm_registry_key__ = 'openai_responseschat'
    PROVIDER_NAME = 'openai_responses'

    def _get_chat_url(self, url):
        parts = urlsplit(url)
        path = parts.path.rstrip('/')
        for suffix in ('/chat/completions', '/responses'):
            if path.endswith(suffix):
                path = path[:-len(suffix)]
                break
        return parts._replace(path=path + '/responses').geturl()

    def _prepare_request_data(self, data):
        return chat_to_responses_request(data)

    def _forward_impl(self, data, *, runtime_url, stream_output, proxies, request_timeout,
                      state=None, call_control=None):
        with model_response(runtime_url, json=data, headers=self._prepare_request_headers(runtime_url),
                            stream=True, proxies=proxies, timeout=request_timeout, control=call_control) as response:
            if state is not None:
                state.response_started = True
            raise_for_http_error(response, self.RESPONSE_PROFILE, control=call_control)
            with self.stream_output(stream_output):
                frames = response_frames(response, bool(data.get('stream')), call_control)
                try:
                    if data.get('stream'):
                        result = self._response_parser(stream_output).collect(_response_stream_as_chat(frames), state)
                    else:
                        payload = _response_json(next(iter(frames)))
                        if payload.get('error') and payload.get('status') != 'failed':
                            converted = [_chat_frame({'error': payload['error']})]
                        elif payload.get('status') == 'failed':
                            converted = _response_stream_as_chat([_chat_frame(
                                {'type': 'response.failed', 'response': payload})])
                        else:
                            converted = [_chat_frame(_response_chunk(payload)), 'data: [DONE]']
                        result = self._response_parser(stream_output).collect(converted, state)
                finally:
                    if state is not None:
                        state.usage = self._extract_usage(state.frames)
                return result
