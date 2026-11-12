'''OpenAI Responses transport for the existing LazyLLM chat contract.'''

import json
import re
from urllib.parse import urlsplit

from ..base.model_request import model_response, response_frames
from ..base.protocol_ir import RequestIR, ResponseIR, _PROVIDER_ITEMS
from ..base.provider_response import raise_for_http_error
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


def chat_to_responses_request(chat):
    '''Translate chat messages and function definitions without losing call IDs.'''
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
        body['reasoning'] = {'effort': chat['reasoning_effort']}
    _normalize_sampling_params(body, chat)
    if chat.get('response_format'):
        response_format = chat['response_format']
        if response_format.get('type') == 'json_schema':
            body['text'] = {'format': {'type': 'json_schema', **response_format.get('json_schema', {})}}
        else:
            body['text'] = {'format': response_format}
    return body


def responses_to_chat_chunk(response):
    return ResponseIR.from_responses(response).to_chat_chunk()


def _chat_frame(chunk):
    return 'data: ' + json.dumps(chunk, ensure_ascii=False)


def _response_stream_as_chat(frames):
    text_emitted = False
    for raw in frames:
        line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if not payload or payload == '[DONE]':
            continue
        event = json.loads(payload)
        kind = event.get('type')
        if kind == 'response.output_text.delta':
            text_emitted = True
            yield _chat_frame({'choices': [{'index': 0, 'delta': {'content': event.get('delta', '')},
                                            'finish_reason': None}]})
        elif kind in ('response.completed', 'response.incomplete'):
            complete = responses_to_chat_chunk(event['response'])
            choice = complete['choices'][0]
            if not text_emitted and choice['message'].get('content'):
                yield _chat_frame({'choices': [{'index': 0, 'delta': {'content': choice['message']['content']},
                                                'finish_reason': None}]})
            delta = {'tool_calls': choice['message']['tool_calls']} if choice['message'].get('tool_calls') else {}
            if choice['message'].get(_PROVIDER_ITEMS):
                delta[_PROVIDER_ITEMS] = choice['message'][_PROVIDER_ITEMS]
            yield _chat_frame({'choices': [{'index': 0, 'delta': delta,
                                            'finish_reason': choice['finish_reason']}],
                               **({'usage': complete['usage']} if 'usage' in complete else {})})
            yield 'data: [DONE]'
            return
        elif kind in ('response.failed', 'error'):
            error = event.get('error') or (event.get('response') or {}).get('error') or {'message': 'response failed'}
            yield _chat_frame({'error': error})
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
                if data.get('stream'):
                    result = self._response_parser(stream_output).collect(_response_stream_as_chat(frames), state)
                else:
                    raw = next(iter(frames))
                    payload = json.loads(raw)
                    chunk = ({'error': payload.get('error') or {'message': 'response failed'}}
                             if payload.get('status') == 'failed' else responses_to_chat_chunk(payload))
                    result = self._response_parser(stream_output).collect(
                        (_chat_frame(chunk), 'data: [DONE]'), state)
                if state is not None:
                    state.usage = self._extract_usage(result)
                return result
