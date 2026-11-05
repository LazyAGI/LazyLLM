from .provider_response import _OpenAICompatibleResponseParser


def responses_request(data):
    data = dict(data)
    items = []
    for message in data.pop('messages', []):
        role = message['role']
        if role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                          'output': message.get('content') or ''})
            continue
        content = message.get('content')
        if content:
            if isinstance(content, list):
                converted = []
                for part in content:
                    if part['type'] == 'text':
                        converted.append({'type': 'output_text' if role == 'assistant' else 'input_text',
                                          'text': part['text']})
                    elif part['type'] == 'image_url':
                        image = part['image_url']
                        converted.append({'type': 'input_image', 'image_url': image['url'],
                                          'detail': image.get('detail', 'auto')})
                    else:
                        converted.append(part)
                content = converted
            items.append({'role': role, 'content': content})
        for call in message.get('tool_calls') or []:
            items.append({'type': 'function_call', 'call_id': call['id'],
                          'name': call['function']['name'], 'arguments': call['function']['arguments']})
    data['input'] = items
    data.setdefault('store', False)
    if 'tools' in data:
        data['tools'] = [dict(type='function', **dict({'strict': False}, **t['function']))
                         if t.get('type') == 'function' and 'function' in t else t for t in data['tools']]
    choice = data.get('tool_choice')
    if isinstance(choice, dict) and 'function' in choice:
        data['tool_choice'] = {'type': 'function', 'name': choice['function']['name']}
    limits = [data.pop(k) for k in ('max_tokens', 'max_completion_tokens', 'max_output_tokens')
              if data.get(k) is not None]
    if limits:
        data['max_output_tokens'] = min(limits)
    effort = data.pop('reasoning_effort', None)
    if effort is not None:
        data['reasoning'] = dict(data.get('reasoning') or {}, effort=effort)
    if (data.get('reasoning') or {}).get('effort') != 'none':
        for key in ('temperature', 'top_p', 'logprobs', 'top_logprobs'):
            data.pop(key, None)
    for key in ('stream_options', 'n'):
        data.pop(key, None)
    if 'response_format' in data:
        fmt = data.pop('response_format')
        if fmt.get('type') == 'json_schema':
            fmt = dict(type='json_schema', **fmt['json_schema'])
        data['text'] = dict(data.get('text') or {}, format=fmt)
    return data


class OpenAIResponsesParser(_OpenAICompatibleResponseParser):
    def _extract_provider_error(self, message):
        if message.get('type') == 'response.failed' or message.get('status') == 'failed':
            return self._profile.extract_error(message.get('response', message)) or (None, None)
        return super()._extract_provider_error(message)


def responses_message(message):
    kind = message.get('type', '')
    if not kind.startswith('response.') and message.get('object') != 'response':
        return message
    delta = {'role': 'assistant'}
    finish = None
    usage = None
    if kind in ('response.output_text.delta', 'response.refusal.delta'):
        delta['content'] = message.get('delta', '')
    elif kind == 'response.output_item.added':
        item = message['item']
        if item['type'] == 'function_call':
            delta['tool_calls'] = [{'index': message['output_index'], 'id': item['call_id'], 'type': 'function',
                                    'function': {'name': item['name'], 'arguments': ''}}]
    elif kind == 'response.function_call_arguments.delta':
        delta['tool_calls'] = [{'index': message['output_index'], 'function': {'arguments': message['delta']}}]
    elif kind in ('response.completed', 'response.incomplete') or message.get('object') == 'response':
        response = message.get('response', message)
        calls = [item for item in response.get('output', []) if item['type'] == 'function_call']
        finish = 'tool_calls' if calls else 'stop'
        if response.get('status') == 'incomplete':
            reason = (response.get('incomplete_details') or {}).get('reason')
            finish = {'max_output_tokens': 'length', 'content_filter': 'content_filter'}.get(reason, 'unknown')
        if message.get('object') == 'response':
            delta['content'] = ''.join(part.get('text', part.get('refusal', ''))
                                       for item in response.get('output', []) if item['type'] == 'message'
                                       for part in item.get('content', []))
            if calls:
                delta['tool_calls'] = [{'index': i, 'id': item['call_id'], 'type': 'function',
                                        'function': {'name': item['name'], 'arguments': item['arguments']}}
                                       for i, item in enumerate(calls)]
        raw = response.get('usage')
        if raw:
            usage = dict(raw, prompt_tokens=raw.get('input_tokens', -1),
                         completion_tokens=raw.get('output_tokens', -1))
    else:
        return ''
    result = {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
    if usage is not None:
        result['usage'] = usage
    return result
