from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Optional, Tuple

from .model_outcome import ModelFinish


# Protocol extensions are private to an adapter, never workflow instructions.
_PROVIDER_ITEMS = '_lazyllm_provider_items'


@dataclass(frozen=True)
class ContentBlock:
    kind: str
    value: Any = field(repr=False)
    attributes: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_chat(cls, value):
        if isinstance(value, str):
            return cls('text', value)
        value = deepcopy(value)
        kind = value.get('type')
        if kind == 'image':
            source = value.get('source') or {}
            reference = source.get('url')
            if source.get('type') == 'base64':
                reference = 'data:' + source.get('media_type', 'image/jpeg') + ';base64,' + source.get('data', '')
            return cls('image', reference, {'_original': value})
        if kind in ('text', 'input_text', 'output_text'):
            return cls('text', value.get('text', ''), {k: v for k, v in value.items() if k not in ('type', 'text')})
        if kind in ('image_url', 'input_image'):
            image = value.get('image_url') or {}
            if isinstance(image, str):
                image = {'url': image}
            attrs = {k: v for k, v in image.items() if k != 'url'}
            attrs.update({k: v for k, v in value.items() if k not in ('type', 'image_url')})
            return cls('image', image.get('url'), attrs)
        if kind in ('file', 'input_file'):
            return cls('file', value.get('file', {k: v for k, v in value.items() if k != 'type'}))
        if kind == 'refusal':
            return cls('refusal', value.get('refusal', ''))
        return cls('opaque', value)

    def to_chat(self):
        if '_original' in self.attributes:
            return deepcopy(self.attributes['_original'])
        if self.kind == 'text':
            return {'type': 'text', 'text': self.value, **deepcopy(self.attributes)}
        if self.kind == 'image':
            return {'type': 'image_url', 'image_url': {'url': self.value, **deepcopy(self.attributes)}}
        if self.kind == 'file':
            return {'type': 'file', 'file': deepcopy(self.value)}
        if self.kind == 'refusal':
            return {'type': 'refusal', 'refusal': self.value}
        return deepcopy(self.value)

    def to_responses(self, role):
        if self.kind == 'text':
            return {'type': 'output_text' if role == 'assistant' else 'input_text',
                    'text': self.value, **deepcopy(self.attributes)}
        if self.kind == 'image':
            return {'type': 'input_image', **({'image_url': self.value} if self.value else {}),
                    **{k: deepcopy(v) for k, v in self.attributes.items() if k != '_original'}}
        if self.kind == 'file':
            return {'type': 'input_file', **deepcopy(self.value)}
        return self.to_chat()


@dataclass(frozen=True)
class ToolCallIR:
    id: str
    name: str
    arguments: Any = field(repr=False)
    attributes: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_chat(cls, call):
        function = call.get('function') or {}
        return cls(call.get('id', ''), function.get('name', ''), deepcopy(function.get('arguments', '{}')),
                   {k: deepcopy(v) for k, v in call.items() if k not in ('id', 'function')})

    def to_chat(self):
        return {**deepcopy(self.attributes), 'id': self.id, 'type': 'function',
                'function': {'name': self.name, 'arguments': deepcopy(self.arguments)}}

    def to_responses(self):
        arguments = self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments, ensure_ascii=False)
        return {'type': 'function_call', 'call_id': self.id, 'name': self.name, 'arguments': arguments}


@dataclass(frozen=True)
class MessageIR:
    role: str
    content: Tuple[ContentBlock, ...] = ()
    tool_calls: Tuple[ToolCallIR, ...] = ()
    tool_call_id: Optional[str] = None
    attributes: dict = field(default_factory=dict, repr=False)
    string_content: bool = False
    native_item: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def from_chat(cls, message):
        if isinstance(message, str):
            message = {'role': 'assistant', 'content': message}
        content = message.get('content')
        if content is not None and not isinstance(content, (str, list)):
            content = json.dumps(content, ensure_ascii=False)
        if message.get('role') == 'tool' and isinstance(content, list) and any(
                not isinstance(part, dict) or 'type' not in part for part in content):
            content = json.dumps(content, ensure_ascii=False)
        blocks = (ContentBlock('text', content),) if isinstance(content, str) else tuple(
            ContentBlock.from_chat(part) for part in content or [])
        return cls(message.get('role', 'assistant'), blocks,
                   tuple(ToolCallIR.from_chat(call) for call in message.get('tool_calls') or []),
                   message.get('tool_call_id'),
                   {k: deepcopy(v) for k, v in message.items()
                    if k not in ('role', 'content', 'tool_calls', 'tool_call_id')}, isinstance(content, str))

    @property
    def text(self):
        return ''.join(block.value for block in self.content if block.kind == 'text')

    def to_chat(self):
        message = {**deepcopy(self.attributes), 'role': self.role,
                   'content': self.text if self.string_content else [block.to_chat() for block in self.content]}
        if self.tool_calls:
            message['tool_calls'] = [call.to_chat() for call in self.tool_calls]
        if self.tool_call_id is not None:
            message['tool_call_id'] = self.tool_call_id
        return message

    def to_responses(self):
        if self.native_item is not None:
            return [deepcopy(self.native_item)]
        extensions = self.attributes.get(_PROVIDER_ITEMS, {})
        if extensions:
            if extensions.get('protocol') != 'responses':
                raise ValueError('Cannot replay opaque items across protocol boundaries')
            prefix = deepcopy(extensions['items'])
        else:
            prefix = []
        if self.role == 'tool':
            output = self.text if self.string_content else [block.to_responses('user') for block in self.content]
            return [{'type': 'function_call_output', 'call_id': self.tool_call_id, 'output': output}]
        items = []
        if self.content and (not self.string_content or self.text):
            content = self.text if self.string_content else [block.to_responses(self.role) for block in self.content]
            items.append({'role': self.role, 'content': content})
        items.extend(call.to_responses() for call in self.tool_calls)
        return _ordered_response_items(items, prefix, extensions.get('order'), extensions.get('text_digest'))

    def estimate_tokens(self, text_estimator, message_estimator=None):
        message = self.to_chat()
        # Estimate semantic content, never encoded media or encrypted continuation.
        media = sum(block.kind in ('image', 'file') for block in self.content)
        if not self.string_content:
            message['content'] = [block.to_chat() for block in self.content if block.kind not in ('image', 'file')]
        message.pop(_PROVIDER_ITEMS, None)
        estimator = message_estimator or (
            lambda value: text_estimator(json.dumps(value, ensure_ascii=False, default=str)))
        return estimator(message) + 1536 * media


@dataclass(frozen=True)
class RequestIR:
    messages: Tuple[MessageIR, ...]
    parameters: dict = field(default_factory=dict, repr=False)
    protocol: str = 'completions'

    @classmethod
    def from_payload(cls, payload, protocol=None):
        protocol = protocol or ('responses' if 'input' in payload and 'messages' not in payload else 'completions')
        if protocol == 'completions':
            return cls(tuple(MessageIR.from_chat(message) for message in payload.get('messages', [])),
                       {k: deepcopy(v) for k, v in payload.items() if k != 'messages'}, protocol)
        if protocol != 'responses':
            raise ValueError(f'Unsupported request protocol: {protocol}')
        items = payload.get('input') or []
        if isinstance(items, str):
            items = [{'role': 'user', 'content': items}]
        return cls(tuple(_response_item(item) for item in items),
                   {k: deepcopy(v) for k, v in payload.items() if k != 'input'}, protocol)

    def to_payload(self, protocol=None):
        protocol = protocol or self.protocol
        if protocol not in ('completions', 'responses'):
            raise ValueError(f'Unsupported request protocol: {protocol}')
        if protocol != self.protocol:
            # Options (sampling, structured output, tools) are adapter-owned.
            raise ValueError('Use a protocol adapter to translate request parameters')
        if protocol == 'completions':
            return {**deepcopy(self.parameters), 'messages': [message.to_chat() for message in self.messages]}
        items = [item for message in self.messages for item in message.to_responses()]
        return {**deepcopy(self.parameters), 'input': items}

    def estimate_tokens(self, text_estimator, message_estimator=None):
        return text_estimator(json.dumps(self.parameters, ensure_ascii=False)) + sum(
            message.estimate_tokens(text_estimator, message_estimator) + 4 for message in self.messages)


def _response_item(item):
    kind = item.get('type')
    if kind == 'function_call':
        call = ToolCallIR(item['call_id'], item['name'], item.get('arguments', '{}'))
        message = MessageIR('assistant', tool_calls=(call,))
    elif kind == 'function_call_output':
        message = MessageIR.from_chat({'role': 'tool', 'tool_call_id': item['call_id'],
                                       'content': item.get('output', '')})
    elif item.get('role') or kind == 'message':
        message = MessageIR.from_chat(item)
    else:
        message = MessageIR('assistant')
    # Retain IDs, signed/encrypted reasoning and provider-native item ordering.
    return MessageIR(message.role, message.content, message.tool_calls, message.tool_call_id,
                     message.attributes, message.string_content, deepcopy(item))


def _ordered_response_items(items, opaque, order, text_digest=None):
    if order is None:
        return opaque + items
    remaining = list(items)
    result = []
    message = next((item for item in remaining if 'role' in item), None)
    text = message.get('content') if message else None
    split_text = isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == text_digest
    offset = 0
    if split_text:
        remaining.remove(message)
    for slot in order:
        if 'opaque' in slot:
            result.append(deepcopy(opaque[slot['opaque']]))
            continue
        if 'message' in slot and split_text:
            length = slot['message']
            if length:
                result.append({**message, 'content': text[offset:offset + length]})
            offset += length
            continue
        index = next((i for i, item in enumerate(remaining)
                      if (item.get('call_id') == slot['call_id'] if 'call_id' in slot else 'role' in item)), None)
        if index is not None:
            result.append(remaining.pop(index))
    return result + remaining


def _response_extensions(items):
    opaque, order, text = [], [], ''
    for item in items:
        if item.get('type') == 'message':
            segment = ''.join(part.get('text', '') for part in item.get('content') or []
                              if part.get('type') == 'output_text')
            text += segment
            order.append({'message': len(segment)})
        elif item.get('type') == 'function_call':
            order.append({'call_id': item['call_id']})
        else:
            order.append({'opaque': len(opaque)})
            opaque.append(deepcopy(item))
    if not opaque and not any(item.get('type') == 'message' for item in items[1:]):
        return {}
    return {_PROVIDER_ITEMS: {'protocol': 'responses', 'items': opaque, 'order': order,
                              'text_digest': hashlib.sha256(text.encode()).hexdigest()}}


@dataclass(frozen=True)
class ResponseIR:
    message: MessageIR
    finish: ModelFinish = ModelFinish.STOP
    usage: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def from_responses(cls, response):
        items = response.get('output') or []
        messages = [_response_item(item) for item in items]
        blocks = tuple(block for message in messages for block in message.content)
        calls = tuple(call for message in messages for call in message.tool_calls)
        # Incomplete/filter status wins over tool presence: partial arguments must not execute.
        finish = ModelFinish.TOOL_CALLS if calls else ModelFinish.STOP
        if response.get('status') == 'incomplete':
            reason = (response.get('incomplete_details') or {}).get('reason')
            finish = ModelFinish.CONTENT_FILTER if reason == 'content_filter' else ModelFinish.LENGTH
        elif any(block.kind == 'refusal' for block in blocks):
            finish = ModelFinish.CONTENT_FILTER
        elif (response.get('status') not in (None, 'completed')
              or not calls and not any(block.kind == 'text' and block.value for block in blocks)):
            finish = ModelFinish.UNKNOWN
        attributes = _response_extensions(items)
        message = MessageIR('assistant', blocks, calls, attributes=attributes, string_content=True)
        usage = response.get('usage')
        if isinstance(usage, dict):
            usage = {**deepcopy(usage), 'prompt_tokens': usage.get('input_tokens', 0),
                     'completion_tokens': usage.get('output_tokens', 0)}
        return cls(message, finish, usage)

    def to_chat_chunk(self):
        message = self.message.to_chat()
        result = {'choices': [{'index': 0, 'message': message, 'finish_reason': self.finish.value}]}
        if self.usage is not None:
            result['usage'] = deepcopy(self.usage)
        return result
