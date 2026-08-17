from __future__ import annotations

from contextvars import copy_context
import json
from queue import Empty, Queue
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from lazyllm.common import ThreadPoolExecutor
from lazyllm.configs import config

from ..data_models.planning import SectionInstruction
from ..data_models.writer_ir import WriterBlock
from ..utils import render_block_markdown


_DRAFT_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=config['thread_pool_worker_num'])


class MarkdownStreamNormalizer:
    _think_open_pattern = re.compile(r'^<think\b[^>]*>', re.IGNORECASE)
    _think_close_pattern = re.compile(r'</think\s*>', re.IGNORECASE)

    def __init__(self):
        self._leading = ''
        self._trailing = ''
        self._started = False
        self._parts: List[str] = []

    @property
    def body(self) -> str:
        return ''.join(self._parts)

    def feed(self, text: str) -> List[str]:
        if not text:
            return []
        if self._started:
            return self._feed_body(text)
        self._leading += text
        return self._drain_leading()

    def finish(self) -> List[str]:
        deltas = self._drain_leading(final=True) if not self._started else []
        self._trailing = ''
        return deltas

    def _drain_leading(self, final: bool = False) -> List[str]:
        while True:
            stripped = self._leading.lstrip()
            if not stripped:
                if final:
                    self._leading = ''
                return []

            open_match = self._think_open_pattern.match(stripped)
            if open_match:
                close_match = self._think_close_pattern.search(stripped, open_match.end())
                if close_match is None:
                    if not final:
                        return []
                    self._started = True
                    self._leading = ''
                    return self._feed_body(stripped)
                self._leading = stripped[close_match.end():]
                continue

            if not final and self._could_be_think_prefix(stripped):
                return []

            self._started = True
            self._leading = ''
            return self._feed_body(stripped)

    @staticmethod
    def _could_be_think_prefix(text: str) -> bool:
        lowered = text.lower()
        if '<think'.startswith(lowered):
            return True
        return bool(re.match(r'^<think\b', lowered)) and '>' not in lowered

    def _feed_body(self, text: str) -> List[str]:
        combined = self._trailing + text
        trailing_match = re.search(r'\s*$', combined)
        content = combined[:trailing_match.start()]
        self._trailing = combined[trailing_match.start():]
        if not content:
            return []
        self._parts.append(content)
        return [content]


class DraftPreviewStream(Iterator[str]):
    def __init__(
        self,
        call: Callable[[Callable[[Dict[str, Any]], None]], Any],
        consume: Callable[[Dict[str, Any]], List[str]],
        finalize: Callable[[Any], Tuple[List[str], dict]],
        idle_timeout: float,
        *,
        initial_deltas: Optional[List[str]] = None,
        label: str = 'Draft preview',
    ):
        self._queue: Queue[Dict[str, Any]] = Queue()
        self._cancelled = False
        self._last_activity = time.monotonic()
        self._idle_timeout = idle_timeout
        self._consume = consume
        self._finalize = finalize
        self._initial_deltas = initial_deltas or []
        self._label = label
        self._result: Optional[dict] = None
        self._error: Optional[BaseException] = None
        context = copy_context()
        self._future = _DRAFT_STREAM_EXECUTOR.submit(context.run, call, self._sink)
        self._iterator = self._iterate()

    def __iter__(self) -> 'DraftPreviewStream':
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __enter__(self) -> 'DraftPreviewStream':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def result(self) -> dict:
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError(f'{self._label} stream must be fully consumed before result().')
        return self._result

    def close(self) -> None:
        self._cancelled = True
        self._future.cancel()
        self._iterator.close()
        self._drain_queue()

    def _sink(self, payload: Dict[str, Any]) -> None:
        if self._cancelled:
            return
        self._last_activity = time.monotonic()
        self._queue.put(dict(payload))

    def _iterate(self) -> Iterator[str]:
        try:
            yield from self._initial_deltas
            while not self._future.done():
                try:
                    payload = self._queue.get(timeout=self._poll_timeout())
                except Empty:
                    self._raise_if_idle()
                    continue
                yield from self._consume(payload)

            for payload in self._drain_queue():
                yield from self._consume(payload)

            final_deltas, self._result = self._finalize(self._future.result())
            yield from final_deltas
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            if self._result is None:
                self._cancelled = True
                self._future.cancel()

    def _poll_timeout(self) -> float:
        remaining = self._idle_timeout - (time.monotonic() - self._last_activity)
        return max(min(remaining, 0.1), 0.001)

    def _raise_if_idle(self) -> None:
        if not self._future.done() and time.monotonic() - self._last_activity >= self._idle_timeout:
            raise TimeoutError(f'{self._label} stream was idle for {self._idle_timeout:g} seconds.')

    def _drain_queue(self) -> List[Dict[str, Any]]:
        payloads = []
        while True:
            try:
                payloads.append(self._queue.get_nowait())
            except Empty:
                return payloads


class DraftMarkdownStream(DraftPreviewStream):
    def __init__(
        self,
        call: Callable[[Callable[[Dict[str, Any]], None]], str],
        finalize: Callable[[str], dict],
        prefix: str,
        idle_timeout: float,
    ):
        normalizer = MarkdownStreamNormalizer()

        def consume(payload: Dict[str, Any]) -> List[str]:
            if payload.get('tag') != 'text':
                return []
            return normalizer.feed(str(payload.get('delta') or ''))

        def finish(response: Any) -> Tuple[List[str], dict]:
            body = str(response).strip()
            deltas = normalizer.finish()
            if normalizer.body != body:
                raise ValueError('Streamed Markdown body does not match the normalized LLM response.')
            return [*deltas, '\n'], finalize(body)

        super().__init__(
            call=call,
            consume=consume,
            finalize=finish,
            idle_timeout=idle_timeout,
            initial_deltas=[prefix],
            label='Draft Markdown',
        )


class IRPreviewOutput:
    def __init__(self, prefix: str, emit: Callable[[str], None]):
        self.prefix = prefix
        self._emit = emit
        self._has_body = False

    def start_item(self) -> None:
        if self._has_body:
            self._emit('\n\n')
        self._has_body = True

    def append_complete(self, markdown: str) -> None:
        content = markdown.strip()
        if not content:
            return
        self.start_item()
        self._emit(content)


class IRBlockStreamState:
    def __init__(
        self,
        parser: 'IRJSONMarkdownParser',
        *,
        start_index: int,
        level: int,
        root: bool = False,
        suppressed: bool = False,
    ):
        self.parser = parser
        self.start_index = start_index
        self.level = level
        self.root = root
        self.suppressed = suppressed
        self.block_type: Optional[str] = None
        self.numbering: Dict[str, Any] = {}
        self.content_parts: List[str] = []
        self.content_complete = False
        self.streaming = False
        self.force_buffered = suppressed
        self._leading = True
        self._trailing = ''
        self._output_started = False

    def set_type(self, block_type: str) -> None:
        self.block_type = block_type
        if self.root or self.force_buffered:
            return
        if block_type not in self.parser.STREAMABLE_TYPES:
            self.force_buffered = True
            return
        if block_type == 'list_item' and not isinstance(
            self.numbering.get('ordered'), bool,
        ):
            return
        self._enable_streaming()

    def set_numbering(self, numbering: Dict[str, Any]) -> None:
        self.numbering = numbering
        if (
            self.block_type == 'list_item'
            and isinstance(numbering.get('ordered'), bool)
            and not self.root
            and not self.force_buffered
        ):
            self._enable_streaming()

    def _enable_streaming(self) -> None:
        if self.streaming:
            return
        self.streaming = True
        if self.content_parts:
            self._feed_visible(''.join(self.content_parts))
        if self.content_complete:
            self._finish_visible()

    def feed_content(self, text: str) -> None:
        self.content_parts.append(text)
        if self.streaming:
            self._feed_visible(text)

    def finish_content(self) -> None:
        self.content_complete = True
        if self.streaming:
            self._finish_visible()

    def prepare_children(self) -> bool:
        if self.root:
            return False
        if not self.streaming:
            self.force_buffered = True
        return self.force_buffered or self.suppressed

    def finish(self, end_index: int) -> None:
        if self.root or self.suppressed or self.streaming:
            return
        raw = self.parser.raw_json[self.start_index:end_index]
        try:
            block = WriterBlock.model_validate(json.loads(raw))
        except Exception:
            # The complete response is validated by the normal structured-output
            # path. Do not leak an invalid preview block before that validation.
            return
        self.parser.output.append_complete(
            render_block_markdown(block, level=self.level),
        )

    def _feed_visible(self, text: str) -> None:
        for char in text:
            if self._leading and char.isspace():
                continue
            self._leading = False
            if char.isspace():
                self._trailing += char
                continue
            visible = self._trailing + char
            self._trailing = ''
            if not self._output_started:
                self.parser.output.start_item()
                prefix = self._markdown_prefix()
                if prefix:
                    self.parser.emit(prefix)
                self._output_started = True
            self.parser.emit(visible)

    def _finish_visible(self) -> None:
        # Writer's Markdown renderer strips each block's outer whitespace.
        self._trailing = ''

    def _markdown_prefix(self) -> str:
        if self.block_type == 'heading':
            level = min(max(self.level, 1), 6)
            return f'{"#" * level} '
        if self.block_type == 'list_item':
            return '1. ' if self.numbering.get('ordered') else '- '
        return ''


class IRJSONMarkdownParser:
    '''Incrementally expose safe WriterBlock content from streamed JSON.'''

    STREAMABLE_TYPES = frozenset({
        'paragraph', 'quote', 'code', 'heading', 'list_item',
        'todo', 'callout', 'link_preview', 'table', 'divider',
    })
    _ESCAPES = {
        '"': '"', '\\': '\\', '/': '/',
        'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
    }

    def __init__(self, instruction: SectionInstruction):
        self.prefix = f'## {instruction.section_title.strip()}\n\n'
        self._delta_parts: List[str] = []
        self._emitted_parts: List[str] = [self.prefix]
        self.output = IRPreviewOutput(self.prefix, self.emit)
        self.raw_json = ''
        self._started = False
        self._done = False
        self._stack: List[Dict[str, Any]] = []
        self._in_string = False
        self._string_kind = ''
        self._string_context: Optional[Dict[str, Any]] = None
        self._string_key: Optional[str] = None
        self._string_parts: List[str] = []
        self._escape = False
        self._unicode_digits: Optional[str] = None
        self._pending_high_surrogate: Optional[int] = None
        self._primitive = False

    def emit(self, text: str) -> None:
        if not text:
            return
        self._delta_parts.append(text)
        self._emitted_parts.append(text)

    def feed(self, text: str) -> List[str]:
        self._delta_parts = []
        for char in text:
            self._feed_char(char)
        return [''.join(self._delta_parts)] if self._delta_parts else []

    def finish(self, block: WriterBlock) -> List[str]:
        self._delta_parts = []
        final_markdown = render_block_markdown(block, level=2).rstrip() + '\n'
        emitted = ''.join(self._emitted_parts)
        if final_markdown.startswith(emitted):
            self.emit(final_markdown[len(emitted):])
        elif final_markdown.rstrip() != emitted.rstrip():
            raise ValueError('Streamed IR Markdown does not match the validated WriterBlock.')
        return [''.join(self._delta_parts)] if self._delta_parts else []

    def _feed_char(self, char: str) -> None:  # noqa: C901
        if self._done:
            return
        if not self._started:
            if char != '{':
                return
            self._started = True
            self.raw_json = '{'
            root = IRBlockStreamState(self, start_index=0, level=2, root=True)
            self._stack.append({
                'kind': 'object', 'expect': 'key_or_end', 'key': None,
                'block': root,
            })
            return

        self.raw_json += char
        if self._in_string:
            self._feed_string_char(char)
            return

        if self._primitive:
            if char not in ',}]':
                return
            self._primitive = False
            self._handle_delimiter(char)
            return

        if char.isspace():
            return
        if char == '"':
            self._start_string()
        elif char == '{':
            self._start_object()
        elif char == '[':
            self._start_array()
        elif char in '}]':
            self._handle_delimiter(char)
        elif char == ':':
            if self._stack and self._stack[-1]['kind'] == 'object':
                self._stack[-1]['expect'] = 'value'
        elif char == ',':
            self._handle_delimiter(char)
        else:
            self._mark_value_started()
            self._primitive = True

    def _start_string(self) -> None:
        if not self._stack:
            return
        context = self._stack[-1]
        self._string_context = context
        self._string_parts = []
        self._string_key = None
        if context['kind'] == 'object' and context['expect'] == 'key_or_end':
            self._string_kind = 'key'
        else:
            self._string_kind = 'value'
            if context['kind'] == 'object':
                self._string_key = context.get('key')
            self._mark_value_started()
        self._in_string = True
        self._escape = False
        self._unicode_digits = None
        self._pending_high_surrogate = None

    def _feed_string_char(self, char: str) -> None:
        if self._unicode_digits is not None:
            if char not in '0123456789abcdefABCDEF':
                raise ValueError('Invalid unicode escape in streamed WriterBlock JSON.')
            self._unicode_digits += char
            if len(self._unicode_digits) == 4:
                self._emit_string_codepoint(int(self._unicode_digits, 16))
                self._unicode_digits = None
                self._escape = False
            return
        if self._escape:
            if char == 'u':
                self._unicode_digits = ''
                return
            decoded = self._ESCAPES.get(char)
            if decoded is None:
                raise ValueError('Invalid escape in streamed WriterBlock JSON.')
            self._emit_string_text(decoded)
            self._escape = False
            return
        if char == '\\':
            self._escape = True
            return
        if char == '"':
            self._finish_string()
            return
        self._emit_string_text(char)

    def _emit_string_codepoint(self, codepoint: int) -> None:
        if 0xD800 <= codepoint <= 0xDBFF:
            self._pending_high_surrogate = codepoint
            return
        if 0xDC00 <= codepoint <= 0xDFFF and self._pending_high_surrogate is not None:
            high = self._pending_high_surrogate
            self._pending_high_surrogate = None
            codepoint = 0x10000 + ((high - 0xD800) << 10) + (codepoint - 0xDC00)
        elif self._pending_high_surrogate is not None:
            raise ValueError('Invalid surrogate pair in streamed WriterBlock JSON.')
        self._emit_string_text(chr(codepoint))

    def _emit_string_text(self, text: str) -> None:
        if self._string_kind == 'key' or self._string_key == 'type':
            self._string_parts.append(text)
            return
        context = self._string_context
        block = context.get('block') if context else None
        if self._string_key == 'content' and block is not None:
            block.feed_content(text)

    def _finish_string(self) -> None:
        if self._pending_high_surrogate is not None:
            raise ValueError('Incomplete surrogate pair in streamed WriterBlock JSON.')
        context = self._string_context
        value = ''.join(self._string_parts)
        if self._string_kind == 'key' and context is not None:
            context['key'] = value
            context['expect'] = 'colon'
        elif context is not None and context['kind'] == 'object':
            block = context.get('block')
            if block is not None and self._string_key == 'type':
                block.set_type(value)
            elif block is not None and self._string_key == 'content':
                block.finish_content()
        self._in_string = False
        self._string_kind = ''
        self._string_context = None
        self._string_key = None
        self._string_parts = []

    def _start_object(self) -> None:
        parent = self._stack[-1] if self._stack else None
        role, owner, suppressed = self._value_context(parent)
        self._mark_value_started()
        block = None
        if parent and parent['kind'] == 'array' and role == 'children' and owner is not None:
            block = IRBlockStreamState(
                self,
                start_index=len(self.raw_json) - 1,
                level=owner.level + 1,
                suppressed=suppressed,
            )
        self._stack.append({
            'kind': 'object', 'expect': 'key_or_end', 'key': None,
            'block': block,
            'metadata_role': role if role == 'numbering' and owner is not None else None,
            'metadata_owner': owner,
            'start_index': len(self.raw_json) - 1,
        })

    def _start_array(self) -> None:
        parent = self._stack[-1] if self._stack else None
        role, owner, suppressed = self._value_context(parent)
        self._mark_value_started()
        if role == 'children' and owner is not None:
            suppressed = suppressed or owner.prepare_children()
        self._stack.append({
            'kind': 'array', 'expect': 'value_or_end', 'role': role,
            'owner': owner, 'suppressed': suppressed,
        })

    @staticmethod
    def _value_context(
        parent: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[IRBlockStreamState], bool]:
        if parent is None:
            return None, None, False
        if parent['kind'] == 'object':
            owner = parent.get('block')
            return parent.get('key'), owner, bool(owner and owner.suppressed)
        return parent.get('role'), parent.get('owner'), bool(parent.get('suppressed'))

    def _mark_value_started(self) -> None:
        if not self._stack:
            return
        context = self._stack[-1]
        if context['kind'] == 'object' and context['expect'] == 'value':
            context['expect'] = 'comma_or_end'
        elif context['kind'] == 'array' and context['expect'] == 'value_or_end':
            context['expect'] = 'comma_or_end'

    def _handle_delimiter(self, char: str) -> None:
        if not self._stack:
            return
        if char == ',':
            context = self._stack[-1]
            context['expect'] = 'key_or_end' if context['kind'] == 'object' else 'value_or_end'
            if context['kind'] == 'object':
                context['key'] = None
            return
        context = self._stack.pop()
        if char == '}' and context['kind'] == 'object':
            if context.get('metadata_role') == 'numbering':
                raw = self.raw_json[context['start_index']:len(self.raw_json)]
                try:
                    numbering = json.loads(raw)
                except Exception:
                    numbering = None
                if isinstance(numbering, dict):
                    context['metadata_owner'].set_numbering(numbering)
            block = context.get('block')
            if block is not None:
                block.finish(len(self.raw_json))
        if not self._stack:
            self._done = True


class DraftIRStream(DraftPreviewStream):
    def __init__(
        self,
        call: Callable[[Callable[[Dict[str, Any]], None]], WriterBlock],
        normalize: Callable[[WriterBlock], WriterBlock],
        finalize: Callable[[WriterBlock], dict],
        instruction: SectionInstruction,
        idle_timeout: float,
    ):
        parser = IRJSONMarkdownParser(instruction)

        def consume(payload: Dict[str, Any]) -> List[str]:
            if payload.get('tag') != 'text':
                return []
            return parser.feed(str(payload.get('delta') or ''))

        def finish(response: Any) -> Tuple[List[str], dict]:
            if not isinstance(response, WriterBlock):
                response = WriterBlock.model_validate(response)
            block = normalize(response)
            return parser.finish(block), finalize(block)

        super().__init__(
            call=call,
            consume=consume,
            finalize=finish,
            idle_timeout=idle_timeout,
            initial_deltas=[parser.prefix],
            label='Draft IR',
        )
