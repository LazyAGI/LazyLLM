'''Template contract for WeChat Official Account draft rendering.'''

from __future__ import annotations

from collections.abc import Mapping


class WeChatTemplate:
    '''Provide presentation decisions without owning WeChat API behavior.'''

    template_id = ''
    display_name = ''
    _HEADING_STYLES: Mapping[int, Mapping[str, str]] = {}
    _PARAGRAPH_STYLE: Mapping[str, str] = {}
    _CAPTION_STYLE: Mapping[str, str] = {}
    _QUOTE_STYLE: Mapping[str, str] = {}
    _CODE_STYLE: Mapping[str, str] = {}
    _DIVIDER_STYLE: Mapping[str, str] = {}
    _LIST_STYLE: Mapping[str, str] = {'margin': '10px 0 16px', 'padding-left': '18px'}
    _LIST_ITEM_STYLE: Mapping[str, str] = {'margin': '7px 0', 'padding-left': '28px'}
    _UNORDERED_MARKER_STYLE: Mapping[str, str] = {}
    _ORDERED_MARKER_STYLE: Mapping[str, str] = {}

    def heading_style(self, level: int) -> Mapping[str, str]:
        return self._HEADING_STYLES.get(min(3, max(1, level)), {})

    def paragraph_style(self) -> Mapping[str, str]:
        return self._PARAGRAPH_STYLE

    def caption_style(self) -> Mapping[str, str]:
        return self._CAPTION_STYLE

    def quote_style(self) -> Mapping[str, str]:
        return self._QUOTE_STYLE

    def code_style(self) -> Mapping[str, str]:
        return self._CODE_STYLE

    def divider_style(self) -> Mapping[str, str]:
        return self._DIVIDER_STYLE

    def list_style(self) -> Mapping[str, str]:
        return self._LIST_STYLE

    def list_item_style(self) -> Mapping[str, str]:
        return self._LIST_ITEM_STYLE

    def list_marker_style(self, ordered: bool) -> Mapping[str, str]:
        return self._ORDERED_MARKER_STYLE if ordered else self._UNORDERED_MARKER_STYLE
