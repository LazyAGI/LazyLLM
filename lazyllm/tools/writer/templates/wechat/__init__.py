'''Built-in WeChat Official Account rendering templates.'''

from __future__ import annotations

from .base import WeChatTemplate
from .clean import CleanWeChatTemplate
from .editorial import EditorialWeChatTemplate
from .structured import StructuredWeChatTemplate

WECHAT_CLEAN_TEMPLATE = 'clean'
WECHAT_STRUCTURED_TEMPLATE = 'structured'
WECHAT_EDITORIAL_TEMPLATE = 'editorial'

_TEMPLATES: dict[str, WeChatTemplate] = {
    WECHAT_CLEAN_TEMPLATE: CleanWeChatTemplate(),
    WECHAT_STRUCTURED_TEMPLATE: StructuredWeChatTemplate(),
    WECHAT_EDITORIAL_TEMPLATE: EditorialWeChatTemplate(),
}


def _normalize_template(template: str | None) -> str:
    value = str(template or '').strip().lower()
    if value.startswith('wechat.'):
        value = value.removeprefix('wechat.')
    return value or WECHAT_CLEAN_TEMPLATE


def get_wechat_template(template: str | None = None) -> WeChatTemplate:
    key = _normalize_template(template)
    resolved = _TEMPLATES.get(key)
    if resolved is None:
        available = ', '.join(sorted(_TEMPLATES))
        raise ValueError(
            f'Unknown WeChat template {template!r}. Available templates: {available}.')
    return resolved


def list_wechat_templates() -> list[dict[str, str]]:
    return [
        {'id': item.template_id, 'name': item.display_name}
        for item in _TEMPLATES.values()
    ]


__all__ = [
    'WECHAT_CLEAN_TEMPLATE',
    'WECHAT_EDITORIAL_TEMPLATE',
    'WECHAT_STRUCTURED_TEMPLATE',
    'WeChatTemplate',
    'get_wechat_template',
    'list_wechat_templates',
]
