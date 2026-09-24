'''Bundled document templates for Writer exports.'''

from .wechat import (
    WECHAT_CLEAN_TEMPLATE,
    WECHAT_EDITORIAL_TEMPLATE,
    WECHAT_STRUCTURED_TEMPLATE,
    WeChatTemplate,
    get_wechat_template,
    list_wechat_templates,
)

__all__ = [
    'WECHAT_CLEAN_TEMPLATE',
    'WECHAT_EDITORIAL_TEMPLATE',
    'WECHAT_STRUCTURED_TEMPLATE',
    'WeChatTemplate',
    'get_wechat_template',
    'list_wechat_templates',
]
