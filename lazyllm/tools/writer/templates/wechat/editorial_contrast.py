# Adapted from https://github.com/zhouke0929/wechat-visual-director (Apache-2.0).
# Source revision 9b0d47cdb456c371c8ba0fb4dcc7e0d1955944b0.
# Modified for Writer's template API and mobile layout.
from .base import WeChatTemplate


class EditorialContrastWeChatTemplate(WeChatTemplate):
    template_id = 'editorial_contrast'
    display_name = 'Editorial Contrast'

    _HEADING_STYLES = {
        1: {
            'margin': '45px 0 23px',
            'padding': '11px 0 13px',
            'border-top': '11px solid #202B33',
            'border-bottom': '3px solid #202B33',
            'color': '#202B33',
            'font-size': '21px',
            'font-weight': '850',
            'line-height': '1.48',
        },
        2: {
            'margin': '30px 0 13px',
            'padding': '0 0 8px',
            'border-bottom': '2px solid #202B33',
            'white-space': 'normal',
            'color': '#202B33',
            'font-size': '18px',
            'line-height': '1.5',
            'font-weight': '850',
        },
        3: {
            'margin': '30px 0 13px',
            'padding': '0 0 8px',
            'border-bottom': '2px solid #202B33',
            'white-space': 'normal',
            'color': '#202B33',
            'font-size': '16px',
            'line-height': '1.5',
            'font-weight': '850',
        },
    }

    _PARAGRAPH_STYLE = {
        'margin': '0 0 18px',
        'color': '#202B33',
        'font-size': '16px',
        'line-height': '1.85',
        'text-align': 'justify',
    }

    _CAPTION_STYLE = {
        'color': '#6b7882', 'font-size': '12px', 'line-height': '1.6',
        'margin': '8px 0 20px', 'text-align': 'center',
    }

    _QUOTE_STYLE = {
        'margin': '33px 0',
        'padding': '8px 0 13px',
        'border-top': '11px solid #202B33',
        'border-bottom': '3px solid #202B33',
        'white-space': 'normal',
        'font-size': '17px',
        'line-height': '1.9',
        'color': '#202B33',
    }

    _CODE_STYLE = {
        'background-color': '#EEF2F3',
        'color': '#202B33',
        'margin': '18px 0',
        'padding': '14px',
        'border-left': '3px solid #D6A84B',
        'font-size': '13px',
        'line-height': '1.7',
        'white-space': 'pre-wrap',
        'word-break': 'break-word',
    }

    _DIVIDER_STYLE = {
        'border': '0',
        'border-top': '2px solid #D6A84B',
        'margin': '32px 0',
    }

    _LIST_STYLE = {
        'margin': '22px 0 25px',
        'border-top': '10px solid #202B33',
        'border-bottom': '3px solid #202B33',
        'padding': '10px 12px 2px 16px',
    }

    _ORDERED_MARKER_STYLE = {
        'display': 'inline-block',
        'min-width': '22px',
        'height': '22px',
        'line-height': '22px',
        'margin-left': '-28px',
        'margin-right': '6px',
        'text-align': 'center',
        'vertical-align': 'middle',
        'font-size': '12px',
        'font-weight': '700',
        'color': '#ffffff',
        'background-color': '#243B53',
        'border-radius': '0',
    }

    _UNORDERED_MARKER_STYLE = {
        **_ORDERED_MARKER_STYLE, 'min-width': '0', 'width': '7px', 'height': '7px',
        'margin-left': '-21px', 'margin-right': '14px',
    }

    _BLOCK_STYLES = {
        'container': {
            'background-color': '#FFFDF8',
            'color': '#202B33',
            'padding': '12px 16px',
            'font-size': '16px',
            'line-height': '1.85',
            'word-break': 'break-word',
            'box-sizing': 'border-box',
        },
        'strong': {
            'border-bottom': '5px solid #B85C47',
            'color': '#243B53',
            'font-weight': '850',
        },
        'table': {
            'width': '100%',
            'border-collapse': 'collapse',
            'table-layout': 'fixed',
            'margin': '22px 0',
            'background-color': '#FFFDF8',
        },
        'table_header': {
            'padding': '10px 8px',
            'border-top': '10px solid #202B33',
            'border-bottom': '3px solid #202B33',
            'background-color': '#FFFFFF',
            'color': '#202B33',
            'font-size': '12px',
            'font-weight': '850',
            'line-height': '1.4',
            'text-align': 'left',
        },
        'table_cell': {
            'padding': '11px 8px',
            'border-bottom': '1px solid #202B33',
            'border-left': '5px solid #B85C47',
            'background-color': '#EEF2F3',
            'color': '#202B33',
            'font-size': '13px',
            'line-height': '1.55',
            'vertical-align': 'top',
        },
    }
