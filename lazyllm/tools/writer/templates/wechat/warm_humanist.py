# Adapted from https://github.com/zhouke0929/wechat-visual-director (Apache-2.0).
# Source revision 9b0d47cdb456c371c8ba0fb4dcc7e0d1955944b0.
# Modified for Writer's template API and mobile layout.
from .base import WeChatTemplate


class WarmHumanistWeChatTemplate(WeChatTemplate):
    template_id = 'warm_humanist'
    display_name = 'Warm Humanist'

    _HEADING_STYLES = {
        1: {
            'margin': '44px 0 22px',
            'padding': '3px 0 12px 17px',
            'border-left': '5px solid #D7A83E',
            'border-bottom': '1px solid #D7B995',
            'color': '#342B28',
            'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
            'font-size': '22px',
            'line-height': '1.52',
        },
        2: {
            'margin': '29px 0 13px',
            'white-space': 'normal',
            'box-sizing': 'border-box',
            'padding': '0 0 8px',
            'border-bottom': '1px dashed #D7B995',
            'color': '#342B28',
            'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
            'font-size': '18px',
            'line-height': '1.58',
        },
        3: {
            'margin': '29px 0 13px',
            'white-space': 'normal',
            'box-sizing': 'border-box',
            'padding': '0 0 8px',
            'border-bottom': '1px dashed #D7B995',
            'color': '#342B28',
            'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
            'font-size': '16px',
            'line-height': '1.58',
        },
    }

    _PARAGRAPH_STYLE = {
        'margin': '0 0 18px',
        'color': '#342B28',
        'font-size': '16px',
        'line-height': '1.85',
        'text-align': 'justify',
    }

    _CAPTION_STYLE = {
        'color': '#8a7466', 'font-size': '12px', 'line-height': '1.6',
        'margin': '8px 0 20px', 'text-align': 'center',
    }

    _QUOTE_STYLE = {
        'margin': '33px 4px 32px',
        'padding': '8px 13px 13px 17px',
        'border-bottom': '1px solid #D7B995',
        'white-space': 'normal',
        'box-sizing': 'border-box',
        'border-left': '3px solid #D66B4D',
        'background-color': '#FFF9EF',
        'box-shadow': '7px 7px 0 #FBF2D8',
        'color': '#342B28',
        'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
        'font-size': '17px',
        'font-weight': '650',
        'line-height': '1.9',
    }

    _CODE_STYLE = {
        'background-color': '#F8EEE9',
        'color': '#342B28',
        'margin': '18px 0',
        'padding': '14px',
        'border-left': '3px solid #D7A83E',
        'font-size': '13px',
        'line-height': '1.7',
        'white-space': 'pre-wrap',
        'word-break': 'break-word',
    }

    _DIVIDER_STYLE = {
        'border': '0',
        'border-top': '2px solid #D7A83E',
        'margin': '32px 0',
    }

    _LIST_STYLE = {
        'margin': '22px 0 25px',
        'padding': '10px 12px 2px 16px',
        'border-left': '3px dotted #D7A83E',
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
        'background-color': '#8E4B3B',
        'border-radius': '12px',
    }

    _UNORDERED_MARKER_STYLE = {
        **_ORDERED_MARKER_STYLE, 'min-width': '0', 'width': '7px', 'height': '7px',
        'margin-left': '-21px', 'margin-right': '14px',
    }

    _BLOCK_STYLES = {
        'container': {
            'background-color': '#FFFCF7',
            'color': '#342B28',
            'padding': '12px 16px',
            'font-size': '16px',
            'line-height': '1.85',
            'word-break': 'break-word',
            'box-sizing': 'border-box',
        },
        'strong': {
            'padding': '1px 4px',
            'border-bottom': '7px solid #FBF2D8',
            'color': '#D66B4D',
            'font-weight': '800',
        },
        'table': {
            'width': '100%',
            'border-collapse': 'collapse',
            'table-layout': 'fixed',
            'margin': '22px 0',
            'background-color': '#FFFCF7',
        },
        'table_header': {
            'padding': '10px 8px',
            'border-bottom': '2px solid #8E4B3B',
            'background-color': '#FFF8ED',
            'color': '#8E4B3B',
            'font-family': "Georgia,'Noto Serif SC',serif",
            'font-size': '12px',
            'line-height': '1.45',
            'text-align': 'left',
        },
        'table_cell': {
            'padding': '11px 8px',
            'border-bottom': '1px dashed #D7B995',
            'background-color': '#FBF2D8',
            'color': '#342B28',
            'font-size': '13px',
            'line-height': '1.58',
            'vertical-align': 'top',
        },
    }
