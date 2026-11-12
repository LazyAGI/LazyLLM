# Adapted from the light_reading theme in https://github.com/zhouke0929/wechat-visual-director (Apache-2.0).
# Source revision 9b0d47cdb456c371c8ba0fb4dcc7e0d1955944b0.
# Modified for Writer's template API and mobile layout.
from .base import WeChatTemplate


class CleanWeChatTemplate(WeChatTemplate):
    template_id = 'clean'
    display_name = 'Clean Reading'

    _HEADING_STYLES: dict[int, dict[str, str]] = {
        1: {
            'border-left': '3px solid #52766F',
            'border-bottom': '1px solid #8BB9C0',
            'color': '#283532',
            'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
            'font-size': '21px',
            'font-weight': '700',
            'line-height': '1.5',
            'margin': '36px 0 18px',
            'padding': '3px 0 10px 13px',
        },
        2: {
            'border-bottom': '1px solid #DDBD61',
            'color': '#263331',
            'font-size': '18px',
            'font-weight': '700',
            'line-height': '1.55',
            'margin': '27px 0 12px',
            'padding-bottom': '7px',
        },
        3: {
            'border-left': '3px solid #8BB9C0',
            'color': '#52766F',
            'font-size': '16px',
            'font-weight': '700',
            'line-height': '1.6',
            'margin': '20px 0 10px',
            'padding-left': '10px',
        },
    }

    _PARAGRAPH_STYLE: dict[str, str] = {
        'color': '#34403E',
        'font-size': '16px',
        'line-height': '1.85',
        'margin': '0 0 18px',
        'text-align': 'justify',
    }

    _CAPTION_STYLE: dict[str, str] = {
        'color': '#75817C',
        'font-size': '12px',
        'line-height': '1.6',
        'margin': '8px 0 20px',
        'text-align': 'center',
    }

    _QUOTE_STYLE = {
        'border-bottom': '1px solid #8BB9C0',
        'color': '#1F2B29',
        'font-family': 'Georgia,"Noto Serif SC","Songti SC",serif',
        'font-size': '17px',
        'font-weight': '600',
        'line-height': '1.9',
        'margin': '28px 0 30px',
        'padding': '8px 0 14px',
        'white-space': 'normal',
    }

    _CODE_STYLE = {
        'background-color': '#F1F7F5',
        'border-left': '3px solid #8BB9C0',
        'border-radius': '6px',
        'box-sizing': 'border-box',
        'color': '#34403E',
        'font-family': 'ui-monospace,SFMono-Regular,Menlo,Consolas,monospace',
        'font-size': '13px',
        'line-height': '1.7',
        'margin': '20px 0',
        'max-width': '100%',
        'overflow-x': 'auto',
        'padding': '14px',
        'white-space': 'pre',
    }

    _DIVIDER_STYLE = {
        'border': '0',
        'border-top': '1px solid #8BB9C0',
        'margin': '32px auto',
        'width': '70%',
    }

    _LIST_STYLE = {
        'border-left': '1px dotted #8BB9C0',
        'margin': '20px 0 24px',
        'padding': '6px 12px 2px 16px',
    }

    _UNORDERED_MARKER_STYLE = {
        'background-color': '#52766F',
        'border-radius': '6px 1px 6px 1px',
        'display': 'inline-block',
        'height': '7px',
        'margin-left': '-25px',
        'margin-right': '18px',
        'vertical-align': 'middle',
        'width': '7px',
    }

    _ORDERED_MARKER_STYLE = {
        'background-color': '#52766F',
        'border-radius': '12px 2px 12px 2px',
        'color': '#ffffff',
        'display': 'inline-block',
        'font-size': '12px',
        'font-weight': '700',
        'height': '22px',
        'line-height': '22px',
        'margin-left': '-28px',
        'margin-right': '6px',
        'min-width': '22px',
        'text-align': 'center',
        'vertical-align': 'middle',
    }

    _BLOCK_STYLES = {
        'container': {
            'background-color': '#FFFFFF',
            'color': '#34403E',
            'padding': '12px 16px',
            'font-size': '16px',
            'line-height': '1.85',
            'text-align': 'justify',
            'word-break': 'break-word',
            'box-sizing': 'border-box',
        },
        'strong': {
            'padding': '0 3px',
            'border-bottom': '5px solid #D98268',
            'color': '#52766F',
            'font-weight': '800',
        },
        'table': {
            'width': '100%',
            'border-collapse': 'collapse',
            'table-layout': 'fixed',
            'margin': '22px 0',
            'background-color': '#FFFFFF',
        },
        'table_header': {
            'padding': '10px 8px',
            'border': '1px solid #52766F',
            'background-color': '#52766F',
            'color': '#ffffff',
            'font-size': '13px',
            'line-height': '1.5',
            'text-align': 'left',
        },
        'table_cell': {
            'padding': '10px 8px',
            'border': '1px solid #D9DED9',
            'background-color': '#FFFFFF',
            'color': '#34403E',
            'font-size': '13px',
            'line-height': '1.6',
            'vertical-align': 'top',
        },
    }
