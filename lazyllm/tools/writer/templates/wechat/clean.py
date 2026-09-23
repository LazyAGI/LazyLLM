from .base import WeChatTemplate


class CleanWeChatTemplate(WeChatTemplate):
    template_id = 'clean'
    display_name = 'Clean Reading'

    _HEADING_STYLES: dict[int, dict[str, str]] = {
        1: {
            'background-color': '#edf7f3',
            'border-left': '5px solid #2f806d',
            'border-radius': '3px',
            'color': '#20584b',
            'font-size': '20px',
            'font-weight': '700',
            'line-height': '1.6',
            'margin': '26px 0 12px',
            'padding': '8px 12px',
        },
        2: {
            'border-bottom': '1px solid #9bc8bc',
            'color': '#286b5a',
            'font-size': '18px',
            'font-weight': '700',
            'line-height': '1.6',
            'margin': '22px 0 10px',
            'padding-bottom': '5px',
        },
        3: {
            'border-left': '3px solid #9bc8bc',
            'color': '#344054',
            'font-size': '16px',
            'font-weight': '700',
            'line-height': '1.6',
            'margin': '18px 0 8px',
            'padding-left': '9px',
        },
    }

    _PARAGRAPH_STYLE: dict[str, str] = {
        'color': '#33373d',
        'font-size': '16px',
        'line-height': '1.9',
        'margin': '0 0 16px',
    }

    _CAPTION_STYLE: dict[str, str] = {
        'color': '#8a9099',
        'font-size': '12px',
        'line-height': '1.6',
        'margin': '7px 0 18px',
        'text-align': 'center',
    }

    _QUOTE_STYLE = {
        'border-left': '3px solid #d0d5dd',
        'color': '#667085',
        'margin': '16px 0',
        'padding': '4px 0 4px 12px',
    }

    _CODE_STYLE = {
        'background-color': '#f6f7f9',
        'border': '1px solid #e4e7ec',
        'border-radius': '3px',
        'color': '#344054',
        'line-height': '1.6',
        'padding': '12px',
    }

    _DIVIDER_STYLE = {
        'border': '0',
        'border-top': '1px solid #e4e7ec',
        'margin': '24px 0',
    }

    _UNORDERED_MARKER_STYLE = {
        'background-color': '#2f806d',
        'border-radius': '50%',
        'display': 'inline-block',
        'height': '7px',
        'margin-left': '-25px',
        'margin-right': '18px',
        'vertical-align': 'middle',
        'width': '7px',
    }

    _ORDERED_MARKER_STYLE = {
        'border': '1px solid #2f806d',
        'border-radius': '11px',
        'color': '#286b5a',
        'display': 'inline-block',
        'font-size': '12px',
        'height': '22px',
        'line-height': '20px',
        'margin-left': '-28px',
        'margin-right': '6px',
        'min-width': '22px',
        'text-align': 'center',
        'vertical-align': 'middle',
    }
