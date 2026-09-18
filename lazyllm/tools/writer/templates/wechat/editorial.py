from .base import WeChatTemplate


class EditorialWeChatTemplate(WeChatTemplate):
    template_id = 'editorial'
    display_name = 'Warm Editorial'

    _HEADING_STYLES: dict[int, dict[str, str]] = {
        1: {
            'background-color': '#faf2ec',
            'border-left': '3px solid #b9795f',
            'border-right': '3px solid #b9795f',
            'color': '#6b3131',
            'font-size': '21px',
            'font-weight': '700',
            'line-height': '1.55',
            'margin': '30px 0 16px',
            'padding': '9px 12px',
            'text-align': 'center',
        },
        2: {
            'border-bottom': '1px solid #d7b3a3',
            'color': '#763f32',
            'font-size': '18px',
            'font-weight': '700',
            'line-height': '1.55',
            'margin': '24px 0 12px',
            'padding-bottom': '6px',
        },
        3: {
            'border-left': '3px solid #d7b3a3',
            'color': '#8a5545',
            'font-size': '16px',
            'font-weight': '700',
            'line-height': '1.6',
            'margin': '20px 0 9px',
            'padding-left': '9px',
        },
    }

    _PARAGRAPH_STYLE: dict[str, str] = {
        'color': '#403a36',
        'font-size': '16px',
        'line-height': '1.95',
        'margin': '0 0 16px',
    }

    _CAPTION_STYLE: dict[str, str] = {
        'color': '#9a8174',
        'font-size': '12px',
        'line-height': '1.6',
        'margin': '7px 0 18px',
        'text-align': 'center',
    }

    _QUOTE_STYLE: dict[str, str] = {
        'background-color': '#fbf6ef',
        'border-left': '4px solid #b9795f',
        'border-radius': '4px',
        'color': '#6b5449',
        'line-height': '1.85',
        'margin': '18px 0',
        'padding': '12px 14px',
    }

    _CODE_STYLE: dict[str, str] = {
        'background-color': '#302a28',
        'border-radius': '4px',
        'color': '#f8f4ef',
        'line-height': '1.65',
        'padding': '14px',
    }

    _DIVIDER_STYLE: dict[str, str] = {
        'border': '0',
        'border-top': '1px solid #caa98e',
        'margin': '28px 0',
    }

    _UNORDERED_MARKER_STYLE = {
        'background-color': '#ffffff',
        'border': '2px solid #b9795f',
        'border-radius': '50%',
        'display': 'inline-block',
        'height': '7px',
        'margin-left': '-27px',
        'margin-right': '16px',
        'vertical-align': 'middle',
        'width': '7px',
    }

    _ORDERED_MARKER_STYLE = {
        'border': '1px solid #b9795f',
        'border-radius': '11px',
        'color': '#8a5545',
        'display': 'inline-block',
        'font-size': '12px',
        'font-weight': '700',
        'height': '22px',
        'line-height': '20px',
        'margin-left': '-28px',
        'margin-right': '6px',
        'min-width': '22px',
        'text-align': 'center',
        'vertical-align': 'middle',
    }
