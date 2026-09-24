from .base import WeChatTemplate


class StructuredWeChatTemplate(WeChatTemplate):
    template_id = 'structured'
    display_name = 'Structured Blue'

    _HEADING_STYLES: dict[int, dict[str, str]] = {
        1: {
            'background-color': '#2563eb',
            'border-bottom': '3px solid #1746aa',
            'border-radius': '3px',
            'color': '#ffffff',
            'font-size': '20px',
            'font-weight': '700',
            'line-height': '1.5',
            'margin': '26px 0 14px',
            'padding': '9px 13px',
        },
        2: {
            'background-color': '#eff5ff',
            'border-left': '4px solid #2563eb',
            'color': '#1e4f9f',
            'font-size': '18px',
            'font-weight': '700',
            'line-height': '1.5',
            'margin': '22px 0 12px',
            'padding': '6px 10px',
        },
        3: {
            'border-bottom': '1px solid #bfd2f5',
            'color': '#2458a6',
            'font-size': '16px',
            'font-weight': '700',
            'line-height': '1.55',
            'margin': '18px 0 9px',
            'padding-bottom': '4px',
        },
    }

    _PARAGRAPH_STYLE: dict[str, str] = {
        'color': '#374151',
        'font-size': '15px',
        'line-height': '1.82',
        'margin': '0 0 14px',
    }

    _CAPTION_STYLE: dict[str, str] = {
        'color': '#8490a3',
        'font-size': '12px',
        'line-height': '1.5',
        'margin': '6px 0 16px',
        'text-align': 'center',
    }

    _QUOTE_STYLE: dict[str, str] = {
        'background-color': '#f4f7fc',
        'border-left': '4px solid #2563eb',
        'border-radius': '4px',
        'color': '#516078',
        'line-height': '1.75',
        'margin': '16px 0',
        'padding': '11px 14px',
    }

    _CODE_STYLE: dict[str, str] = {
        'background-color': '#f3f5f8',
        'border-radius': '4px',
        'padding': '12px',
    }

    _DIVIDER_STYLE: dict[str, str] = {
        'border': '0',
        'border-top': '2px solid #d7e2f5',
        'margin': '26px 0',
    }

    _UNORDERED_MARKER_STYLE = {
        'background-color': '#2563eb',
        'border-radius': '50%',
        'display': 'inline-block',
        'height': '7px',
        'margin-left': '-23px',
        'margin-right': '16px',
        'vertical-align': 'middle',
        'width': '7px',
    }

    _ORDERED_MARKER_STYLE = {
        'background-color': '#2563eb',
        'border-radius': '3px',
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
