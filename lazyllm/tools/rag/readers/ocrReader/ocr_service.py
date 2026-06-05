from enum import Enum
from typing import Optional

OCR_ONLINE_HOSTS = {
    'mineru': 'mineru.net',
    'paddleocr': 'aistudio-app.com',
}

DEFAULT_ONLINE_URLS = {
    'mineru': 'https://mineru.net',
    'paddleocr': 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
}


class OcrServiceVariant(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'


def resolve_ocr_variant(ocr_type: str, url: Optional[str]) -> OcrServiceVariant:
    host = OCR_ONLINE_HOSTS.get(ocr_type)
    if host is None:
        supported = ', '.join(sorted(OCR_ONLINE_HOSTS))
        raise ValueError(f'Unsupported ocr_type: {ocr_type!r}, only support: {supported}')
    if not url or host in url.lower():
        return OcrServiceVariant.ONLINE
    return OcrServiceVariant.OFFLINE


def default_online_url(ocr_type: str) -> str:
    try:
        return DEFAULT_ONLINE_URLS[ocr_type]
    except KeyError:
        supported = ', '.join(sorted(DEFAULT_ONLINE_URLS))
        raise ValueError(f'Unsupported ocr_type: {ocr_type!r}, only support: {supported}')
