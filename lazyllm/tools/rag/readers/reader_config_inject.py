from typing import Any, Dict, Optional

from lazyllm import globals

globals.config.add(
    'dynamic_ocr_configs',
    dict,
    None,
    'DYNAMIC_OCR_CONFIGS',
    description='Per-request OCR routing dict (ocr_type, ocr_url) for DynamicPDFReader.',
)
globals.config.add(
    'dynamic_ocr_auth',
    dict,
    None,
    'DYNAMIC_OCR_AUTH',
    description='Per-request OCR auth mapping for CredentialMixin-enabled OCR readers.',
)


def inject_reader_config(
    *,
    ocr_config: Optional[Dict[str, Any]] = None,
) -> None:
    if not ocr_config or not isinstance(ocr_config, dict):
        return

    cleaned = {}
    for key in ('ocr_type', 'ocr_url'):
        value = ocr_config.get(key)
        if value is not None:
            cleaned[key] = value
    if cleaned:
        globals.config['dynamic_ocr_configs'] = cleaned

    auth_mapping = ocr_config.get('ocr_auth')
    if not isinstance(auth_mapping, dict):
        auth_mapping = {}
    auth_mapping = dict(auth_mapping)
    if ocr_config.get('mineru_api_key'):
        auth_mapping.setdefault('mineru', ocr_config['mineru_api_key'])
    if ocr_config.get('paddle_api_key'):
        auth_mapping.setdefault('paddleocr', ocr_config['paddle_api_key'])
    if auth_mapping:
        globals.config['dynamic_ocr_auth'] = auth_mapping
