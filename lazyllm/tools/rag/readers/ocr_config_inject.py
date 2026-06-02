from typing import Any, Dict, Optional

from lazyllm import globals

globals.config.add(
    'dynamic_ocr_configs',
    dict,
    None,
    'DYNAMIC_OCR_CONFIGS',
    description='Per-request OCR config dict for DynamicPDFReader.',
)
globals.config.add(
    'dynamic_ocr_auth',
    dict,
    None,
    'DYNAMIC_OCR_AUTH',
    description='Per-request OCR auth mapping for CredentialMixin-enabled OCR readers.',
)


def inject_ocr_config(ocr_config: Optional[Dict[str, Any]]) -> None:
    if not ocr_config:
        return
    if not isinstance(ocr_config, dict):
        return

    cleaned = {}
    for key in (
        'ocr_type',
        'ocr_url',
        'ocr_dynamic',
        'mineru_upload_mode',
        'mineru_api_key',
        'paddle_api_key',
    ):
        value = ocr_config.get(key)
        if value is not None:
            cleaned[key] = value
    if cleaned:
        globals.config['dynamic_ocr_configs'] = cleaned
    auth_mapping = ocr_config.get('ocr_auth')
    if isinstance(auth_mapping, dict) and auth_mapping:
        globals.config['dynamic_ocr_auth'] = auth_mapping
