# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from typing import Any, Optional

import lazyllm
from lazyllm import config

from .base import LazyLLMFSBase


_PLATFORM_ENV_VARS = {
    'feishu': ('FEISHU_APP_TOKEN', 'FEISHU_TOKEN', 'LARK_TOKEN'),
    'confluence': ('CONFLUENCE_TOKEN', 'ATLASSIAN_TOKEN'),
    'notion': ('NOTION_TOKEN', 'NOTION_INTEGRATION_TOKEN'),
    'googledrive': ('GOOGLE_DRIVE_TOKEN', 'GDRIVE_TOKEN'),
    'onedrive': ('ONEDRIVE_TOKEN', 'MSGRAPH_TOKEN'),
    'yuque': ('YUQUE_TOKEN',),
    'ones': ('ONES_TOKEN',),
    's3': ('AWS_ACCESS_KEY_ID', 'S3_ACCESS_KEY', 'S3_TOKEN'),
    'obsidian': ('OBSIDIAN_VAULT_PATH', 'OBSIDIAN_VAULT'),
}


def _detect_platform_from_env() -> tuple:
    for platform, keys in _PLATFORM_ENV_VARS.items():
        for key in keys:
            value = os.environ.get(key)
            if value and value.strip():
                return platform, value.strip()
    return None, None


def _resolve_token(platform: str, token: Optional[str]) -> Optional[str]:
    if token and token.strip():
        return token.strip()
    for key in _PLATFORM_ENV_VARS.get(platform, ()):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


_S3_SECRET_KEYS = ('AWS_SECRET_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_SECRET')


def _resolve_s3_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    if 'access_key' not in out and 'secret_key' not in out:
        access_key = token or ''
        secret_key = ''
        for key in _S3_SECRET_KEYS:
            val = os.environ.get(key)
            if val and val.strip():
                secret_key = val.strip()
                break
        if not access_key:
            for key in _PLATFORM_ENV_VARS['s3']:
                val = os.environ.get(key)
                if val and val.strip():
                    access_key = val.strip()
                    break
        if access_key:
            out['access_key'] = access_key
        if secret_key:
            out['secret_key'] = secret_key
    return out


config.add('cloudfs_platform', str, None, 'CLOUDFS_PLATFORM',
           description='Default cloud filesystem platform: feishu, confluence, notion, '
                       'googledrive, onedrive, yuque, ones, s3, obsidian. None for auto-detect.')


class CloudFS:

    def __new__(cls, platform: Optional[str] = None, token: Optional[str] = None,
                **kwargs) -> LazyLLMFSBase:
        if not platform:
            platform = config['cloudfs_platform']
        if not platform:
            platform, env_token = _detect_platform_from_env()
            if platform and not token:
                token = env_token
        if not platform:
            raise ValueError(
                'Cannot determine cloud platform.  Pass platform=... or set '
                'CLOUDFS_PLATFORM / one of the per-platform token env vars.'
            )
        platform = platform.lower()
        resolved = _resolve_token(platform, token)
        if platform == 's3':
            kwargs = _resolve_s3_kwargs(resolved, **kwargs)
            return getattr(lazyllm.fs, platform)(token=resolved or '', **kwargs)
        return getattr(lazyllm.fs, platform)(token=resolved, **kwargs)
