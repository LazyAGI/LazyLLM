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


def _get_from_env_or_config(param: str, env_keys: tuple, config_key: Optional[str] = None) -> Optional[str]:
    for key in env_keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    if config_key is not None:
        try:
            val = config[config_key]
            if val is not None and str(val).strip():
                return str(val).strip()
        except KeyError:
            pass
    return None


_S3_SECRET_KEYS = ('AWS_SECRET_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_SECRET')


def _s3_secret_from_env() -> str:
    for key in _S3_SECRET_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    return ''


def _s3_access_key_from_env(token: Optional[str]) -> str:
    if token and token.strip():
        return token.strip()
    for key in _PLATFORM_ENV_VARS['s3']:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    return ''


def _resolve_s3_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    if 'token' not in out:
        out['token'] = token or ''
    if 'access_key' not in out and 'secret_key' not in out:
        access_key = _s3_access_key_from_env(token)
        secret_key = _s3_secret_from_env()
        if access_key:
            out['access_key'] = access_key
        if secret_key:
            out['secret_key'] = secret_key
    if 'endpoint_url' not in out:
        v = _get_from_env_or_config(
            'endpoint_url',
            ('CLOUDFS_S3_ENDPOINT_URL',), 'cloudfs_s3_endpoint_url',
        )
        if v:
            out['endpoint_url'] = v
    if 'region_name' not in out:
        v = _get_from_env_or_config(
            'region_name',
            ('AWS_REGION', 'CLOUDFS_S3_REGION_NAME'), 'cloudfs_s3_region_name',
        )
        if v:
            out['region_name'] = v
    return out


def _resolve_feishu_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    if 'app_id' not in out:
        v = _get_from_env_or_config(
            'app_id',
            ('FEISHU_APP_ID', 'CLOUDFS_FEISHU_APP_ID'), 'cloudfs_feishu_app_id',
        )
        if v:
            out['app_id'] = v
    if 'app_secret' not in out:
        v = _get_from_env_or_config(
            'app_secret',
            ('FEISHU_APP_SECRET', 'CLOUDFS_FEISHU_APP_SECRET'), 'cloudfs_feishu_app_secret',
        )
        if v:
            out['app_secret'] = v
    return out


def _resolve_googledrive_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    if 'credentials' not in out:
        v = _get_from_env_or_config(
            'credentials',
            ('GOOGLE_DRIVE_CREDENTIALS', 'GDRIVE_CREDENTIALS', 'CLOUDFS_GOOGLEDRIVE_CREDENTIALS'),
            'cloudfs_googledrive_credentials',
        )
        if v:
            out['credentials'] = v
    return out


def _resolve_onedrive_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    for param, env_names, config_key in (
        ('client_id', ('ONEDRIVE_CLIENT_ID', 'MSGRAPH_CLIENT_ID', 'CLOUDFS_ONEDRIVE_CLIENT_ID'),
         'cloudfs_onedrive_client_id'),
        ('client_secret', ('ONEDRIVE_CLIENT_SECRET', 'MSGRAPH_CLIENT_SECRET', 'CLOUDFS_ONEDRIVE_CLIENT_SECRET'),
         'cloudfs_onedrive_client_secret'),
        ('tenant_id', ('ONEDRIVE_TENANT_ID', 'MSGRAPH_TENANT_ID', 'CLOUDFS_ONEDRIVE_TENANT_ID'),
         'cloudfs_onedrive_tenant_id'),
    ):
        if param not in out:
            v = _get_from_env_or_config(param, env_names, config_key)
            if v:
                out[param] = v
    return out


def _resolve_confluence_kwargs(token: Optional[str], **kwargs: Any) -> dict:
    out = dict(kwargs)
    if 'token' not in out:
        out['token'] = token
    for param, env_names, config_key in (
        ('email', ('CONFLUENCE_EMAIL', 'ATLASSIAN_EMAIL', 'CLOUDFS_CONFLUENCE_EMAIL'),
         'cloudfs_confluence_email'),
        ('cloud_id', ('CONFLUENCE_CLOUD_ID', 'ATLASSIAN_CLOUD_ID', 'CLOUDFS_CONFLUENCE_CLOUD_ID'),
         'cloudfs_confluence_cloud_id'),
    ):
        if param not in out:
            v = _get_from_env_or_config(param, env_names, config_key)
            if v:
                out[param] = v
    return out


def _resolve_platform_kwargs(platform: str, token: Optional[str], **kwargs: Any) -> dict:
    resolvers = {
        'feishu': _resolve_feishu_kwargs,
        's3': _resolve_s3_kwargs,
        'googledrive': _resolve_googledrive_kwargs,
        'onedrive': _resolve_onedrive_kwargs,
        'confluence': _resolve_confluence_kwargs,
    }
    resolver = resolvers.get(platform)
    if resolver:
        return resolver(token, **kwargs)
    out = dict(kwargs)
    if 'token' not in out:
        out['token'] = token
    return out


config.add('cloudfs_platform', str, None, 'CLOUDFS_PLATFORM',
           description='Default cloud filesystem platform: feishu, confluence, notion, '
                       'googledrive, onedrive, yuque, ones, s3, obsidian. None for auto-detect.')

# Per-platform config (env CLOUDFS_<PLATFORM>_<KEY> and config key cloudfs_<platform>_<key>)
config.add('cloudfs_feishu_app_id', str, None, 'CLOUDFS_FEISHU_APP_ID',
           description='Feishu App ID for tenant_access_token (preferred over token).')
config.add('cloudfs_feishu_app_secret', str, None, 'CLOUDFS_FEISHU_APP_SECRET',
           description='Feishu App Secret for tenant_access_token.')
config.add('cloudfs_googledrive_credentials', str, None, 'CLOUDFS_GOOGLEDRIVE_CREDENTIALS',
           description='Path to Google service account JSON or env var for credentials.')
config.add('cloudfs_onedrive_client_id', str, None, 'CLOUDFS_ONEDRIVE_CLIENT_ID',
           description='OneDrive/Microsoft Graph app client ID.')
config.add('cloudfs_onedrive_client_secret', str, None, 'CLOUDFS_ONEDRIVE_CLIENT_SECRET',
           description='OneDrive/Microsoft Graph app client secret.')
config.add('cloudfs_onedrive_tenant_id', str, None, 'CLOUDFS_ONEDRIVE_TENANT_ID',
           description='OneDrive/Microsoft Graph tenant ID (default: common).')
config.add('cloudfs_confluence_email', str, None, 'CLOUDFS_CONFLUENCE_EMAIL',
           description='Confluence cloud login email for Basic auth.')
config.add('cloudfs_confluence_cloud_id', str, None, 'CLOUDFS_CONFLUENCE_CLOUD_ID',
           description='Confluence cloud instance ID for base URL.')
config.add('cloudfs_s3_endpoint_url', str, None, 'CLOUDFS_S3_ENDPOINT_URL',
           description='S3-compatible endpoint URL (optional).')
config.add('cloudfs_s3_region_name', str, None, 'CLOUDFS_S3_REGION_NAME',
           description='S3 region name (optional).')


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
        resolved_token = _resolve_token(platform, token)
        kwargs = _resolve_platform_kwargs(platform, resolved_token, **kwargs)
        return getattr(lazyllm.fs, platform)(**kwargs)
