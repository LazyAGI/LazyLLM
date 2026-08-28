from __future__ import annotations

from typing import Any, Dict, Type

from .base import WriterProviderBase


_PROVIDERS: Dict[str, Type[WriterProviderBase]] = {}


def register_writer_provider(provider_class: Type[WriterProviderBase]) -> None:
    if not issubclass(provider_class, WriterProviderBase):
        raise TypeError('provider_class must inherit WriterProviderBase')
    key = str(provider_class.provider or '').strip().lower()
    if not key:
        raise ValueError('provider_class.provider is required')
    existing = _PROVIDERS.get(key)
    if existing is not None and existing is not provider_class:
        raise ValueError(f'Writer provider {key!r} is already registered.')
    _PROVIDERS[key] = provider_class


def get_writer_provider(provider: str, **kwargs: Any) -> WriterProviderBase:
    key = str(provider or '').strip().lower()
    provider_class = _PROVIDERS.get(key)
    if provider_class is None:
        available = ', '.join(sorted(_PROVIDERS)) or 'none'
        raise ValueError(
            f'No Writer provider is registered for {key!r}. Available providers: {available}.')
    return provider_class(**kwargs)


def match_writer_provider(locator: str, **kwargs: Any) -> WriterProviderBase:
    candidates = [
        provider_class
        for provider_class in _PROVIDERS.values()
        if provider_class.matches(locator)
    ]
    if not candidates:
        raise ValueError(f'No Writer provider matches locator {locator!r}.')
    if len(candidates) > 1:
        names = ', '.join(sorted(provider.provider for provider in candidates))
        raise ValueError(f'Multiple Writer providers match locator {locator!r}: {names}.')
    return candidates[0](**kwargs)


__all__ = [
    'get_writer_provider',
    'match_writer_provider',
    'register_writer_provider',
]
