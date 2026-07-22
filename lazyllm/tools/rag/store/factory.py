from typing import Any, Dict, Union

import lazyllm

from .store_base import LazyLLMStoreBase, StoreCapability


def create_segment_store(
    store: Union[Dict[str, Any], LazyLLMStoreBase],
) -> LazyLLMStoreBase:
    '''Construct or validate a segment-capable store without wrapping it.'''

    if isinstance(store, LazyLLMStoreBase):
        impl = store
    elif isinstance(store, dict):
        cfg = dict(store)
        store_type = cfg.get('type')
        if not store_type:
            raise ValueError('Segment store config requires "type"')
        store_cls = getattr(lazyllm.store, store_type, None)
        if store_cls is None:
            raise NotImplementedError(f'Not implemented store type: {store_type}')
        impl = store_cls(**cfg.get('kwargs', {}))
    else:
        raise TypeError('store must be a LazyLLMStoreBase instance or config dict')

    if not isinstance(impl, LazyLLMStoreBase):
        raise TypeError(f'{type(impl).__name__} is not a LazyLLMStoreBase')
    if not impl.capability & StoreCapability.SEGMENT:
        raise ValueError(f'{type(impl).__name__} is not segment-capable')
    return impl
