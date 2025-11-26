import json
import os
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
from lazyllm import LOG, once_wrapper

@dataclass
class EmbeddingModelInfo:
    embed_key: str
    model_name: str
    dimension: int
    model_provider: str
    data_type: str = 'dense'
    db_type: Optional[str] = None
    db_connection: Optional[str] = None


class EmbeddingModelRegistry:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @once_wrapper(reset_on_pickle=True)
    def __init__(self):
        self._registry: Dict[str, EmbeddingModelInfo] = {}
        self._persistence_path: Optional[str] = None
        self._auto_save: bool = False

    def configure(self, persistence_path: Optional[str] = None, auto_save: bool = False):
        self._persistence_path = persistence_path
        self._auto_save = auto_save
        if persistence_path and os.path.exists(persistence_path):
            self._load_from_file()

    def validate_and_register(self, store: Any, embed: Dict[str, Callable],
                              embed_dims: Dict[str, int], embed_datatypes: Dict[str, Any]) -> None:
        db_type = self._extract_db_type(store)

        db_connection = self._extract_db_connection(store)

        for embed_key in embed.keys():
            if embed_key not in embed_dims:
                continue

            dimension = embed_dims[embed_key]
            data_type = embed_datatypes.get(embed_key, 'dense')

            model_info = self._extract_model_info(embed[embed_key], embed_key, dimension, data_type)

            self._validate_or_register_model(embed_key, model_info, db_type, db_connection)

    def _validate_or_register_model(self, embed_key: str, model_info: EmbeddingModelInfo,
                                    db_type: Optional[str], db_connection: Optional[str]) -> None:
        model_info.db_type = db_type
        model_info.db_connection = db_connection
        if embed_key in self._registry:
            existing_info = self._registry[embed_key]

            if existing_info.db_type == db_type and existing_info.db_connection == db_connection:
                if not self._check_embed_key_model_compatibility(embed_key, model_info, existing_info):
                    self._raise_registration_conflict_error(embed_key, model_info, existing_info)

        else:
            self._registry[embed_key] = model_info

            if self._auto_save:
                self.save()

            LOG.info(f'Registered embedding model: {embed_key} ({model_info.model_name}, {model_info.dimension})')

    def _check_embed_key_model_compatibility(self, embed_key: str, new_info: EmbeddingModelInfo,
                                             existing_info: EmbeddingModelInfo) -> bool:
        return (new_info.model_name == existing_info.model_name
                and new_info.dimension == existing_info.dimension)

    def _raise_registration_conflict_error(self, embed_key: str, new_info: EmbeddingModelInfo,
                                           existing_info: EmbeddingModelInfo) -> None:
        error_msg = f'''
================================================================================
Embedding Model Registration Conflict!
================================================================================
Embed Key: "{embed_key}"
Registered Model Info:
DB Type: {existing_info.db_type}
DB Connection: {existing_info.db_connection}
Dimension: {existing_info.dimension}
Model: {existing_info.model_name}
Provider: {existing_info.model_provider}
--------------------------------------------------------------------------------
New Model Info:
DB Type: {new_info.db_type}
DB Connection: {new_info.db_connection}
Dimension: {new_info.dimension}
Model: {new_info.model_name}
Provider: {new_info.model_provider}

This error indicates you're trying to use different embedding models
or database configurations with the same embed_key.

Try with:
1. Use the same embedding model and database configuration (recommended)
2. Use a different embed_key for the new model/configuration
3. Clear the registry if starting fresh
================================================================================
'''
        LOG.error('\n' + error_msg.strip())
        raise ValueError

    def _extract_db_type(self, store: Any) -> Optional[str]:
        try:
            if isinstance(store, dict):
                return store.get('type')

            if hasattr(store, 'type'):
                return store.type
            elif hasattr(store, '__class__'):
                return store.__class__.__name__.lower()
            else:
                return str(type(store)).lower()
        except Exception:
            return None

    def _extract_db_connection(self, store: Any) -> Optional[str]:
        try:
            if isinstance(store, dict):
                kwargs = store.get('kwargs', {})
                connection_attrs = ['uri', 'url', 'host', 'address', 'connection_string', 'db_path']
                for attr in connection_attrs:
                    if attr in kwargs:
                        value = kwargs[attr]
                        if value:
                            return str(value)
                return None

            connection_attrs = ['uri', 'url', 'host', 'address', 'connection_string']
            for attr in connection_attrs:
                if hasattr(store, attr):
                    value = getattr(store, attr)
                    if value:
                        return str(value)
            return None
        except Exception:
            return None

    def _extract_model_info(self, embed_func: Callable, embed_key: str, dimension: int,   # noqa: C901
                            data_type: str) -> EmbeddingModelInfo:
        try:
            model_name = 'unknown'
            provider = 'unknown'

            if hasattr(embed_func, '_model_series') and hasattr(embed_func, '_embed_model_name'):
                provider = getattr(embed_func, '_model_series', 'unknown')
                model_name = getattr(embed_func, '_embed_model_name', 'unknown')

                if provider == 'unknown' and hasattr(embed_func, 'series'):
                    provider = embed_func.series
                if model_name == 'unknown':
                    model_name = getattr(embed_func, 'embed_model_name', 'unknown')

                if provider != 'unknown':
                    provider = str(provider).lower()

            elif hasattr(embed_func, '__class__') and 'TrainableModule' in str(embed_func.__class__):
                if hasattr(embed_func, 'series'):
                    try:
                        provider = embed_func.series.lower()
                    except Exception:
                        provider = 'trainable'
                else:
                    provider = 'trainable'

                if hasattr(embed_func, 'base_model'):
                    try:
                        model_name = embed_func.base_model
                        if model_name and ('/' in model_name or '\\' in model_name):
                            model_name = os.path.basename(model_name)
                    except Exception:
                        model_name = 'unknown'
                if model_name == 'unknown' and hasattr(embed_func, '_impl'):
                    impl = embed_func._impl
                    if hasattr(impl, '_base_model'):
                        model_name = impl._base_model
                        if model_name and ('/' in model_name or '\\' in model_name):
                            model_name = os.path.basename(model_name)

            else:
                if hasattr(embed_func, 'model_name'):
                    model_name = embed_func.model_name
                elif hasattr(embed_func, '_model_name'):
                    model_name = embed_func._model_name
                elif hasattr(embed_func, 'name'):
                    model_name = embed_func.name
                elif hasattr(embed_func, '__name__'):
                    model_name = embed_func.__name__

                if hasattr(embed_func, 'source'):
                    provider = embed_func.source
                elif hasattr(embed_func, '__class__'):
                    class_name = embed_func.__class__.__name__
                    for suffix in ['Embedding', 'Module', 'Base']:
                        if class_name.endswith(suffix):
                            class_name = class_name[:-len(suffix)]
                    provider = class_name.lower() if class_name else 'unknown'

            if model_name is None or model_name == '':
                model_name = 'unknown'
            if provider is None or provider == '':
                provider = 'unknown'

            return EmbeddingModelInfo(
                embed_key=embed_key,
                model_name=str(model_name),
                dimension=dimension,
                model_provider=str(provider),
                data_type=data_type
            )
        except Exception as e:
            LOG.warning(f'Failed to extract model info for {embed_key}: {e}')
            return EmbeddingModelInfo(
                embed_key=embed_key,
                model_name='unknown',
                dimension=dimension,
                model_provider='unknown',
                data_type=data_type
            )

    def get(self, embed_key: str) -> Optional[EmbeddingModelInfo]:
        return self._registry.get(embed_key)

    def get_all(self) -> Dict[str, EmbeddingModelInfo]:
        return self._registry.copy()

    def unregister(self, embed_key: str) -> bool:
        if embed_key in self._registry:
            del self._registry[embed_key]
            if self._auto_save:
                self.save()
            return True
        return False

    def clear(self) -> None:
        self._registry.clear()
        if self._auto_save:
            self.save()

    def save(self) -> None:
        if not self._persistence_path:
            return

        try:
            data = {
                embed_key: asdict(info)
                for embed_key, info in self._registry.items()
            }
            os.makedirs(os.path.dirname(self._persistence_path), exist_ok=True)
            with open(self._persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            LOG.error(f'Failed to save embedding registry: {e}')

    def _load_from_file(self) -> None:
        if not self._persistence_path or not os.path.exists(self._persistence_path):
            return

        try:
            with open(self._persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for embed_key, info_dict in data.items():
                info = EmbeddingModelInfo(**info_dict)
                self._registry[embed_key] = info
        except Exception as e:
            LOG.error(f'Failed to load embedding registry: {e}')


def get_embedding_registry() -> EmbeddingModelRegistry:
    return EmbeddingModelRegistry()


def extract_model_info(embed_func: Any, embed_key: str, dimension: int, data_type: str = 'dense') -> EmbeddingModelInfo:
    registry = get_embedding_registry()
    return registry._extract_model_info(embed_func, embed_key, dimension, data_type)
