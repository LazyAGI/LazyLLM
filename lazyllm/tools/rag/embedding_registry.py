import json
import os
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
import sqlalchemy
from lazyllm import LOG, once_wrapper
from lazyllm.tools.sql.sql_manager import SqlManager

@dataclass
class EmbeddingModelInfo:
    embed_key: str
    model_name: str
    dimension: int
    model_provider: str
    data_type: str = 'dense'
    db_type: Optional[str] = None
    db_connection: Optional[str] = None
    collection_name: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, EmbeddingModelInfo):
            return NotImplemented
        return (self.embed_key == other.embed_key and self.model_name == other.model_name
                and self.dimension == other.dimension and self.data_type == other.data_type
                and self.db_type == other.db_type and self.db_connection == other.db_connection
        )


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
        self._auto_save: bool = True
        self._sql_manager: Optional[SqlManager] = None
        self._configure()

    def _configure(self, persistence_path: Optional[str] = None, auto_save: bool = True):
        if persistence_path is None:
            persistence_path = os.getenv('EMBED_REGISTRY_CACHE')
            if not persistence_path:
                persistence_path = os.path.expanduser('~/.lazyllm/cached_embed')

        if not persistence_path.endswith('.db'):
            persistence_path = os.path.join(persistence_path, 'embedding_registry.db')

        self._persistence_path = persistence_path
        self._auto_save = auto_save
        self._init_sql_manager()
        self._load_from_db()

    def _init_sql_manager(self) -> None:
        if not self._persistence_path:
            return

        try:
            db_dir = os.path.dirname(self._persistence_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            columns_spec = [
                ('collection_name', 'string', 'Collection name', True, False, None),
                ('embed_key', 'string', 'Embedding key', False, False, None),
                ('model_name', 'string', 'Model name', False, False, None),
                ('dimension', 'integer', 'Embedding dimension', False, False, None),
                ('model_provider', 'string', 'Model provider', False, False, None),
                ('data_type', 'string', 'Data type', False, False, 'dense'),
                ('db_type', 'string', 'Database type', False, True, None),
                ('db_connection', 'string', 'Database connection', False, True, None),
            ]

            columns = []
            for name, data_type, comment, is_pk, nullable, default in columns_spec:
                col = {
                    'name': name,
                    'data_type': data_type,
                    'comment': comment,
                    'is_primary_key': is_pk,
                    'nullable': nullable
                }
                if default is not None:
                    col['default'] = default
                columns.append(col)

            tables_info_dict = {
                'tables': [{
                    'name': 'embedding_models',
                    'comment': 'Embedding model registry',
                    'columns': columns
                }]
            }

            self._sql_manager = SqlManager(
                db_type='sqlite',
                user='',
                password='',
                host='',
                port=0,
                db_name=self._persistence_path,
                tables_info_dict=tables_info_dict
            )
        except Exception as e:
            LOG.error(f'Failed to initialize SQL manager: {e}')
            self._sql_manager = None

    def validate_and_register(self, store: Any, embed: Dict[str, Callable],
                              embed_dims: Dict[str, int], embed_datatypes: Dict[str, Any],
                              algo_name: str, kb_group_name: str) -> None:
        db_type = self._extract_db_type(store)
        collection_name = gen_collection_name(algo_name, kb_group_name)
        db_connection = self._extract_db_connection(store)

        for embed_key in embed.keys():
            if embed_key not in embed_dims:
                continue

            dimension = embed_dims[embed_key]
            data_type = embed_datatypes.get(embed_key, 'dense')

            model_info = self._extract_model_info(embed[embed_key], embed_key, dimension, data_type)
            model_info.db_type = db_type
            model_info.db_connection = db_connection
            if collection_name in self._registry:
                existing_info = self._registry[collection_name]
                if db_type != existing_info.db_type or db_connection != existing_info.db_connection:
                    return
            self._validate_or_register_model(embed_key, model_info, collection_name)

    def _validate_or_register_model(self, embed_key: str, model_info: EmbeddingModelInfo, collection_name: str) -> None:
        if collection_name in self._registry:
            if model_info == self._registry[collection_name]:
                return
            else:
                self._raise_registration_conflict_error(collection_name, embed_key, model_info,
                                                        existing_info=self._registry[collection_name])
        else:
            model_info.collection_name = collection_name
            self._registry[collection_name] = model_info

            if self._auto_save:
                self.save()

            LOG.info(f'Registered embedding model: {embed_key} ({model_info.model_name}, {model_info.dimension})')

    def _raise_registration_conflict_error(self, collection_name: str, embed_key: str, new_info: EmbeddingModelInfo,
                                           existing_info: EmbeddingModelInfo) -> None:
        error_msg = f'''
================================================================================
Embedding Conflict for collection: {collection_name} on embed key: {embed_key}!
================================================================================
Registered Model Info:
DB Type: {existing_info.db_type}
DB Connection: {existing_info.db_connection}
Collection Name: {existing_info.collection_name}
Dimension: {existing_info.dimension}
Model: {existing_info.model_name}
Provider: {existing_info.model_provider}
--------------------------------------------------------------------------------
New Model Info:
DB Type: {new_info.db_type}
DB Connection: {new_info.db_connection}
Collection Name: {collection_name}
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

    def get(self, collection_name: str) -> Optional[EmbeddingModelInfo]:
        return self._registry.get(collection_name)

    def get_all(self) -> Dict[str, EmbeddingModelInfo]:
        return self._registry.copy()

    def unregister(self, collection_name: str) -> bool:
        if collection_name in self._registry:
            del self._registry[collection_name]
            if self._auto_save:
                self.save()
            return True
        return False

    def clear(self) -> None:
        self._registry.clear()
        if self._auto_save:
            self.save()

    def save(self) -> None:
        if not self._persistence_path or not self._sql_manager:
            return

        try:
            with self._sql_manager.get_session() as session:
                session.execute(sqlalchemy.text('DELETE FROM embedding_models'))

                if self._registry:
                    TableCls = self._sql_manager.get_table_orm_class('embedding_models')
                    if TableCls:
                        for info in self._registry.values():
                            obj = TableCls(**asdict(info))
                            session.add(obj)
        except Exception as e:
            LOG.error(f'Failed to save embedding registry: {e}')

    def _load_from_db(self) -> None:
        if not self._persistence_path or not self._sql_manager:
            return

        try:
            result_str = self._sql_manager.execute_query('SELECT * FROM embedding_models')

            if result_str.startswith('Execute SQL ERROR:'):
                return

            data = json.loads(result_str)

            for row in data:
                collection_name = row.get('collection_name')
                if collection_name:
                    info = EmbeddingModelInfo(**row)
                    self._registry[collection_name] = info
        except Exception as e:
            LOG.error(f'Failed to load embedding registry: {e}')

def get_embedding_registry() -> EmbeddingModelRegistry:
    return EmbeddingModelRegistry()

def extract_model_info(embed_func: Any, embed_key: str, dimension: int, data_type: str = 'dense') -> EmbeddingModelInfo:
    registry = get_embedding_registry()
    return registry._extract_model_info(embed_func, embed_key, dimension, data_type)

def gen_collection_name(algo_name: str, kb_group_name: str) -> str:
    return f'col_{algo_name}_{kb_group_name}'.lower()
