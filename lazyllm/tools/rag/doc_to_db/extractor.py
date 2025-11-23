from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

import sqlalchemy
import json
from uuid import uuid4
from pydantic import BaseModel
from datetime import datetime

from lazyllm import LOG, OnlineChatModule, TrainableModule, once_wrapper
from ..doc_node import DocNode
from ..global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.sql.sql_manager import DBStatus, SqlManager
from .model import (
    TABLE_SCHEMA_SET_INFO, Table_ALGO_KB_SCHEMA, ExtractionMode,
    _TableBase, ExtractResult
)
from ..utils import DocListManager


class SchemaExtractor:
    """Schema aware extractor that materializes BaseModel schemas into database tables."""

    TABLE_PREFIX = 'lazyllm_schema'
    SYS_KB_ID = 'kb_id'
    SYS_DOC_ID = 'doc_id'
    SYS_ALGO_ID = 'algo_id'

    TYPE_MAP = {
        str: sqlalchemy.Text,
        int: sqlalchemy.Integer,
        float: sqlalchemy.Float,
        bool: sqlalchemy.Boolean,
        list: sqlalchemy.JSON,
        dict: sqlalchemy.JSON,
    }

    def __init__(self, db_config: Dict[str, Any],
                 llm: Union[OnlineChatModule, TrainableModule] = OnlineChatModule(),
                 *, table_prefix: Optional[str] = None, force_refresh: bool = False,
                 extraction_mode: ExtractionMode = ExtractionMode.TEXT):
        self._llm = llm
        self._table_prefix = table_prefix or self.TABLE_PREFIX
        self._sql_manager = None
        self._db_config = db_config
        self._table_cache: Dict[str, Type[_TableBase]] = {}
        self._schema_registry: Dict[str, Type[BaseModel]] = {}
        self._force_refresh = force_refresh
        self._extraction_mode = extraction_mode

    @property
    def sql_manager(self) -> SqlManager:
        return self._sql_manager

    @once_wrapper
    def _lazy_init(self):
        self._sql_manager = self._init_sql_manager(self._db_config) if self._db_config else None
        if self._sql_manager:
            self._ensure_management_tables()

    def register_schema_set(self, schema_set: Type[BaseModel], schema_set_id: str = None) -> str:
        """schema set registration, idempotent"""
        try:
            self._lazy_init()
            self._validate_schema_model(schema_set)

            fields = getattr(schema_set, 'model_fields', None) or getattr(schema_set, '__fields__', {})
            signature = [
                (name, str(getattr(f, 'annotation', None) or getattr(f, 'outer_type_', None)),
                 getattr(f, 'default', None), getattr(f, 'default_factory', None) is not None,
                 getattr(f, 'is_required', lambda: False)())
                for name, f in fields.items()
            ]
            signature.sort(key=lambda x: x[0])
            idem_key = json.dumps(signature, ensure_ascii=False)

            if self._sql_manager:
                table_cls = self._sql_manager.get_table_orm_class(TABLE_SCHEMA_SET_INFO['name'])
                if table_cls is None:
                    raise ValueError('Schema set table not initialized')
                with self._sql_manager.get_session() as session:
                    existing = session.query(table_cls).filter_by(idem_key=idem_key).first()
                    if existing:
                        existing_id = str(getattr(existing, 'schema_set_id', getattr(existing, 'id')))
                        if schema_set_id and str(schema_set_id) != existing_id:
                            raise ValueError(
                                f'schema_set_id mismatch for idem_key, expect {existing_id}, got {schema_set_id}'
                            )
                        schema_set_id = schema_set_id or existing_id
                    else:
                        schema_json = (schema_set.model_json_schema() if hasattr(schema_set, 'model_json_schema')
                                       else schema_set.schema())
                        desc = (schema_set.__doc__ or '').strip() or 'Schema set'
                        obj_kwargs = dict(schema_set_json=json.dumps(schema_json, ensure_ascii=False),
                                          desc=desc, idem_key=idem_key, created_at=datetime.now(),
                                          updated_at=datetime.now())
                        if schema_set_id is None:
                            schema_set_id = str(uuid4())
                        obj_kwargs['schema_set_id'] = str(schema_set_id)
                        new_obj = table_cls(**obj_kwargs)
                        session.add(new_obj)
                        session.flush()
                        schema_set_id = str(getattr(new_obj, 'schema_set_id', getattr(new_obj, 'id')))

            if schema_set_id is None:
                raise ValueError('schema_set_id is required and could not be derived')

            schema_set_id = str(schema_set_id)
            self._schema_registry[schema_set_id] = schema_set
            if self._sql_manager:
                self._ensure_table(schema_set_id, schema_set)
            return schema_set_id
        except Exception as e:
            LOG.error(f'Failed to register schema set: {e}')
            raise e

    def has_schema_set(self, schema_set_id: str) -> bool:
        self._lazy_init()
        if self._sql_manager:
            table_cls = self._sql_manager.get_table_orm_class(TABLE_SCHEMA_SET_INFO['name'])
            if table_cls is None:
                raise ValueError('Schema set table not initialized')
            with self._sql_manager.get_session() as session:
                existing = session.query(table_cls).filter_by(schema_set_id=schema_set_id).first()
                return True if existing else False

    def bind_kb_to_schema_set(self, algo_id: str, kb_id: str, schema_set_id: Optional[str] = None,
                              schema_set: Type[BaseModel] = None, force_refresh: bool = False) -> str:
        """
        Bind a KB to a schema set.

        This is used to ensure that the KB is compatible with the schema set.
        """
        try:
            self._lazy_init()
            if not schema_set_id and not schema_set:
                raise ValueError('schema_set_id or schema_set is required')

            # register
            if schema_set is not None:
                schema_set_id = self.register_schema_set(schema_set, schema_set_id)
            else:
                if not self.has_schema_set(schema_set_id):
                    raise ValueError(f'schema_set_id {schema_set_id} not found')

            # ensure table
            self._ensure_table(schema_set_id, self._schema_registry[schema_set_id])

            bind_table_cls = self._sql_manager.get_table_orm_class(Table_ALGO_KB_SCHEMA['name'])
            if bind_table_cls is None:
                raise ValueError('Algo-KB-Schema mapping table not initialized')

            with self._sql_manager.get_session() as session:
                existing = session.query(bind_table_cls).filter_by(algo_id=algo_id, kb_id=kb_id).first()
                if existing:
                    existing_schema_id = str(getattr(existing, 'schema_set_id'))
                    if existing_schema_id != str(schema_set_id):
                        if not force_refresh:
                            raise ValueError(
                                f'kb_id {kb_id} already bound to schema_set_id {existing_schema_id} for algo {algo_id}'
                            )
                        # clean up
                        self._delete_records(existing_schema_id, kb_id, algo_id)
                        session.delete(existing)
                        session.flush()
                    else:
                        return

                new_obj = bind_table_cls(algo_id=algo_id, kb_id=kb_id, schema_set_id=str(schema_set_id))
                session.add(new_obj)
            return schema_set_id
        except Exception as e:
            LOG.error(f'Failed to bind kb_id {kb_id} to schema_set_id {schema_set_id} for algo {algo_id}: {e}')
            raise e

    def _exec_text_extract(self, doc_nodes: List[DocNode], algo_id: str):
        pass

    def _exec_multimodal_extract(self, doc_nodes: List[DocNode], algo_id: str):
        # TODO: currently only support text extract
        raise NotImplementedError('Multimodal extract not implemented')

    def _validate_extract_params(self, data: List[DocNode], algo_id: str) -> Tuple[str, str, bool]:
        self._lazy_init()
        if not data:
            raise ValueError('data is empty')
        kb_id = doc_id = None
        for node in data:
            meta = getattr(node, 'global_metadata', {}) or {}
            cur_kb_id = meta.get(RAG_KB_ID)
            cur_doc_id = meta.get(RAG_DOC_ID)
            if cur_kb_id is None or cur_doc_id is None:
                raise ValueError('node.global_metadata must contain kb_id and doc_id')
            cur_kb_id = str(cur_kb_id)
            cur_doc_id = str(cur_doc_id)
            if kb_id is None:
                kb_id = cur_kb_id
            elif kb_id != cur_kb_id:
                raise ValueError('kb_id in data must be unique')
            if doc_id is None:
                doc_id = cur_doc_id
            elif doc_id != cur_doc_id:
                raise ValueError('doc_id in data must be unique')

        bind_table_cls = self._sql_manager.get_table_orm_class(Table_ALGO_KB_SCHEMA['name'])
        if bind_table_cls is None:
            raise ValueError('Algo-KB-Schema mapping table not initialized')

        with self._sql_manager.get_session() as session:
            bound = session.query(bind_table_cls).filter_by(algo_id=algo_id, kb_id=kb_id).first() is not None

        return kb_id, doc_id, bound

    def extract_and_store(self, data: List[DocNode], algo_id: str = DocListManager.DEFAULT_GROUP_NAME,
                          schema_set_id: str = None, schema_set: Type[BaseModel] = None) -> ExtractResult:
        """
        Persist extracted fields for a document.

        payload is validated against the given schema_set; algo_id maps to the
        Document/algorithm name。schema_set 支持按需传入用于临时注册。
        """
        self._lazy_init()
        if schema_set is not None:
            self.register_schema_set(schema_set, schema_set_id)
        if schema_set_id not in self._schema_registry:
            raise ValueError(f'Unknown schema_set_id: {schema_set_id}')

        # model_cls = self._schema_registry[schema_set_id]
        # table_name = self._ensure_table(schema_set_id, model_cls)
        # model_data = self._to_model_dict(payload, model_cls)
        # row = {
        #     self.SYS_KB_ID: kb_id,
        #     self.SYS_DOC_ID: doc_id,
        #     self.SYS_ALGO_ID: algo_id,
        #     **model_data,
        # }
        # db_result = self._sql_manager.insert_values(table_name, [row])
        # if db_result.status != DBStatus.SUCCESS:
        #     raise ValueError(f'Insert values failed: {db_result.detail}')

    def __call__(self, data: List[DocNode], algo_id: str = DocListManager.DEFAULT_GROUP_NAME) -> ExtractResult:
        # NOTE: data should be from single file source (kb_id, doc_id should be the same)
        self._lazy_init()
        if not isinstance(data, list):
            data = [data]
        return self.extract_and_store(data=data, algo_id=algo_id)

    def _init_sql_manager(self, db_config: Dict[str, Any]) -> SqlManager:
        return SqlManager(**db_config)

    def _table_name(self, schema_set_id: str) -> str:
        return f'{self._table_prefix}_{schema_set_id}'

    def _ensure_management_tables(self) -> None:
        """Ensure internal schema management tables exist."""
        tables_info_dict = {'tables': [TABLE_SCHEMA_SET_INFO, Table_ALGO_KB_SCHEMA]}
        try:
            self._sql_manager._init_tables_by_info(tables_info_dict)
        except Exception as e:
            LOG.warning(f'Ensure management tables failed: {e}')

    def _ensure_table(self, schema_set_id: str, schema_model: Optional[Type[BaseModel]] = None) -> str:
        if not self._sql_manager:
            raise ValueError('SqlManager is not initialized')
        table_name = self._table_name(schema_set_id)
        if table_name in self._table_cache:
            return table_name
        if schema_model is None:
            schema_model = self._schema_registry.get(schema_set_id)
        if schema_model is None:
            raise ValueError(f'No schema model registered for {schema_set_id}')

        attrs: Dict[str, Any] = {
            '__tablename__': table_name,
            '__table_args__': (
                sqlalchemy.PrimaryKeyConstraint(self.SYS_KB_ID, self.SYS_DOC_ID, name=f'pk_{table_name}_kb_doc'),
                sqlalchemy.Index(f'idx_{table_name}_kb', self.SYS_KB_ID),
                {'extend_existing': True},
            ),
        }
        attrs[self.SYS_KB_ID] = sqlalchemy.Column(sqlalchemy.String(128), nullable=False)
        attrs[self.SYS_DOC_ID] = sqlalchemy.Column(sqlalchemy.String(128), nullable=False)
        attrs[self.SYS_ALGO_ID] = sqlalchemy.Column(sqlalchemy.String(128), nullable=False)

        for field_name, field_type in self._iter_schema_fields(schema_model):
            if field_name in attrs:
                continue
            attrs[field_name] = sqlalchemy.Column(field_type, nullable=True)

        table_cls = type(table_name.capitalize(), (_TableBase,), attrs)
        db_result = self._sql_manager.create_table(table_cls)
        if db_result.status != DBStatus.SUCCESS:
            LOG.warning(f'Create table failed: {db_result.detail}')
        else:
            self._table_cache[table_name] = table_cls
        return table_name

    def _delete_records(self, schema_set_id: str, kb_id: str, algo_id: str) -> None:
        """删除指定 schema_set 表中对应 kb/algo 的记录。"""
        if not self._sql_manager:
            return
        table_name = self._table_name(schema_set_id)
        table_cls = self._sql_manager.get_table_orm_class(table_name)
        if table_cls is None:
            return
        with self._sql_manager.get_session() as session:
            session.query(table_cls).filter_by(
                **{self.SYS_KB_ID: kb_id, self.SYS_ALGO_ID: algo_id}
            ).delete()

    def _iter_schema_fields(self, model: Type[BaseModel]) -> List[tuple[str, Any]]:
        try:
            fields = model.model_fields  # pydantic v2
        except AttributeError:
            fields = model.__fields__  # type: ignore[attr-defined]  # pydantic v1
        result = []
        for name, field in fields.items():
            annotation = getattr(field, 'annotation', None) or getattr(field, 'outer_type_', None)
            column_type = self._column_type(annotation)
            result.append((name, column_type))
        return result

    def _column_type(self, annotation: Any):
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Union and args:
            non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
            annotation = non_none[0] if non_none else str
            origin = get_origin(annotation)
        if origin in (list, set, tuple):
            return sqlalchemy.JSON
        if annotation in self.TYPE_MAP:
            return self.TYPE_MAP[annotation]
        return sqlalchemy.Text

    def _to_model_dict(self, payload: Union[BaseModel, Dict[str, Any]], model_cls: Type[BaseModel]) -> Dict[str, Any]:
        if isinstance(payload, BaseModel):
            try:
                return payload.model_dump()
            except AttributeError:
                return payload.dict()
        validated = model_cls(**payload)
        try:
            return validated.model_dump()
        except AttributeError:
            return validated.dict()

    def _validate_schema_model(self, model: Type[BaseModel]) -> None:
        if not model or not issubclass(model, BaseModel):
            raise TypeError('schema_set must be a pydantic BaseModel subclass')
