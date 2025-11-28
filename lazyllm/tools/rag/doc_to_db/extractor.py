from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin
from enum import Enum

import json
import sqlalchemy
import hashlib
from uuid import uuid4
from pydantic import BaseModel, Field, create_model
from datetime import datetime
from string import Template

from lazyllm import LOG, ThreadPoolExecutor, once_wrapper
from lazyllm.components import JsonFormatter
from lazyllm.module import LLMBase

from ...sql.sql_manager import DBStatus, SqlManager
from ..doc_node import DocNode
from ..global_metadata import RAG_DOC_ID, RAG_KB_ID
from ..utils import DocListManager, _orm_to_dict
from ..store.store_base import DEFAULT_KB_ID
from .model import (
    TABLE_SCHEMA_SET_INFO, Table_ALGO_KB_SCHEMA, ExtractionMode,
    _TableBase, ExtractResult, ExtractMeta, ExtractClue, SchemaSetInfo
)
from .prompts import (
    SCHEMA_EXTRACT_PROMPT, SCHEMA_EXTRACT_INPUT_FORMAT,
    SCHEMA_ANALYZE_PROMPT, SCHEMA_ANALYZE_INPUT_FORMAT
)
from .utils import _col_type_name

ONE_DOC_LENGTH_LIMIT = 102400


class SchemaExtractor:
    '''Schema aware extractor that materializes BaseModel schemas into database tables.'''

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
    TYPE_NAME_MAP = {
        'string': str,
        'text': str,
        'int': int,
        'integer': int,
        'float': float,
        'number': float,
        'boolean': bool,
        'bool': bool,
        'list': list,
        'array': list,
        'dict': dict,
        'object': dict,
        'map': dict,
    }

    def __init__(self, db_config: Dict[str, Any], llm: LLMBase, *, table_prefix: Optional[str] = None,
                 force_refresh: bool = False, extraction_mode: ExtractionMode = ExtractionMode.TEXT,
                 max_len: int = ONE_DOC_LENGTH_LIMIT, num_workers: int = 4):
        if not isinstance(llm, LLMBase):
            raise TypeError('llm must be an instance of LLMBase')
        self._llm = llm
        self._table_prefix = table_prefix or self.TABLE_PREFIX
        self._sql_manager = None
        self._db_config = db_config
        self._table_cache: Dict[str, Type[_TableBase]] = {}
        self._schema_registry: Dict[str, Type[BaseModel]] = {}
        self._force_refresh = force_refresh
        self._extraction_mode = extraction_mode
        self._max_len = max_len
        self._num_workers = num_workers

    @property
    def sql_manager(self) -> SqlManager:
        self._lazy_init()
        return self._sql_manager

    def sql_manager_for_nl2sql(self, algo_id: str = None,  # noqa: C901
                               kb_ids: Union[str, List[str]] = None) -> SqlManager:
        self._lazy_init()
        if not self._sql_manager:
            raise ValueError('SqlManager is not initialized')
        if not self._db_config:
            raise ValueError('db_config is required to build SqlManager')

        bind_table_name = Table_ALGO_KB_SCHEMA['name']
        schema_info_table = TABLE_SCHEMA_SET_INFO['name']
        desc_map: Dict[str, str] = {}

        # Prepare description for mapping table (only used when it is exposed)
        bind_desc_parts: List[str] = []
        if Table_ALGO_KB_SCHEMA.get('comment'):
            bind_desc_parts.append(Table_ALGO_KB_SCHEMA['comment'])
        col_comments = [
            f"{col.get('name')}: {col.get('comment')}"
            for col in Table_ALGO_KB_SCHEMA.get('columns', [])
            if col.get('comment')
        ]
        if col_comments:
            bind_desc_parts.append('\n'.join(col_comments))
        bind_desc = '\n'.join(bind_desc_parts) if bind_desc_parts else ''

        def _schema_table_desc(model: Type[BaseModel]) -> str:
            schema_desc = self._get_schema_set_str(model)
            return '\n'.join([s for s in [
                (model.__doc__ or '').strip(),
                schema_desc,
                f'System columns: {self.SYS_KB_ID}, {self.SYS_DOC_ID}, {self.SYS_ALGO_ID}, extract_meta',
            ] if s])

        bind_table_cls = self._sql_manager.get_table_orm_class(bind_table_name)
        if bind_table_cls is None:
            raise ValueError('Algo-KB-Schema mapping table not initialized')

        kb_id_list = None
        if kb_ids is not None:
            if isinstance(kb_ids, (list, tuple, set)):
                kb_id_list = [str(k) for k in kb_ids if k is not None]
            else:
                kb_id_list = [str(kb_ids)]
            if not kb_id_list:
                kb_id_list = None

        with self._sql_manager.get_session() as session:
            query = session.query(bind_table_cls)
            if algo_id:
                query = query.filter_by(algo_id=str(algo_id))
            if kb_id_list:
                query = query.filter(bind_table_cls.kb_id.in_(kb_id_list))
            bound_rows = query.all()
        if not bound_rows:
            raise ValueError(f'No schema binding found for algo_id={algo_id} kb_ids={kb_ids}')

        target_tables = {bind_table_name} if not (algo_id and kb_id_list) else set()
        if bind_desc and bind_table_name in target_tables:
            desc_map[bind_table_name] = bind_desc
        for row in bound_rows:
            schema_set_id = str(row.schema_set_id)
            if not self.has_schema_set(schema_set_id):
                raise ValueError(f'Schema set {schema_set_id} not found')
            schema_model = self._schema_registry[schema_set_id]
            table_name = self._ensure_table(schema_set_id, schema_model)
            target_tables.add(table_name)
            desc_map[table_name] = _schema_table_desc(schema_model)

        # Exclude internal schema registry table from NL2SQL exposure
        target_tables.discard(schema_info_table)
        tables_info_dict = {'tables': []}
        for table_name in target_tables:
            table_cls = self._sql_manager.get_table_orm_class(table_name)
            if table_cls is None:
                continue
            columns = []
            for col in table_cls.__table__.columns:
                columns.append({
                    'name': col.name,
                    'data_type': _col_type_name(col),
                    'nullable': bool(col.nullable),
                    'is_primary_key': bool(col.primary_key),
                    'comment': getattr(col, 'comment', '') or '',
                })
            tables_info_dict['tables'].append({'name': table_name, 'columns': columns, 'comment': ''})
        new_manager = self._init_sql_manager({**self._db_config, 'tables_info_dict': tables_info_dict})
        new_manager.visible_tables = list(target_tables)
        if desc_map:
            new_manager.set_desc(desc_map)
        return new_manager

    @once_wrapper
    def _lazy_init(self):
        self._sql_manager = self._init_sql_manager(self._db_config) if self._db_config else None
        if self._sql_manager:
            self._ensure_management_tables()

    def register_schema_set(self, schema_set: Type[BaseModel], schema_set_id: str = None,   # noqa: C901
                            force_refresh: bool = False) -> str:
        '''schema set registration, idempotent'''
        try:
            self._lazy_init()
            self._validate_schema_model(schema_set)

            fields = getattr(schema_set, 'model_fields', None) or getattr(schema_set, '__fields__', {})

            def _safe_default(val: Any):
                if val is None:
                    return None
                if val.__class__.__name__ in ('PydanticUndefinedType', 'UndefinedType'):
                    return None
                if isinstance(val, (str, int, float, bool)):
                    return val
                return str(val)

            signature = [
                (name, str(getattr(f, 'annotation', None) or getattr(f, 'outer_type_', None)),
                 _safe_default(getattr(f, 'default', None)), getattr(f, 'default_factory', None) is not None,
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
                        existing_id = str(existing.schema_set_id if hasattr(existing, 'schema_set_id') else existing.id)
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
                            schema_set_id = str(uuid4().hex)
                        obj_kwargs['schema_set_id'] = str(schema_set_id)
                        new_obj = table_cls(**obj_kwargs)
                        session.add(new_obj)
                        session.flush()
                        schema_set_id = str(new_obj.schema_set_id if hasattr(new_obj, 'schema_set_id') else new_obj.id)

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

    def _model_from_schema_json(self, schema_json: str, model_name: str = 'RecoveredSchema') -> Type[BaseModel]:
        '''Reconstruct a minimal BaseModel subclass from stored JSON schema.'''
        try:
            schema_dict = json.loads(schema_json)
        except Exception as exc:
            raise ValueError(f'Invalid schema json: {exc}') from exc
        properties = schema_dict.get('properties', {})
        required = set(schema_dict.get('required', []) or [])
        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        fields_def: Dict[str, Tuple[Any, Any]] = {}
        for name, prop in properties.items():
            t_name = prop.get('type')
            py_type = type_map.get(t_name, str)
            desc = prop.get('description', '')
            default = ... if name in required else None
            fields_def[name] = (py_type, Field(default=default, description=desc))
        return create_model(model_name, **fields_def)  # type: ignore[arg-type]

    def has_schema_set(self, schema_set_id: str) -> bool:
        self._lazy_init()
        if self._sql_manager:
            table_cls = self._sql_manager.get_table_orm_class(TABLE_SCHEMA_SET_INFO['name'])
            if table_cls is None:
                raise ValueError('Schema set table not initialized')
            with self._sql_manager.get_session() as session:
                existing = session.query(table_cls).filter_by(schema_set_id=schema_set_id).first()
                if not existing:
                    return False
                if schema_set_id not in self._schema_registry:
                    recovered_schema = self._model_from_schema_json(existing.schema_set_json,
                                                                    model_name=f'Schema_{schema_set_id}')
                    self._schema_registry[schema_set_id] = recovered_schema
                    self._ensure_table(schema_set_id, recovered_schema)
                return True
        return schema_set_id in self._schema_registry

    def register_schema_set_to_kb(self, algo_id: Optional[str] = DocListManager.DEFAULT_GROUP_NAME,
                                  kb_id: Optional[str] = DEFAULT_KB_ID, schema_set_id: Optional[str] = None,
                                  schema_set: Type[BaseModel] = None, force_refresh: bool = False) -> str:
        '''
        Bind a KB to a schema set.

        This is used to ensure that the KB is compatible with the schema set.
        '''
        try:
            self._lazy_init()
            force_refresh = force_refresh or self._force_refresh
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
                    existing_schema_id = str(existing.schema_set_id)
                    if existing_schema_id != str(schema_set_id):
                        if not force_refresh:
                            raise ValueError(
                                f'kb_id {kb_id} already bound to schema_set_id {existing_schema_id} for algo {algo_id}'
                            )
                        # clean up
                        LOG.info(f'Clean up records for schema_set_id {existing_schema_id} kb_id {kb_id} algo {algo_id}')
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

    def _get_schema_set_str(self, schema_set) -> str:
        '''Return a human readable schema description: name, description, data type.'''
        model = None
        if isinstance(schema_set, str):
            model = self._schema_registry.get(schema_set)
        else:
            model = schema_set
        if not model:
            raise ValueError(f'Unknown schema_set: {schema_set}')
        fields = getattr(model, 'model_fields', None) or getattr(model, '__fields__', {})

        def _field_type_str(field_obj: Any) -> str:
            anno = getattr(field_obj, 'annotation', None) or getattr(field_obj, 'outer_type_', None)
            origin = get_origin(anno)
            args = get_args(anno)
            if origin is Union and args:
                non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
                anno = non_none[0] if non_none else anno
            return getattr(anno, '__name__', str(anno))

        lines: List[str] = []
        for name, field in fields.items():
            desc = getattr(field, 'description', None)
            if desc is None:
                field_info = getattr(field, 'field_info', None)
                desc = getattr(field_info, 'description', None) if field_info else None
            type_str = _field_type_str(field)
            lines.append(f"name: {name}, description: {desc or ''}, type: {type_str}")
        return '\n'.join(lines)

    def analyze_schema_and_register(self, data: Union[str, List[DocNode]],
                                    schema_set_id: Optional[str] = None) -> SchemaSetInfo:
        '''Infer a schema from sample data, register it, and return the registration info.'''
        self._lazy_init()
        if not self._llm:
            raise ValueError('LLM not initialized')
        if not data:
            raise ValueError('data is empty')

        if isinstance(data, str):
            sample_text = data[:self._max_len]
        else:
            chunks = self._gen_text_list_from_nodes(data)
            sample_text = '\n\n'.join(chunks)[:self._max_len] if chunks else ''
        if not sample_text:
            raise ValueError('No content available for schema analysis')

        llm = self._llm.share(prompt=SCHEMA_ANALYZE_PROMPT, format=JsonFormatter())
        payload = Template(SCHEMA_ANALYZE_INPUT_FORMAT).substitute(text=sample_text)
        res = llm(payload)
        fields_def: Dict[str, Tuple[Any, Any]] = {}
        for item in res:
            if not isinstance(item, dict):
                continue
            name = item.get('name')
            if not name:
                continue
            desc = item.get('description') or ''
            py_type = self._normalize_py_type(item.get('type'))
            fields_def[name] = (py_type, Field(default=None, description=desc))
        if not fields_def:
            # Fallback: single generic field capturing text content
            fields_def['content'] = (str, Field(default=None, description='Raw content snippet'))

        model_name = f'AutoSchema{uuid4().hex}'
        schema_model = create_model(model_name, **fields_def)  # type: ignore[arg-type]
        reg_id = self.register_schema_set(schema_model, schema_set_id)
        return SchemaSetInfo(schema_set_id=reg_id, schema_model=schema_model)

    def _gen_text_list_from_nodes(self, nodes: List[DocNode]) -> list[str]:
        '''Generate full text blocks with metadata, each capped by `self._max_len`.'''
        if not nodes:
            return []
        template = 'File Info:\n{file_metas}\nFile Content:\n{file_content}\n\n'
        metas = '\n'.join([f'{k}: {v}' for k, v in nodes[0].global_metadata.items()])

        # Reserve space for metadata and static prompt text.
        base_len = len(template.format(file_metas=metas, file_content=''))
        content_limit = max(self._max_len - base_len, 0)
        if content_limit == 0:
            return [template.format(file_metas=metas, file_content='')]

        chunks: List[str] = []
        current = ''
        for node in nodes:
            node_text = node.text
            sep_len = 1 if current else 0
            if len(current) + sep_len + len(node_text) <= content_limit:
                current = f'{current}\n{node_text}' if current else node_text
                continue

            if current:
                chunks.append(current)
            start = 0
            while start < len(node_text):
                end = start + content_limit
                chunks.append(node_text[start:end])
                start = end
            current = ''

        if current:
            chunks.append(current)

        return [template.format(file_metas=metas, file_content=chunk) for chunk in chunks]

    def _text_extract_impl(self, data: Union[str, List[DocNode]], schema_set_id: str) -> ExtractResult:  # noqa: C901
        if not self._llm:
            raise ValueError('LLM not initialized')
        schema_set = self._schema_registry.get(schema_set_id)
        llm = self._llm.share(prompt=SCHEMA_EXTRACT_PROMPT, format=JsonFormatter())
        content_list = self._gen_text_list_from_nodes(data) if isinstance(data, list) else [data]
        schema_str = self._get_schema_set_str(schema_set)
        input_list = [
            Template(SCHEMA_EXTRACT_INPUT_FORMAT).substitute(schema=schema_str, text=content)
            for content in content_list
        ]
        if self._num_workers > 1:
            pool = ThreadPoolExecutor(max_workers=self._num_workers)
            fs = [pool.submit(llm, text) for text in input_list]
            res = [f.result() for f in fs]
        else:
            res = [llm(text) for text in input_list]
        # process res by vote
        schema_val_clues: Dict[str, Dict[str, List[str]]] = {}
        for res_item in res:
            if not isinstance(res_item, list):
                LOG.error(f'[Schema Extractor - _text_extract_impl] invalid format {res_item}')
                continue
            for info in res_item:
                if not isinstance(info, dict):
                    continue
                schema_name = info.get('schema_name') or info.get('field_name')
                if not schema_name:
                    continue
                val_js = json.dumps(info.get('value'), ensure_ascii=False)
                if val_js is None:
                    continue
                clues = info.get('clues') or []
                schema_val_clues.setdefault(schema_name, {}).setdefault(val_js, []).extend(clues)

        data: Dict[str, Any] = {}
        clue_meta: Dict[str, ExtractClue] = {}
        for name, val_map in schema_val_clues.items():
            best_val_js = None
            best_clues: List[str] = []
            for v_js, clues in val_map.items():
                if len(clues) > len(best_clues):
                    best_val_js = v_js
                    best_clues = clues
            if best_val_js is None:
                continue
            try:
                best_val = json.loads(best_val_js)
            except Exception:
                best_val = best_val_js
            data[name] = best_val
            clue_meta[name] = ExtractClue(reason='selected_by_max_clues', citation=best_clues)

        meta = ExtractMeta(
            schema_set_id=schema_set_id,
            mode=self._extraction_mode,
            algo_id='',
            kb_id='',
            doc_id='',
            clues=clue_meta,
        )
        return [ExtractResult(data=data, metadata=meta)]

    def _multimodal_extract_impl(self, doc_nodes: List[DocNode], schemet_set_id: str) -> ExtractResult:
        # TODO: currently only support text extract
        raise NotImplementedError('Multimodal extract not implemented')

    def _schema_extract_impl(self, doc_nodes: List[DocNode]):
        raise NotImplementedError('Schema extract not implemented')

    def _validate_extract_params(self, data: Union[str, List[DocNode]], algo_id: str) -> Tuple[str, str, bool]:
        self._lazy_init()
        if not data:
            raise ValueError('data is empty')
        kb_id = doc_id = None
        if isinstance(data, str):
            kb_id = DEFAULT_KB_ID
            doc_id = hashlib.sha256(data.encode('utf-8')).hexdigest()
        else:
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
            bound = session.query(bind_table_cls).filter_by(algo_id=algo_id, kb_id=kb_id).first()
        return kb_id, doc_id, _orm_to_dict(bound)

    def extract_and_store(self, data: Union[str, List[DocNode]],  # noqa: C901
                          algo_id: str = DocListManager.DEFAULT_GROUP_NAME,
                          schema_set_id: str = None, schema_set: Type[BaseModel] = None) -> ExtractResult:
        '''Persist extracted fields for a document'''
        self._lazy_init()
        if schema_set is not None:
            schema_set_id = self.register_schema_set(schema_set, schema_set_id)
        if schema_set_id and not self.has_schema_set(schema_set_id):
            raise ValueError(f'schema_set_id {schema_set_id} not found')
        if not isinstance(data, (str, list)):
            raise TypeError(f'data must be a string or a list of DocNode, got {type(data)}')
        if isinstance(data, list) and any(not isinstance(n, DocNode) for n in data):
            raise TypeError('data list must contain DocNode instances')
        kb_id, doc_id, bound = self._validate_extract_params(data, algo_id)
        if not bound:
            raise ValueError(f'Algo {algo_id}, KB {kb_id} not bound to schema_set_id {schema_set_id}')
        # cache
        search_res = self._get_extract_data(algo_id=algo_id, kb_id=kb_id, doc_ids=[doc_id])
        if search_res: return search_res[0]
        schema_set_id = bound['schema_set_id'] if schema_set_id is None else schema_set_id
        if schema_set_id not in self._schema_registry:
            raise ValueError(f'Unknown schema_set_id: {schema_set_id}')
        if self._extraction_mode == ExtractionMode.TEXT:
            res = self._text_extract_impl(data, schema_set_id)
        elif self._extraction_mode == ExtractionMode.MULTIMODAL:
            res = self._multimodal_extract_impl(data, schema_set_id)
        else:
            raise ValueError(f'Unknown extraction mode: {self._extraction_mode}')
        if not res:
            return None
        res_item = res[0] if isinstance(res, list) else res
        res_item.metadata.algo_id = algo_id
        res_item.metadata.kb_id = kb_id
        res_item.metadata.doc_id = doc_id

        schema_model = self._schema_registry[schema_set_id]
        table_name = self._ensure_table(schema_set_id, schema_model)
        table_cls = self._sql_manager.get_table_orm_class(table_name)
        if table_cls is None:
            raise ValueError(f'Target table {table_name} not initialized')
        payload = {
            self.SYS_KB_ID: kb_id,
            self.SYS_DOC_ID: doc_id,
            self.SYS_ALGO_ID: algo_id,
        }
        payload.update(self._to_model_dict(res_item.data, schema_model))
        meta_obj = getattr(res_item, 'metadata', None) or {}
        if isinstance(meta_obj, BaseModel):
            try:
                meta_payload = meta_obj.model_dump(mode='json')
            except AttributeError:
                meta_payload = meta_obj.dict(use_enum_values=True)
        elif isinstance(meta_obj, dict):
            meta_payload = self._json_safe(meta_obj)
        else:
            meta_payload = {}
        payload['extract_meta'] = self._json_safe(meta_payload)

        with self._sql_manager.get_session() as session:
            session.query(table_cls).filter_by(
                **{self.SYS_KB_ID: kb_id, self.SYS_DOC_ID: doc_id}
            ).delete()
            session.add(table_cls(**payload))
        return res_item

    def _delete_extract_data(self, algo_id: str, doc_ids: List[str], kb_id: str = None) -> bool:
        '''Delete extracted data for docs.'''
        try:
            self._lazy_init()
            if not self._sql_manager:
                raise ValueError('SqlManager is not initialized')
            if not doc_ids:
                return True

            kb_id = kb_id or DEFAULT_KB_ID
            doc_ids = [str(d) for d in doc_ids]

            bind_table_cls = self._sql_manager.get_table_orm_class(Table_ALGO_KB_SCHEMA['name'])
            if bind_table_cls is None:
                raise ValueError('Algo-KB-Schema mapping table not initialized')

            with self._sql_manager.get_session() as session:
                bind_row = session.query(bind_table_cls).filter_by(algo_id=algo_id, kb_id=kb_id).first()
            if not bind_row:
                return True

            schema_set_id = str(bind_row.schema_set_id)
            table_name = self._table_name(schema_set_id)
            table_cls = self._sql_manager.get_table_orm_class(table_name)
            if table_cls is None:
                return True

            with self._sql_manager.get_session() as session:
                session.query(table_cls).filter_by(
                    **{self.SYS_KB_ID: kb_id, self.SYS_ALGO_ID: algo_id}
                ).filter(
                    table_cls.doc_id.in_(doc_ids)
                ).delete(synchronize_session=False)
            return True
        except Exception as e:
            LOG.error(f'Failed to delete doc_ids={doc_ids} from kb_id={kb_id}', e)
            return False

    def _get_extract_data(self, algo_id: str, doc_ids: List[str],  # noqa: C901
                          kb_id: str = None) -> List[ExtractResult]:
        '''Batch fetch extracted data.'''
        self._lazy_init()
        if not self._sql_manager:
            raise ValueError('SqlManager is not initialized')
        if not doc_ids:
            return []

        bind_table_cls = self._sql_manager.get_table_orm_class(Table_ALGO_KB_SCHEMA['name'])
        if bind_table_cls is None:
            raise ValueError('Algo-KB-Schema mapping table not initialized')

        with self._sql_manager.get_session() as session:
            bind_row = session.query(bind_table_cls).filter_by(algo_id=algo_id, kb_id=kb_id).first()
        if not bind_row:
            return []

        schema_set_id = str(bind_row.schema_set_id)
        self.has_schema_set(schema_set_id)
        table_name = self._table_name(schema_set_id)
        table_cls = self._sql_manager.get_table_orm_class(table_name)
        if table_cls is None:
            return []

        schema_model = self._schema_registry.get(schema_set_id)
        with self._sql_manager.get_session() as session:
            rows = session.query(table_cls).filter_by(
                **{self.SYS_KB_ID: kb_id, self.SYS_ALGO_ID: algo_id}
            ).filter(
                table_cls.doc_id.in_(doc_ids)
            ).all()

        results: List[ExtractResult] = []
        sys_fields = {self.SYS_KB_ID, self.SYS_DOC_ID, self.SYS_ALGO_ID, 'extract_meta'}
        for row in rows:
            row_data = {}
            for col in table_cls.__table__.columns:
                name = col.name
                if name in sys_fields:
                    continue
                row_data[name] = getattr(row, name)
            if schema_model:
                try:
                    row_data = self._to_model_dict(row_data, schema_model)
                except Exception:
                    pass

            meta_payload = getattr(row, 'extract_meta', {}) or {}
            if not isinstance(meta_payload, dict):
                try:
                    meta_payload = json.loads(meta_payload)
                except Exception:
                    meta_payload = {}
            meta_payload = meta_payload if isinstance(meta_payload, dict) else {}
            meta_payload.setdefault('schema_set_id', schema_set_id)
            meta_payload.setdefault('algo_id', algo_id)
            meta_payload.setdefault('kb_id', kb_id)
            meta_payload.setdefault('doc_id', str(getattr(row, self.SYS_DOC_ID, '')))
            try:
                meta = ExtractMeta(**meta_payload)
            except Exception:
                meta = ExtractMeta(schema_set_id=schema_set_id, algo_id=algo_id, kb_id=kb_id,
                                   doc_id=str(getattr(row, self.SYS_DOC_ID, '')))
            results.append(ExtractResult(data=row_data, metadata=meta))
        return results

    def __call__(self, data: Union[str, List[DocNode]],
                 algo_id: str = DocListManager.DEFAULT_GROUP_NAME) -> ExtractResult:
        # NOTE: data should be from single file source (kb_id, doc_id should be the same)
        self._lazy_init()
        res = self.extract_and_store(data=data, algo_id=algo_id)
        LOG.info(f'[Schema Extractor] extract res: {res} for algo {algo_id} {data}...')
        return res

    def _init_sql_manager(self, db_config: Dict[str, Any]) -> SqlManager:
        return SqlManager(**db_config)

    def _table_name(self, schema_set_id: str) -> str:
        return f'{self._table_prefix}_{schema_set_id}'

    def _ensure_management_tables(self) -> None:
        '''Ensure internal schema management tables exist.'''
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
        attrs['extract_meta'] = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)

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
        if not self._sql_manager:
            return
        table_name = self._table_name(schema_set_id)
        table_cls = self._sql_manager.get_table_orm_class(table_name)
        if table_cls is None:
            return
        with self._sql_manager.get_session() as session:
            record = session.query(table_cls).filter_by(
                **{self.SYS_KB_ID: kb_id, self.SYS_ALGO_ID: algo_id}
            ).delete()
            LOG.info(f'Deleted {record} records from {table_name}...')

    def _iter_schema_fields(self, model: Type[BaseModel]) -> List[tuple[str, Any]]:
        try:
            fields = model.model_fields  # pydantic v2
        except AttributeError:
            fields = model.__fields__  # type: ignore[attr-defined]  # pydantic v1
        result = []
        for name, field in fields.items():
            annotation = getattr(field, 'annotation', None) or getattr(field, 'outer_type_', None)
            result.append((name, self._column_type(annotation)))
        return result

    def _normalize_py_type(self, type_hint: Any):
        if isinstance(type_hint, str):
            return self.TYPE_NAME_MAP.get(type_hint.lower(), str)
        return type_hint or str

    def _column_type(self, annotation: Any):
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Union and args:
            non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
            annotation = non_none[0] if non_none else str
            origin = get_origin(annotation)
        if origin in (list, set, tuple):
            return sqlalchemy.JSON
        resolved = self._normalize_py_type(annotation)
        if resolved in self.TYPE_MAP:
            return self.TYPE_MAP[resolved]
        if resolved in (list, set, tuple):
            return sqlalchemy.JSON
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

    def _json_safe(self, obj: Any) -> Any:
        '''Convert common objects (Enum/BaseModel) to JSON-serializable primitives.'''
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            try:
                return obj.model_dump(mode='json')
            except AttributeError:
                return obj.dict(use_enum_values=True)
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        return obj
