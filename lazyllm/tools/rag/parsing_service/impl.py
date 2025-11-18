import os
import time
import traceback
from typing import Any, Dict, List, Optional

from lazyllm import LOG, config

from ..data_loaders import DirectoryReader
from ..doc_node import DocNode
from ..global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID
from ..store import LAZY_IMAGE_GROUP, LAZY_ROOT_NAME
from ..store.document_store import _DocumentStore
from ..store.store_base import DEFAULT_KB_ID
from ..store.utils import fibonacci_backoff
from ..transform import AdaptiveTransform, make_transform
from ..utils import gen_docid


def _get_default_db_config():
    root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
    os.makedirs(root_dir, exist_ok=True)
    db_path = os.path.join(root_dir, 'lazyllm_doc_task_management.db')
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_path,
    }


class _Processor:
    def __init__(self, store: _DocumentStore, reader: DirectoryReader, node_groups: Dict[str, Dict],
                 display_name: Optional[str] = None, description: Optional[str] = None):
        self._store = store
        self._reader = reader
        self._node_groups = node_groups
        self._display_name = display_name
        self._description = description

    @property
    def store(self) -> _DocumentStore:
        return self._store

    @property
    def reader(self) -> DirectoryReader:
        return self._reader

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,
                metadatas: Optional[List[Dict[str, Any]]] = None):
        try:
            if not input_files: return
            if not ids: ids = [gen_docid(path) for path in input_files]
            if metadatas is None:
                metadatas = [{} for _ in input_files]
            for metadata, doc_id, path in zip(metadatas, ids, input_files):
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, path)
                metadata.setdefault(RAG_KB_ID, DEFAULT_KB_ID)
            root_nodes = self._reader.load_data(input_files, metadatas, split_nodes_by_type=True)
            for k, v in root_nodes.items():
                if not v: continue
                self._store.update_nodes(self._set_nodes_number(v))
                self._create_nodes_recursive(v, k)
            LOG.info('Add documents done!')
        except Exception as e:
            LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e

    def _set_nodes_number(self, nodes: List[DocNode]) -> List[DocNode]:
        doc_group_number = {}
        for node in nodes:
            doc_id = node.global_metadata.get(RAG_DOC_ID)
            group_name = node.group
            if doc_id not in doc_group_number:
                doc_group_number[doc_id] = {}
            if group_name not in doc_group_number[doc_id]:
                doc_group_number[doc_id][group_name] = 1
            node.metadata['lazyllm_store_num'] = doc_group_number[doc_id][group_name]
            doc_group_number[doc_id][group_name] += 1
        return nodes

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str):
        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group "{group_name}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')

            if group['parent'] == p_name:
                nodes = self._create_nodes_impl(p_nodes, group_name)
                if nodes: self._create_nodes_recursive(nodes, group_name)

    def _create_nodes_impl(self, p_nodes, group_name):
        # NOTE transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        t = self._node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, group_name)
        nodes = transform.batch_forward(p_nodes, group_name)
        self._store.update_nodes(self._set_nodes_number(nodes))
        return nodes

    def _get_or_create_nodes(self, group_name, uids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(uids=uids, group=group_name) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(self._node_groups[group_name]['parent'], uids)
            nodes = self._create_nodes_impl(p_nodes, group_name)
        return nodes

    def reparse(self, group_name: str, uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None,
                **kwargs):
        if doc_ids:
            self._reparse_docs(group_name=group_name, doc_ids=doc_ids, **kwargs)
        else:
            self._get_or_create_nodes(group_name, uids)

    def _reparse_docs(self, group_name: str, doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict]):
        if not metadatas:
            raise ValueError('metadatas is required for reparse')
        kb_id = metadatas[0].get(RAG_KB_ID, None)
        if group_name == 'all':
            self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
            removed_flag = False
            for wait_time in fibonacci_backoff():
                nodes = self._store.get_nodes(group=LAZY_ROOT_NAME, kb_id=kb_id, doc_ids=doc_ids)
                if not nodes:
                    removed_flag = True
                    break
                time.sleep(wait_time)
            if not removed_flag:
                raise Exception(f'Failed to remove nodes for docs {doc_ids} from store')
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas)
            LOG.info(f'Reparse docs {doc_ids} from store done')
        else:
            p_nodes = self._store.get_nodes(group=self._node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name,
                                          doc_ids=doc_ids)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str, doc_ids: List[str]):
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, None)
        self._store.remove_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)

        removed_flag = False
        for wait_time in fibonacci_backoff():
            nodes = self._store.get_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)
            if not nodes:
                removed_flag = True
                break
            time.sleep(wait_time)
        if not removed_flag:
            raise Exception(f'Failed to remove nodes for docs {doc_ids} group {cur_name} from store')
        t = self._node_groups[cur_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, cur_name)
        nodes = transform.batch_forward(p_nodes, cur_name)
        # reparse need set global_metadata
        self._store.update_nodes(self._set_nodes_number(nodes))

        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group "{group_name}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if group['parent'] == cur_name:
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name, doc_ids=doc_ids)

    def update_doc_meta(self, doc_id: str, metadata: dict):
        try:
            self._store.update_doc_meta(doc_id=doc_id, metadata=metadata)
        except Exception as e:
            LOG.error(f'Failed to update doc meta: {e}, {traceback.format_exc()}')
            raise e

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None) -> None:
        try:
            self._store.remove_nodes(kb_id=kb_id, doc_ids=doc_ids)
        except Exception as e:
            LOG.error(f'Failed to delete doc: {e}, {traceback.format_exc()}')
            raise e
