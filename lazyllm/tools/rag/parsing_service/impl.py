import time
import traceback
from typing import Any, Dict, List, Optional
from graphlib import CycleError, TopologicalSorter
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

from lazyllm import LOG

from ..data_loaders import DirectoryReader
from ..doc_node import DocNode
from ..global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID
from ..store import LAZY_IMAGE_GROUP, LAZY_ROOT_NAME
from ..store.document_store import _DocumentStore
from ..store.store_base import DEFAULT_KB_ID
from ..store.utils import fibonacci_backoff
from ..transform import AdaptiveTransform, make_transform
from ..utils import gen_docid
from ..doc_to_db import SchemaExtractor


class _NodeGroupDependencyGraph:
    def __init__(self, node_groups: Dict[str, Dict], active: List[str]):
        self._shortest_path_cache: Dict[tuple[str, str], List[str]] = {}
        self._forward_graph = defaultdict(set)
        self._dep_graph = {node: set() for node in active}

        for group in active:
            cfg = node_groups.get(group)
            if not cfg:
                raise ValueError(f'Node group "{group}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')

            if parent := cfg['parent']:
                self._forward_graph[parent].add(group)
                self._dep_graph[group].add(parent)
            if ref := cfg.get('ref'):
                self._dep_graph[group].add(ref)

    @cached_property
    def topological_order(self) -> List[str]:
        try:
            return list(TopologicalSorter(self._dep_graph).static_order())
        except CycleError as e:
            raise ValueError(f'Detected node group cycle dependency: {e}')

    def get_shortest_path(self, start: str, end: str) -> List[str]:
        # NOTE: The path from start to end is guaranteed to exist.
        # The returned list does not contain `start` itself, only intermediate nodes and `end`.
        key = (start, end)
        if key in self._shortest_path_cache:
            return self._shortest_path_cache[key]

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            for neighbor in self._forward_graph.get(current, []):
                if neighbor == end:
                    result = path + [end]
                    self._shortest_path_cache[key] = result
                    return result
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        raise AssertionError(f'No path found from {start} to {end}, the dependency graph is not valid')

class _Processor:
    def __init__(self, algo_id: str, store: _DocumentStore, reader: DirectoryReader, node_groups: Dict[str, Dict],
                 schema_extractor: Optional[SchemaExtractor] = None, display_name: Optional[str] = None,
                 description: Optional[str] = None, max_workers: int = 4):
        self._algo_id = algo_id
        self._store = store
        self._reader = reader
        self._node_groups = node_groups
        self._schema_extractor = schema_extractor
        self._display_name = display_name
        self._description = description
        self._max_workers = max_workers
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f'{self._algo_id}_processor')
        self._dependency_graph: Optional[_NodeGroupDependencyGraph] = None

    @property
    def store(self) -> _DocumentStore:
        return self._store

    @property
    def reader(self) -> DirectoryReader:
        return self._reader

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,  # noqa: C901
                metadatas: Optional[List[Dict[str, Any]]] = None, kb_id: Optional[str] = None,
                transfer_mode: Optional[str] = None, target_kb_id: Optional[str] = None,
                target_doc_ids: Optional[List[str]] = None):
        try:
            if not input_files: return
            add_start = time.time()
            if not ids: ids = [gen_docid(path) for path in input_files]
            if metadatas is None:
                metadatas = [{} for _ in input_files]
            for metadata, doc_id, path in zip(metadatas, ids, input_files):
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, path)
                metadata.setdefault(RAG_KB_ID, kb_id or DEFAULT_KB_ID)
            kb_id = metadatas[0].get(RAG_KB_ID, DEFAULT_KB_ID) if kb_id is None else kb_id

            load_start = time.time()
            if transfer_mode is None:
                root_nodes = self._reader.load_data(input_files, metadatas, split_nodes_by_type=True)
            else:
                if transfer_mode not in ('cp', 'mv'):
                    raise ValueError(f'Invalid transfer mode: {transfer_mode}')
                if len(ids) != len(target_doc_ids):
                    raise ValueError(f'The length of doc_ids and target_doc_ids must be the same. '
                                     f'doc_ids:{ids}, target_doc_ids:{target_doc_ids}')
                doc_id_map = {ids[i]: (target_doc_ids[i], metadatas[i]) for i in range(len(ids))}

                root_nodes: List[DocNode] = self._store.get_nodes(doc_ids=ids, group=LAZY_ROOT_NAME, kb_id=kb_id)
                root_nodes = [
                    n.copy(
                        global_metadata={
                            RAG_KB_ID: target_kb_id, RAG_DOC_ID: doc_id_map[n.global_metadata[RAG_DOC_ID]][0]
                        },
                        metadata=doc_id_map[n.global_metadata[RAG_DOC_ID]][1]
                    ) for n in root_nodes
                ]
            load_time = time.time() - load_start

            schema_futures = []
            schema_errors: List[Exception] = []
            # run schema extraction in parallel
            if self._schema_extractor and not transfer_mode:
                doc_to_root_nodes = defaultdict(list)
                for n in root_nodes[LAZY_ROOT_NAME]:
                    doc_to_root_nodes[n.global_metadata.get(RAG_DOC_ID)].append(n)

                if doc_to_root_nodes:
                    for nodes in doc_to_root_nodes.values():
                        schema_futures.append(
                            self._thread_pool.submit(self._schema_extractor, nodes, algo_id=self._algo_id)
                        )

            if transfer_mode is None:
                for k, v in root_nodes.items():
                    if not v: continue
                    self._store.update_nodes(self._set_nodes_number(v))
                    self._create_nodes_recursive(v, k)
            else:
                self._store.update_nodes(root_nodes, copy=True)
                root_uid_map = {n._copy_source.get('uid'): n.uid for n in root_nodes}
                self._copy_segments_recursive(ids=ids, kb_id=kb_id, target_kb_id=target_kb_id,
                                              doc_id_map=doc_id_map, p_uid_map=root_uid_map,
                                              p_name=LAZY_ROOT_NAME)

            for future in schema_futures:
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.error(f'Schema extraction failed: {exc}')
                    schema_errors.append(exc)
            if schema_errors:
                raise schema_errors[0]
            add_time = time.time() - add_start
            LOG.info(f'[_Processor - add_doc] Add documents done! files:{input_files}, '
                     f'Total Time: {add_time}s, Data Loading Time: {load_time}s')
        except Exception as e:
            LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e

    def close(self):
        self._thread_pool.shutdown(wait=True)
        self._thread_pool = None

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

    def _get_dependency_graph(self) -> _NodeGroupDependencyGraph:
        if self._dependency_graph is None:
            self._dependency_graph = _NodeGroupDependencyGraph(self._node_groups, self._store.activated_groups())
        return self._dependency_graph

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str):
        graph = self._get_dependency_graph()
        for group_name in graph.topological_order:
            group = self._node_groups.get(group_name)

            if group['parent'] == p_name:
                ref_path = graph.get_shortest_path(group['parent'], group.get('ref')) if group.get('ref') else []
                nodes = self._create_nodes_impl(p_nodes, group_name, ref_path=ref_path)
                if nodes: self._create_nodes_recursive(nodes, group_name)

    def _copy_segments_recursive(self, ids: List[str], kb_id: str, target_kb_id: str, doc_id_map: Dict[str, tuple],
                                 p_uid_map: dict, p_name: str):
        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group {group_name} does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if group['parent'] == p_name:
                nodes = self._store.get_nodes(doc_ids=ids, group=group_name, kb_id=kb_id)
                nodes = [
                    n.copy(
                        global_metadata={
                            RAG_KB_ID: target_kb_id, RAG_DOC_ID: doc_id_map[n.global_metadata[RAG_DOC_ID]][0]
                        },
                        metadata=doc_id_map[n.global_metadata[RAG_DOC_ID]][1]
                    ) for n in nodes
                ]
                uid_map = {}
                for n in nodes:
                    uid_map[n._copy_source.get('uid')] = n.uid
                    n.parent = p_uid_map.get(n.parent, None) if n.parent else None
                self._store.update_nodes(nodes, copy=True)
                if nodes:
                    self._copy_segments_recursive(ids=ids, kb_id=kb_id, target_kb_id=target_kb_id,
                                                  doc_id_map=doc_id_map, p_uid_map=uid_map,
                                                  p_name=group_name)

    def _create_nodes_impl(self, p_nodes, group_name, ref_path=None):
        # NOTE transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        t = self._node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, group_name)
        nodes = transform.batch_forward(p_nodes, group_name, ref_path=ref_path)
        self._store.update_nodes(self._set_nodes_number(nodes))
        return nodes

    def _get_or_create_nodes(self, group_name, uids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(uids=uids, group=group_name) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(self._node_groups[group_name]['parent'], uids)
            nodes = self._create_nodes_impl(p_nodes, group_name)
        return nodes

    def reparse(self, group_name: str, uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None,
                kb_id: Optional[str] = None, **kwargs):
        if doc_ids:
            self._reparse_docs(group_name=group_name, doc_ids=doc_ids, kb_id=kb_id, **kwargs)
        else:
            self._get_or_create_nodes(group_name, uids)

    def _reparse_docs(self, group_name: str, doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict],
                      kb_id: str = None, **kwargs):
        if not metadatas:
            raise ValueError('metadatas is required for reparse')
        kb_id = metadatas[0].get(RAG_KB_ID, None) if kb_id is None else kb_id
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
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas, kb_id=kb_id)
            LOG.info(f'Reparse docs {doc_ids} from store done')
        else:
            p_nodes = self._store.get_nodes(group=self._node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name,
                                          doc_ids=doc_ids, kb_id=kb_id)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str, doc_ids: List[str], kb_id: str = None):
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, None) if kb_id is None else kb_id
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
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name, doc_ids=doc_ids, kb_id=kb_id)

    def update_doc_meta(self, doc_id: str, metadata: dict, kb_id: str = None):
        try:
            self._store.update_doc_meta(doc_id=doc_id, metadata=metadata, kb_id=kb_id)
        except Exception as e:
            LOG.error(f'Failed to update doc meta: {e}, {traceback.format_exc()}')
            raise e

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None) -> None:
        try:
            self._store.remove_nodes(kb_id=kb_id, doc_ids=doc_ids)
            if self._schema_extractor:
                self._schema_extractor._delete_extract_data(algo_id=self._algo_id, kb_id=kb_id, doc_ids=doc_ids)
        except Exception as e:
            LOG.error(f'Failed to delete doc: {e}, {traceback.format_exc()}')
            raise e
