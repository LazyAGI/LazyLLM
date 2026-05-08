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
    def __init__(self, store: _DocumentStore, schema_extractors: Optional[Dict[str, SchemaExtractor]] = None,
                 max_workers: int = 4):
        self._store = store
        self._schema_extractors = schema_extractors or {}
        self._max_workers = max_workers
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='global_processor')
        self._dep_graph_cache: Dict[frozenset, _NodeGroupDependencyGraph] = {}

    @property
    def store(self) -> _DocumentStore:
        return self._store

    @staticmethod
    def _prepare_doc_inputs(input_files: List[str], ids: Optional[List[str]] = None,
                            metadatas: Optional[List[Dict[str, Any]]] = None,
                            kb_id: Optional[str] = None) -> tuple[List[str], List[Dict[str, Any]], str]:
        if not input_files:
            return [], [], kb_id or DEFAULT_KB_ID
        ids = ids or [gen_docid(path) for path in input_files]
        normalized_metadatas = metadatas or [{} for _ in input_files]
        for i, (doc_id, path) in enumerate(zip(ids, input_files)):
            metadata = normalized_metadatas[i] or {}
            metadata.setdefault(RAG_DOC_ID, doc_id)
            metadata.setdefault(RAG_DOC_PATH, path)
            metadata.setdefault(RAG_KB_ID, kb_id or DEFAULT_KB_ID)
            normalized_metadatas[i] = metadata
        resolved_kb_id = normalized_metadatas[0].get(RAG_KB_ID, DEFAULT_KB_ID) if kb_id is None else kb_id
        return ids, normalized_metadatas, resolved_kb_id

    def add_doc(self, input_files: List[str], node_groups: Dict[str, Dict],  # noqa: C901
                reader: DirectoryReader,
                ids: Optional[List[str]] = None,
                metadatas: Optional[List[Dict[str, Any]]] = None, kb_id: Optional[str] = None,
                transfer_mode: Optional[str] = None, target_kb_id: Optional[str] = None,
                target_doc_ids: Optional[List[str]] = None,
                preloaded_root_nodes: Optional[Dict[str, List[DocNode]]] = None,
                skip_ng_ids: Optional[set] = None,
                extractor_names: Optional[List[str]] = None):
        ids = ids or []
        try:
            if not input_files: return
            add_start = time.time()
            ids, metadatas, kb_id = self._prepare_doc_inputs(input_files, ids, metadatas, kb_id)

            load_start = time.time()
            if transfer_mode is None:
                root_nodes = (
                    preloaded_root_nodes
                    if preloaded_root_nodes is not None
                    else reader.load_data(input_files, metadatas, split_nodes_by_type=True)
                )
            else:
                if transfer_mode not in ('cp', 'mv'):
                    raise ValueError(f'Invalid transfer mode: {transfer_mode}')
                if len(ids) != len(target_doc_ids):
                    raise ValueError(f'The length of doc_ids and target_doc_ids must be the same. '
                                     f'doc_ids:{ids}, target_doc_ids:{target_doc_ids}')
                doc_id_map = {ids[i]: (target_doc_ids[i], metadatas[i]) for i in range(len(ids))}

                source_root_nodes: List[DocNode] = self._store.get_nodes(doc_ids=ids, group=LAZY_ROOT_NAME, kb_id=kb_id)
                root_uid_map = {}
                root_nodes = []
                for node in source_root_nodes:
                    copied = self._clone_node_for_transfer(
                        node=node,
                        target_kb_id=target_kb_id,
                        target_doc_id=doc_id_map[node.global_metadata[RAG_DOC_ID]][0],
                        metadata=doc_id_map[node.global_metadata[RAG_DOC_ID]][1],
                    )
                    root_uid_map[node.uid] = copied.uid
                    root_nodes.append(copied)
            load_time = time.time() - load_start

            schema_futures = []
            schema_errors: List[Exception] = []
            # run schema extraction in parallel for each extractor
            active_extractors = {}
            if not transfer_mode and self._schema_extractors:
                names = extractor_names if extractor_names else list(self._schema_extractors.keys())
                for ename in names:
                    ext = self._schema_extractors.get(ename)
                    if ext:
                        active_extractors[ename] = ext
            if active_extractors:
                doc_to_root_nodes = defaultdict(list)
                for n in root_nodes[LAZY_ROOT_NAME]:
                    doc_to_root_nodes[n.global_metadata.get(RAG_DOC_ID)].append(n)

                if doc_to_root_nodes:
                    for ext in active_extractors.values():
                        for nodes in doc_to_root_nodes.values():
                            schema_futures.append(
                                self._thread_pool.submit(ext, nodes)
                            )

            if transfer_mode is None:
                for k, v in root_nodes.items():
                    if not v: continue
                    self._store.update_nodes(self._set_nodes_number(v))
                    self._create_nodes_recursive(v, k, node_groups=node_groups, skip_ng_ids=skip_ng_ids)
            else:
                self._store.update_nodes(root_nodes, copy=True)
                self._copy_segments_recursive(ids=ids, kb_id=kb_id, target_kb_id=target_kb_id,
                                              doc_id_map=doc_id_map, p_uid_map=root_uid_map,
                                              p_name=LAZY_ROOT_NAME, node_groups=node_groups)

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
            cleanup_doc_ids = ids if transfer_mode is None else (target_doc_ids or [])
            cleanup_kb_id = kb_id if transfer_mode is None else target_kb_id
            self._cleanup_failed_add(cleanup_doc_ids, cleanup_kb_id, clear_schema=transfer_mode is None)
            LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e

    def _cleanup_failed_add(self, doc_ids: List[str], kb_id: Optional[str], clear_schema: bool) -> None:
        if not doc_ids:
            return
        try:
            self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
        except Exception as cleanup_exc:
            LOG.error(f'Failed to cleanup nodes for docs {doc_ids} in kb {kb_id}: {cleanup_exc}, '
                      f'{traceback.format_exc()}')
        if clear_schema and self._schema_extractors:
            for ext in self._schema_extractors.values():
                try:
                    ext._delete_extract_data(kb_id=kb_id, doc_ids=doc_ids)
                except Exception as schema_exc:
                    LOG.error(f'Failed to cleanup schema data for docs {doc_ids} in kb {kb_id}: {schema_exc}, '
                              f'{traceback.format_exc()}')

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

    def _get_dependency_graph(self, node_groups: Dict[str, Dict]) -> _NodeGroupDependencyGraph:
        key = frozenset(node_groups.keys())
        if key not in self._dep_graph_cache:
            self._dep_graph_cache[key] = _NodeGroupDependencyGraph(node_groups, self._store.activated_groups())
        return self._dep_graph_cache[key]

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str,
                                node_groups: Dict[str, Dict], skip_ng_ids: Optional[set] = None):
        graph = self._get_dependency_graph(node_groups)
        for group_name in graph.topological_order:
            group = node_groups.get(group_name)

            if group['parent'] == p_name:
                ref_path = graph.get_shortest_path(group['parent'], group.get('ref')) if group.get('ref') else []
                nodes = self._create_nodes_impl(p_nodes, group_name, node_groups,
                                                ref_path=ref_path, skip_ng_ids=skip_ng_ids)
                if nodes: self._create_nodes_recursive(nodes, group_name, node_groups, skip_ng_ids=skip_ng_ids)

    def _clone_node_for_transfer(
        self, node: DocNode, target_kb_id: str, target_doc_id: str, metadata: Dict[str, Any]
    ) -> DocNode:
        copied = node.copy(
            global_metadata={RAG_KB_ID: target_kb_id, RAG_DOC_ID: target_doc_id},
            metadata=metadata,
        )
        copied._global_metadata = {
            **(node.global_metadata or {}),
            RAG_KB_ID: target_kb_id,
            RAG_DOC_ID: target_doc_id,
        }
        copied._metadata = {**(node.metadata or {}), **(metadata or {})}
        return copied

    def _copy_segments_recursive(self, ids: List[str], kb_id: str, target_kb_id: str, doc_id_map: Dict[str, tuple],
                                 p_uid_map: dict, p_name: str, node_groups: Dict[str, Dict],
                                 skip_ng_ids: Optional[set] = None):
        for group_name in self._store.activated_groups():
            group = node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group {group_name} does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if group['parent'] == p_name:
                ng_cfg = node_groups.get(group_name, {})
                ng_id = ng_cfg.get('id') or group_name
                if skip_ng_ids and ng_id in skip_ng_ids:
                    target_doc_ids = [doc_id_map[d][0] for d in ids if d in doc_id_map]
                    source_nodes = self._store.get_nodes(doc_ids=target_doc_ids, group=group_name, kb_id=target_kb_id)
                    if source_nodes:
                        uid_map = {n.uid: n.uid for n in source_nodes}
                        self._copy_segments_recursive(ids=ids, kb_id=kb_id, target_kb_id=target_kb_id,
                                                      doc_id_map=doc_id_map, p_uid_map=uid_map,
                                                      p_name=group_name, node_groups=node_groups,
                                                      skip_ng_ids=skip_ng_ids)
                    continue
                source_nodes = self._store.get_nodes(doc_ids=ids, group=group_name, kb_id=kb_id)
                nodes = []
                uid_map = {}
                for source_node in source_nodes:
                    copied = self._clone_node_for_transfer(
                        node=source_node,
                        target_kb_id=target_kb_id,
                        target_doc_id=doc_id_map[source_node.global_metadata[RAG_DOC_ID]][0],
                        metadata=doc_id_map[source_node.global_metadata[RAG_DOC_ID]][1],
                    )
                    uid_map[source_node.uid] = copied.uid
                    copied.parent = p_uid_map.get(source_node.parent, None) if source_node.parent else None
                    nodes.append(copied)
                self._store.update_nodes(nodes, copy=True)
                if nodes:
                    self._copy_segments_recursive(ids=ids, kb_id=kb_id, target_kb_id=target_kb_id,
                                                  doc_id_map=doc_id_map, p_uid_map=uid_map,
                                                  p_name=group_name, node_groups=node_groups,
                                                  skip_ng_ids=skip_ng_ids)

    def _create_nodes_impl(self, p_nodes, group_name, node_groups: Dict[str, Dict],
                           ref_path=None, skip_ng_ids: Optional[set] = None):
        # NOTE transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        doc_ids = list({n.global_metadata.get(RAG_DOC_ID) for n in p_nodes if n.global_metadata.get(RAG_DOC_ID)})
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, DEFAULT_KB_ID) if p_nodes else DEFAULT_KB_ID
        ng_cfg = node_groups.get(group_name, {})
        ng_id = ng_cfg.get('id') or group_name
        if skip_ng_ids and ng_id in skip_ng_ids:
            if not doc_ids:
                return []
            return self._store.get_nodes(doc_ids=doc_ids, group=group_name, kb_id=kb_id)
        t = node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, group_name)
        nodes = transform.batch_forward(p_nodes, group_name, ref_path=ref_path)
        self._store.update_nodes(self._set_nodes_number(nodes))
        return nodes

    def _get_or_create_nodes(self, group_name, node_groups: Dict[str, Dict],
                             uids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(uids=uids, group=group_name) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(node_groups[group_name]['parent'], node_groups, uids)
            nodes = self._create_nodes_impl(p_nodes, group_name, node_groups)
        return nodes

    def reparse(self, group_name: str, node_groups: Dict[str, Dict],
                uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None,
                kb_id: Optional[str] = None, **kwargs):
        if doc_ids:
            self._reparse_docs(group_name=group_name, node_groups=node_groups,
                               doc_ids=doc_ids, kb_id=kb_id, **kwargs)
        else:
            self._get_or_create_nodes(group_name, node_groups, uids)

    def _reparse_docs(self, group_name: str, node_groups: Dict[str, Dict],
                      doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict],
                      kb_id: str = None, reader: Optional[DirectoryReader] = None, **kwargs):
        doc_ids, metadatas, kb_id = self._prepare_doc_inputs(doc_paths, doc_ids, metadatas, kb_id)
        if group_name == 'all':
            preloaded_root_nodes = reader.load_data(doc_paths, metadatas, split_nodes_by_type=True)
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
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas, kb_id=kb_id,
                         node_groups=node_groups, reader=reader,
                         preloaded_root_nodes=preloaded_root_nodes)
            LOG.info(f'Reparse docs {doc_ids} from store done')
        else:
            p_nodes = self._store.get_nodes(group=node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name,
                                          node_groups=node_groups, doc_ids=doc_ids, kb_id=kb_id)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str,
                                 node_groups: Dict[str, Dict], doc_ids: List[str], kb_id: str = None):
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
        try:
            nodes = self._create_nodes_impl(p_nodes, cur_name, node_groups)
        except Exception:
            raise

        for group_name in self._store.activated_groups():
            group = node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group "{group_name}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if group['parent'] == cur_name:
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name,
                                              node_groups=node_groups, doc_ids=doc_ids, kb_id=kb_id)

    def update_doc_meta(self, doc_id: str, metadata: dict, kb_id: str = None):
        try:
            self._store.update_doc_meta(doc_id=doc_id, metadata=metadata, kb_id=kb_id)
        except Exception as e:
            LOG.error(f'Failed to update doc meta: {e}, {traceback.format_exc()}')
            raise e

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None,
                   node_group_ids_to_delete: Optional[List[str]] = None) -> None:
        try:
            self._store.remove_nodes(kb_id=kb_id, doc_ids=doc_ids,
                                     node_group_ids_to_delete=node_group_ids_to_delete)
            for ext in self._schema_extractors.values():
                ext._delete_extract_data(kb_id=kb_id, doc_ids=doc_ids)
        except Exception as e:
            LOG.error(f'Failed to delete doc: {e}, {traceback.format_exc()}')
            raise e
