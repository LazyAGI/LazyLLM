from collections import defaultdict
from typing import Optional, List, Union, Set, Dict, Callable, Any
import lazyllm
from lazyllm import LOG, once_wrapper

from .store_base import (LazyLLMStoreBase, StoreCapability, SegmentType, Segment, INSERT_BATCH_SIZE,
                         BUILDIN_GLOBAL_META_DESC, DEFAULT_KB_ID)
from .hybrid import HybridStore, MapStore
from ..default_index import DefaultIndex
from ..utils import parallel_do_embedding

from ..doc_node import DocNode, QADocNode, ImageDocNode
from ..index_base import IndexBase
from ..data_type import DataType
from ..global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_KB_ID
from ..similarity import registered_similarities


class _DocumentStore(object):
    def __init__(self, algo_name: str, store: Union[Dict, LazyLLMStoreBase],
                 group_embed_keys: Optional[Dict[str, Set[str]]] = None, embed: Optional[Dict[str, Callable]] = None,
                 embed_dims: Optional[Dict[str, int]] = None, embed_datatypes: Optional[Dict[str, DataType]] = None,
                 global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None):
        self._algo_name = algo_name
        self._group_embed_keys = group_embed_keys
        self._embed = embed
        self._embed_dims = embed_dims
        self._embed_datatypes = embed_datatypes
        self._global_metadata_desc = (global_metadata_desc or {}) | BUILDIN_GLOBAL_META_DESC
        self._activated_groups = set()
        self._indices = {}
        self._impl = self._prepare_store(store)
        if self._impl.supports_index_registration:
            self._indices['default'] = DefaultIndex(self._embed, self)

    def _prepare_store(self, store: Union[Dict, LazyLLMStoreBase]) -> LazyLLMStoreBase:
        if isinstance(store, dict):
            # create store from store config
            if store.get('indices'): store = self._convert_legacy_to_config(store)
            store = self._create_store_from_config(store)
        if store.capability == StoreCapability.VECTOR:
            segment_store = MapStore()
            return HybridStore(segment_store=segment_store, vector_store=store)
        else:
            return store

    def _make_store(self, cfg: Dict[str, Any]) -> LazyLLMStoreBase:
        if not cfg: return None
        stype = cfg.get('type')
        cls = getattr(lazyllm.store, stype, None)
        if not cls:
            raise NotImplementedError(f"Not implemented store type: {stype}")
        return cls(**cfg.get('kwargs', {}))

    def _convert_legacy_to_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        LOG.warning("[_DocumentStore] 'smart_embedding_index' is deprecated, will be converted to milvus type config")
        indices = cfg.pop('indices')
        backend = indices.get('backend')
        if not backend:
            raise ValueError("backend is required in indices")
        cfg = {'type': backend, 'kwargs': indices.get('kwargs', {})}
        return cfg

    def _create_store_from_config(self, cfg: Optional[Dict[str, Any]] = None) -> LazyLLMStoreBase:
        if cfg.get('type'):
            # map/sensecore/chroma/milvus
            s_store = self._make_store(cfg)
            if s_store.capability in (StoreCapability.ALL, StoreCapability.SEGMENT):
                return s_store
            else:
                LOG.warning("Only vector store is supported, segment store will be in‑memory MapStore")
                return HybridStore(segment_store=MapStore(), vector_store=s_store)
        else:
            seg_store = self._make_store(cfg.get('segment_store', {}))
            vec_store = self._make_store(cfg.get('vector_store', {}))
            assert seg_store and vec_store, "Please provide both segment_store and vector_store in store_config"
            assert seg_store.capability in (StoreCapability.ALL, StoreCapability.SEGMENT), \
                "Segment store must be a segment store"
            assert vec_store.capability in (StoreCapability.ALL, StoreCapability.VECTOR), \
                "Vector store must be a vector store"
            return HybridStore(segment_store=seg_store, vector_store=vec_store)

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self):
        self._impl.connect(embed_dims=self._embed_dims, embed_datatypes=self._embed_datatypes,
                           global_metadata_desc=self._global_metadata_desc, collections=self.activated_groups())

    @property
    def impl(self):
        self._lazy_init()
        return self._impl

    def activate_group(self, groups: Union[str, List[str]]) -> bool:
        if isinstance(groups, str):
            groups = [groups]
        for group in groups:
            if group not in self._activated_groups:
                self._activated_groups.add(group)
        return True

    def activated_groups(self) -> List[str]:
        return list(self._activated_groups)

    def is_group_active(self, group: str) -> bool:
        return group in self._activated_groups

    def is_group_empty(self, group: str) -> bool:
        return not self.impl.get(self._gen_collection_name(group), {})

    def update_nodes(self, nodes: List[DocNode]):   # noqa: C901
        if not nodes:
            return
        try:
            if self.impl.need_embedding:
                parallel_do_embedding(self._embed, [], nodes, self._group_embed_keys)
            group_segments = defaultdict(list)
            for node in nodes:
                group_segments[node._group].append(self._serialize_node(node))
            # upsert batch segments
            for group, segments in group_segments.items():
                if not self.is_group_active(group):
                    LOG.warning(f"[_DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                for i in range(0, len(segments), INSERT_BATCH_SIZE):
                    self.impl.upsert(self._gen_collection_name(group), segments[i:i + INSERT_BATCH_SIZE])
            # update indices
            for index in self._indices.values():
                index.update(nodes)
        except Exception as e:
            LOG.error(f"[_DocumentStore - {self._algo_name}] Failed to update nodes: {e}")
            raise

    def remove_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                     group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> None:
        # remove a set of nodes by uids
        # remove the nodes of the whole file -- doc ids only
        # remove the nodes of a certain group for one file -- doc ids and group (kb_id is optional)
        # forbid to remove the nodes from multiple kb
        try:
            criteria = {}
            if uids:
                criteria = {'uid': uids}
            if doc_ids:
                criteria[RAG_DOC_ID] = doc_ids
            if kb_id:
                criteria[RAG_KB_ID] = kb_id
            if not group:
                groups = self._activated_groups
            else:
                groups = [group]
            for group in groups:
                if not self.is_group_active(group):
                    LOG.warning(f"[_DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                self.impl.delete(self._gen_collection_name(group), criteria)
            # update indices
            for index in self._indices.values():
                index.remove(uids, group)
        except Exception as e:
            LOG.error(f"[_DocumentStore - {self._algo_name}] Failed to remove nodes: {e}")
            raise

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> List[DocNode]:
        try:
            segments = self.get_segments(uids, doc_ids, group, kb_id, **kwargs)
            return [self._deserialize_node(segment) for segment in segments]
        except Exception as e:
            LOG.error(f"[_DocumentStore - {self._algo_name}] Failed to get nodes: {e}")
            raise

    def get_segments(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                     group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> List[dict]:
        # get a set of segments by uids
        # get the segments of the whole file -- doc ids only
        # get the segments of a certain group for one file -- doc ids and group (kb_id is optional)
        # forbid to get the segments from multiple kb (only one kb_id is allowed)
        # TODO: pagination
        try:
            criteria = {}
            if uids:
                criteria = {'uid': uids}
            if doc_ids:
                criteria = {RAG_DOC_ID: doc_ids}
            if kb_id:
                criteria[RAG_KB_ID] = kb_id
            if not group:
                groups = self._activated_groups
            else:
                groups = [group]
            segments = []
            for group in groups:
                if not self.is_group_active(group):
                    LOG.warning(f"[_DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                segments.extend(self.impl.get(self._gen_collection_name(group), criteria, **kwargs))
            return segments
        except Exception as e:
            LOG.error(f"[_DocumentStore - {self._algo_name}] Failed to get segments: {e}")
            raise

    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        kb_id = metadata.get(RAG_KB_ID, None)
        segments = self.get_segments(doc_ids=[doc_id], kb_id=kb_id)
        if not segments:
            LOG.warning(f"[_DocumentStore] No segments found for doc_id: {doc_id} in dataset: {kb_id}")
            return
        group_segments = defaultdict(list)
        for segment in segments:
            segment['global_meta'].update(metadata)
            group_segments[segment.get('group')].append(segment)
        for group, segments in group_segments.items():
            self.impl.upsert(self._gen_collection_name(group), segments)
        LOG.info(f"[_DocumentStore] Updated metadata for doc_id: {doc_id} in dataset: {kb_id}")
        return

    def query(self, query: str, group_name: str, similarity_name: Optional[str] = None,
              similarity_cut_off: Union[float, Dict[str, float]] = float('-inf'),
              topk: Optional[int] = 10, embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[DocNode]:
        self._validate_query_params(group_name, similarity_name, embed_keys)
        segments = []
        if embed_keys:
            if self.impl.capability == StoreCapability.SEGMENT:
                raise ValueError(f"[_DocumentStore - {self._algo_name}] Embed keys {embed_keys}"
                                 " are not supported when no vector store is provided")
            # vector search
            for embed_key in embed_keys:
                query_embedding = self._embed.get(embed_key)(query)
                search_res = self.impl.search(collection_name=self._gen_collection_name(group_name),
                                              query=query, query_embedding=query_embedding,
                                              topk=topk, filters=filters, embed_key=embed_key, **kwargs)
                if search_res:
                    sim_cut_off = similarity_cut_off if isinstance(similarity_cut_off, float)\
                        else similarity_cut_off[embed_key]
                    segments.extend([res for res in search_res if res.get('score', 0) >= sim_cut_off])
        else:
            # text search
            if self.impl.capability == StoreCapability.VECTOR:
                raise ValueError(f"[_DocumentStore - {self._algo_name}] Text search is not"
                                 " supported when no segment store is provided")
            segments.extend(self.impl.search(collection_name=self._gen_collection_name(group_name),
                                             query=query, topk=topk, filters=filters, **kwargs))
        return [self._deserialize_node(segment, segment.get('score', 0)) for segment in segments]

    def _validate_query_params(self, group_name: str, similarity: str,
                               embed_keys: Optional[List[str]] = None, **kwargs) -> bool:
        assert self.is_group_active(group_name), f"[_DocumentStore - {self._algo_name}] Group {group_name} is not active"
        if similarity:
            if similarity in registered_similarities:
                _, mode, _ = registered_similarities[similarity]
                if mode == 'embedding' and self.impl.capability == StoreCapability.SEGMENT:
                    raise ValueError(f"[_DocumentStore - {self._algo_name}] Similarity {similarity} is not supported, "
                                     f"embedding similarity is supported for vector or hybrid store")
                elif mode == 'text' and self.impl.capability == StoreCapability.VECTOR:
                    raise ValueError(f"[_DocumentStore - {self._algo_name}] Similarity {similarity} is not supported, "
                                     "text similarity is supported for segment or hybrid store")
            else:
                raise ValueError(f"[_DocumentStore - {self._algo_name}] Similarity {similarity} is not supported")

        if embed_keys:
            assert self.impl.capability != StoreCapability.SEGMENT, \
                f"[_DocumentStore - {self._algo_name}] Embed {embed_keys} not supported when no vector store provided"
            assert all(key in self._embed for key in embed_keys), \
                f"[_DocumentStore - {self._algo_name}] Embed {embed_keys} not supported"
        return True

    def clear_cache(self, groups: Optional[List[str]] = None) -> None:
        if not groups:
            groups = self._activated_groups
        elif isinstance(groups, str):
            groups = [groups]
        elif isinstance(groups, (tuple, list, set)):
            groups = list(groups)
        else:
            raise TypeError(f"Invalid type {type(groups)} for groups, expected list of str")
        for group in groups:
            self.impl.delete(self._gen_collection_name(group))

    def register_index(self, type: str, index: IndexBase) -> None:
        assert self._impl.supports_index_registration, \
            f"[_DocumentStore - {self._algo_name}] Store {type(self.impl)} does not support index registration"
        self._indices[type] = index

    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        return self._indices.get(type)

    def _serialize_node(self, node: DocNode) -> dict:
        segment = Segment(
            uid=node._uid,
            doc_id=node.global_metadata.get(RAG_DOC_ID),
            group=node._group,
            content=node.text,
            meta=node.metadata,
            global_meta=node.global_metadata,
            number=node.metadata.get('number', 0),
            kb_id=node.global_metadata.get(RAG_KB_ID, DEFAULT_KB_ID),
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
        )
        if node.parent:
            segment.parent = node.parent._uid if isinstance(node.parent, DocNode) else node.parent
        if isinstance(node, QADocNode):
            segment.type = SegmentType.QA.value
            segment.answer = node.answer
        elif isinstance(node, ImageDocNode):
            segment.type = SegmentType.IMAGE.value
            segment.image_keys = [node.image_path] if node.image_path else []
        res = segment.model_dump()
        # For speed up, add embedding after serialization
        if node.embedding:
            res['embedding'] = {k: v for k, v in node.embedding.items()}
        return res

    def _deserialize_node(self, data: dict, score: Optional[float] = None) -> DocNode:
        segment_type = data.get('type', SegmentType.TEXT.value)
        if segment_type == SegmentType.QA.value:
            node = QADocNode(query=data.get('content', ''), answer=data.get('answer', ''), uid=data['uid'],
                             group=data['group'], parent=data.get('parent', ''),
                             metadata=data.get('meta', {}),
                             global_metadata=data.get('global_meta', {}))
        elif segment_type == SegmentType.IMAGE.value:
            if not data.get('image_keys', []):
                raise ValueError("ImageDocNode does have any image_keys")
            node = ImageDocNode(image_path=data.get('image_keys')[0],
                                uid=data['uid'], group=data['group'], parent=data.get('parent', ''),
                                metadata=data.get('meta', {}),
                                global_metadata=data.get('global_meta', {}))
        else:
            node = DocNode(uid=data['uid'], group=data['group'], content=data.get('content', ''),
                           parent=data.get('parent', ''), metadata=data.get('meta', {}),
                           global_metadata=data.get('global_meta', {}))
        node.excluded_embed_metadata_keys = data.get('excluded_embed_metadata_keys', [])
        node.excluded_llm_metadata_keys = data.get('excluded_llm_metadata_keys', [])
        if 'embedding' in data:
            node.embedding = {k: v for k, v in data.get('embedding', {}).items()}
        return node.with_sim_score(score) if score else node

    def _gen_collection_name(self, group: str) -> str:
        return f"col_{self._algo_name}_{group}".lower()
