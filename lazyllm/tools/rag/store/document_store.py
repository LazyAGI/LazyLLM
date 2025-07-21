from collections import defaultdict
from typing import Optional, List, Union, Set, Dict, Callable, Any
from lazyllm import LOG

from .store_base import (LazyLLMStoreBase, StoreCapability, SegmentType, Segment, INSERT_BATCH_SIZE,
                         BUILDIN_GLOBAL_META_DESC, DEFAULT_KB_ID, EMBED_PREFIX)
from .hybrid import HybridStore, MapStore, SenseCoreStore
from .segment import OpenSearchStore
from .vector import ChromadbStore, MilvusStore
from ..default_index import DefaultIndex
from ..utils import parallel_do_embedding

from ..doc_node import DocNode, QADocNode, ImageDocNode
from ..index_base import IndexBase
from ..data_type import DataType
from ..global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_KB_ID
from ..similarity import registered_similarities


STORE_FACTORIES = {
    "map": MapStore,
    "opensearch": OpenSearchStore,
    "chroma": ChromadbStore,
    "milvus": MilvusStore,
    "sensecore": SenseCoreStore,
}


class DocumentStore(object):
    def __init__(self, algo_name: str, store_config: Optional[Dict] = None, store: Optional[LazyLLMStoreBase] = None,
                 group_embed_keys: Optional[Dict[str, Set[str]]] = None, embed: Optional[Dict[str, Callable]] = None,
                 embed_dims: Optional[Dict[str, int]] = None, embed_datatypes: Optional[Dict[str, DataType]] = None,
                 global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None):
        self._algo_name = algo_name
        self._group_embed_keys = group_embed_keys
        self._embed = embed
        self._embed_dims = embed_dims
        self._embed_datatypes = embed_datatypes
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC

        self._activated_groups = set()
        self._indices = {}
        self._impl = self._create_store_from_config(store_config) if store_config else self._create_store(store)
        self._impl.lazy_init(embed_dims=self._embed_dims, embed_datatypes=self._embed_datatypes,
                             global_metadata_desc=self._global_metadata_desc)
        if isinstance(self._impl, MapStore): self._indices["default"] = DefaultIndex(embed, self)

    def _make_store(self, cfg: Dict[str, Any], deprecated_msg: str = None) -> LazyLLMStoreBase:
        if not cfg:
            return None
        stype = cfg["type"]
        cls = STORE_FACTORIES.get(stype)
        if not cls:
            raise NotImplementedError(f"Not implemented store type: {stype}")
        if deprecated_msg:
            LOG.warning(deprecated_msg)
        return cls(**cfg.get("kwargs", {}))

    def _handle_legacy(self, cfg: Dict[str, Any]) -> LazyLLMStoreBase:
        msg = "[DocumentStore] Single store in store_config is deprecated, use segment_store/vector_store instead"
        LOG.warning(msg)
        st = self._make_store(cfg, None)
        if st and st.capability == StoreCapability.ALL:
            return st
        if cfg["type"] in ("chroma", "milvus"):
            LOG.warning(f"[DocumentStore] Single {cfg['type']} store is deprecated …")
            segment = MapStore()
            vector = st if isinstance(st, (ChromadbStore, MilvusStore)) else None
            return HybridStore(segment_store=segment, vector_store=vector)
        return st

    def _handle_segment_only(self, seg: Any, cfg: Dict[str, Any]):
        if seg.capability == StoreCapability.ALL:
            return seg
        idx = cfg.get("indices", {}).get("smart_embedding_index")
        if idx:
            LOG.warning("[DocumentStore] 'smart_embedding_index' is deprecated, "
                        "please configure 'vector_store' instead")
            backend = idx["backend"]
            vec_cfg = {"type": backend, "kwargs": idx.get("kwargs", {})}
            vec = self._make_store(vec_cfg)
            return HybridStore(seg, vec)
        LOG.warning("[DocumentStore] Only segment_store provided; to use vector retrieval "
                    "please configure vector_store")
        return seg

    def _handle_vector_only(self, vec: Any):
        if vec.capability == StoreCapability.ALL:
            return vec
        LOG.warning("[DocumentStore] Only vector_store provided; segment will be in‑memory MapStore")
        return HybridStore(segment_store=MapStore(), vector_store=vec)

    def _create_store_from_config(self, cfg: Optional[Dict[str, Any]] = None):
        if cfg and cfg.get("type"):
            return self._handle_legacy(cfg)
        seg = self._make_store(cfg.get("segment_store", {}))
        vec = self._make_store(cfg.get("vector_store", {}))
        if not seg and not vec:
            LOG.warning("[DocumentStore] No store configured; defaulting to in‑memory MapStore")
            return MapStore()

        if seg and vec:
            return HybridStore(segment_store=seg, vector_store=vec)
        if seg:
            return self._handle_segment_only(seg, cfg)
        return self._handle_vector_only(vec)

    def _create_store(self, store: Optional[LazyLLMStoreBase] = None) -> LazyLLMStoreBase:
        if store.capability == StoreCapability.ALL:
            return store
        elif store.capability == StoreCapability.SEGMENT:
            return self._handle_segment_only(store, {})
        elif store.capability == StoreCapability.VECTOR:
            return self._handle_vector_only(store)
        else:
            raise ValueError("store must be a segment or vector store")

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
        return not self._impl.get(self._gen_collection_name(group), {})

    def update_nodes(self, nodes: List[DocNode]):
        if not nodes:
            return
        try:
            parallel_do_embedding(self._embed, [], nodes, self._group_embed_keys)
            group_segments = defaultdict(list)
            for node in nodes:
                group_segments[node._group].append(self._serialize_node(node))
            # upsert batch segments
            for group, segments in group_segments.items():
                if not self.is_group_active(group):
                    LOG.warning(f"[DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                for i in range(0, len(segments), INSERT_BATCH_SIZE):
                    self._impl.upsert(self._gen_collection_name(group), segments[i:i + INSERT_BATCH_SIZE])
            # update indices
            for index in self._indices.values():
                index.update(nodes)
        except Exception as e:
            LOG.error(f"[DocumentStore - {self._algo_name}] Failed to update nodes: {e}")
            raise

    def remove_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                     group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> None:
        # remove a set of nodes by uids
        # remove the nodes of the whole file -- doc ids only
        # remove the nodes of a certain group for one file -- doc ids and group (kb_id is optional)
        # forbid to remove the nodes from multiple kb
        try:
            if uids:
                criteria = {"uid": uids}
            else:
                criteria = {RAG_DOC_ID: doc_ids, RAG_KB_ID: kb_id or DEFAULT_KB_ID}
            if not group:
                groups = self._activated_groups
            else:
                groups = [group]
            for group in groups:
                if not self.is_group_active(group):
                    LOG.warning(f"[DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                self._impl.delete(self._gen_collection_name(group), criteria)
            # update indices
            for index in self._indices.values():
                index.remove(uids, group)
        except Exception as e:
            LOG.error(f"[DocumentStore - {self._algo_name}] Failed to remove nodes: {e}")
            raise

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> List[DocNode]:
        try:
            segments = self.get_segments(uids, doc_ids, group, kb_id, **kwargs)
            return [self._deserialize_node(segment) for segment in segments]
        except Exception as e:
            LOG.error(f"[DocumentStore - {self._algo_name}] Failed to get nodes: {e}")
            raise

    def get_segments(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                     group: Optional[str] = None, kb_id: Optional[str] = None, **kwargs) -> List[dict]:
        # get a set of segments by uids
        # get the segments of the whole file -- doc ids only
        # get the segments of a certain group for one file -- doc ids and group (kb_id is optional)
        # forbid to get the segments from multiple kb (only one kb_id is allowed)
        # TODO: pagination
        try:
            if uids:
                criteria = {"uid": uids}
            else:
                criteria = {RAG_DOC_ID: doc_ids, RAG_KB_ID: kb_id or DEFAULT_KB_ID}
            if not group:
                groups = self._activated_groups
            else:
                groups = [group]
            segments = []
            for group in groups:
                if not self.is_group_active(group):
                    LOG.warning(f"[DocumentStore - {self._algo_name}] Group {group} is not active, skip")
                    continue
                segments.extend(self._impl.get(self._gen_collection_name(group), criteria))
            return segments
        except Exception as e:
            LOG.error(f"[DocumentStore - {self._algo_name}] Failed to get segments: {e}")
            raise

    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        kb_id = metadata.get(RAG_KB_ID, DEFAULT_KB_ID)
        segments = self.get_segments(doc_ids=[doc_id], kb_id=kb_id)
        if not segments:
            LOG.warning(f"[DocumentStore] No segments found for doc_id: {doc_id} in dataset: {kb_id}")
            return
        group_segments = defaultdict(list)
        for segment in segments:
            segment["global_meta"].update(metadata)
            group_segments[segment.get("group")].append(segment)
        for group, segments in group_segments.items():
            self._impl.upsert(self._gen_collection_name(group), segments)
        return

    def query(self, query: str, group_name: str, similarity_name: str,
              similarity_cut_off: Union[float, Dict[str, float]],
              topk: Optional[int] = 10, embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[DocNode]:
        self._validate_query_params(group_name, similarity_name, embed_keys)
        # temporary, when search in map store, use default index
        if isinstance(self._impl, MapStore):
            return self.get_index("default").query(query, group_name, similarity_name, similarity_cut_off,
                                                   topk, embed_keys, filters, **kwargs)
        nodes = []
        uid_score = {}
        if embed_keys:
            if self._impl.capability == StoreCapability.SEGMENT:
                raise ValueError(f"[DocumentStore - {self._algo_name}] Embed keys {embed_keys}"
                                 " are not supported when no vector store is provided")
            # vector search
            for embed_key in embed_keys:
                query_embedding = self._embed.get(embed_key)(query)
                search_res = self._impl.search(self._gen_collection_name(group_name), query_embedding,
                                               topk, filters, embed_key=embed_key)
                if search_res:
                    sim_cut_off = similarity_cut_off if isinstance(similarity_cut_off, float)\
                        else similarity_cut_off[embed_key]
                    for res in search_res:
                        if res["score"] < sim_cut_off:
                            continue
                        uid_score[res['uid']] = res['score'] if res['uid'] not in uid_score \
                            else max(uid_score[res['uid']], res['score'])
        else:
            # text search
            if self._impl.capability == StoreCapability.VECTOR:
                raise ValueError(f"[DocumentStore - {self._algo_name}] Text search is not"
                                 " supported when no segment store is provided")
            search_res = self._impl.search(self._gen_collection_name(group_name), query, topk, filters)
            if search_res:
                uid_score = {res['uid']: res['score'] for res in search_res}
        uids = list(uid_score.keys())
        if not uids: return []
        nodes = self.get_nodes(uids=uids, group=group_name)
        return [node.with_sim_score(uid_score[node._uid]) for node in nodes]

    def _validate_query_params(self, group_name: str, similarity: str,
                               embed_keys: Optional[List[str]] = None, **kwargs) -> bool:
        assert self.is_group_active(group_name), f"[DocumentStore - {self._algo_name}] Group {group_name} is not active"
        if similarity:
            if similarity in registered_similarities:
                _, mode, _ = registered_similarities[similarity]
                if mode == "embedding" and self._impl.capability == StoreCapability.SEGMENT:
                    raise ValueError(f"[DocumentStore - {self._algo_name}] Similarity {similarity} is not supported, "
                                     f"embedding similarity is supported for vector or hybrid store")
                elif mode == "text" and self._impl.capability == StoreCapability.VECTOR:
                    raise ValueError(f"[DocumentStore - {self._algo_name}] Similarity {similarity} is not supported, "
                                     "text similarity is supported for segment or hybrid store")
            else:
                raise ValueError(f"[DocumentStore - {self._algo_name}] Similarity {similarity} is not supported")

        assert not embed_keys or all(key in self._embed for key in embed_keys), \
            f"[DocumentStore - {self._algo_name}] Embed {embed_keys} are not supported"
        assert embed_keys and self._impl.capability != StoreCapability.SEGMENT, \
            f"[DocumentStore - {self._algo_name}] Embed {embed_keys} are not supported when no vector store is provided"
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
            self._impl.delete(self._gen_collection_name(group), {})

    def register_index(self, type: str, index: IndexBase) -> None:
        # TODO: By now, only map store support index registration
        assert isinstance(self._impl, MapStore), \
            f"[DocumentStore - {self._algo_name}] Only map store support index registration"
        self._indices[type] = index

    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        # TODO: By now, only map store support index registration
        assert isinstance(self._impl, MapStore), \
            f"[DocumentStore - {self._algo_name}] Only map store support index registration"
        if not type:
            type = "default"
        return self._indices.get(type)

    def _serialize_node(self, node: DocNode) -> dict:
        segment = Segment(
            uid=node._uid,
            doc_id=node.global_metadata.get(RAG_DOC_ID),
            group=node._group,
            content=node.text,
            meta=node.metadata,
            global_meta=node.global_metadata,
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
            res["embedding"] = {k: v for k, v in node.embedding.items()}
        return res

    def _deserialize_node(self, data: dict) -> DocNode:
        if data["type"] == SegmentType.QA.value:
            node = QADocNode(query=data["content"], answer=data["answer"], uid=data["uid"],
                             group=data["group"], parent=data["parent"],
                             metadata=data["meta"],
                             global_metadata=data["global_meta"])
        elif data["type"] == SegmentType.IMAGE.value:
            node = ImageDocNode(image_path=data["image_path"][0] if data["image_path"] else "",
                                uid=data["uid"], group=data["group"], parent=data["parent"],
                                metadata=data["meta"],
                                global_metadata=data["global_meta"])
        else:
            node = DocNode(uid=data["uid"], group=data["group"], content=data["content"],
                           parent=data["parent"], metadata=data["meta"],
                           global_metadata=data["global_meta"])
        node.excluded_embed_metadata_keys = data["excluded_embed_metadata_keys"]
        node.excluded_llm_metadata_keys = data["excluded_llm_metadata_keys"]
        if "embedding" in data:
            node.embedding = {k[len(EMBED_PREFIX):]: v for k, v in data.get("embedding", {}).items()}
        return node

    def _gen_collection_name(self, group: str) -> str:
        return f"col_{self._algo_name}_{group}".lower()
