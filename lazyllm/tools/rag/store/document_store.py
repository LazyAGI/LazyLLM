import json
from collections import defaultdict
from typing import Optional, List, Union, Set, Dict, Callable
from lazyllm import LOG

from .store_base import (LazyLLMStoreBase, StoreCapability, SegmentType, Segment, INSERT_BATCH_SIZE,
                         BUILDIN_GLOBAL_META_DESC, DEFAULT_KB_ID, EMBED_PREFIX)
from .hybrid_store import HybridStore
from .chroma_store import ChromadbStore
from .milvus_store import MilvusStore
from .sensecore_store import SenseCoreStore
from .map_store import MapStore
from ..default_index import DefaultIndex
from ..utils import parallel_do_embedding

from ..doc_node import DocNode, QADocNode, ImageDocNode
from ..index_base import IndexBase
from ..global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_KB_ID
from ..similarity import registered_similarities


class DocumentStore(object):
    def __init__(self, algo_name: str, store_config: Optional[Dict] = None,
                 segment_store: Optional[LazyLLMStoreBase] = None, vector_store: Optional[LazyLLMStoreBase] = None,
                 group_embed_keys: Optional[Dict[str, Set[str]]] = None, embed: Optional[Dict[str, Callable]] = None,
                 global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None):
        self._algo_name = algo_name
        self._validate_params(store_config, segment_store, vector_store)
        self._impl = self._create_store_from_config(store_config) if store_config\
            else self._create_store(segment_store, vector_store)
        self._group_embed_keys = group_embed_keys
        self._embed = embed
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self._activated_groups = set()
        self._indices = {}
        if isinstance(self._impl, MapStore): self._indices["default"] = DefaultIndex(embed, self)

    def _validate_params(self, store_config: Optional[Dict] = None, segment_store: Optional[LazyLLMStoreBase] = None,
                         vector_store: Optional[LazyLLMStoreBase] = None) -> bool:
        if segment_store:
            assert segment_store.capability in (StoreCapability.SEGMENT, StoreCapability.ALL), \
                "segment_store must be a segment store"
        if vector_store:
            assert vector_store.capability in (StoreCapability.VECTOR, StoreCapability.ALL), \
                "vector_store must be a vector store"
        if store_config and store_config.get("indices"):
            LOG.warning("indices is deprecated.")
        return True

    def _create_store_from_config(self, store_config: Optional[Dict] = None):
        if store_config.get("type"):
            # 向前兼容, {"type", "kwargs"}
            LOG.warning("[DocumentStore] store_config is deprecated, please use segment_store and vector_store instead")
            store_type = store_config.get("type")
            if store_type == "map":
                return MapStore()
            elif store_type == "chroma":
                return HybridStore(segment_store=MapStore(), vector_store=ChromadbStore())
            elif store_type == "milvus":
                return HybridStore(segment_store=MapStore(), vector_store=MilvusStore())
            elif store_type == "sensecore":
                return SenseCoreStore()
            else:
                raise NotImplementedError(f"Not implemented store type for {store_type}")
        else:
            return None

    def _create_store(self, segment_store: Optional[LazyLLMStoreBase] = None,
                      vector_store: Optional[LazyLLMStoreBase] = None) -> LazyLLMStoreBase:
        if segment_store and vector_store:
            return HybridStore(segment_store, vector_store)
        elif segment_store:
            if segment_store.capability == StoreCapability.SEGMENT:
                LOG.warning("[DocumentStore] There's only segment store, if you want to use vector retrieval, "
                            "please provide a vector store!")
                return segment_store
            else:
                return segment_store
        elif vector_store:
            if vector_store.capability == StoreCapability.VECTOR:
                LOG.warning("[DocumentStore] There's only vector store provided, segment wiil be store in ram "
                            "(not recommended)")
                return HybridStore(segment_store=MapStore(), vector_store=vector_store)
            else:
                return vector_store
        else:
            raise ValueError("segment_store or vector_store is required")

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
            global_meta = json.loads(segment["global_meta"])
            global_meta.update(metadata)
            segment["global_meta"] = json.dumps(global_meta, ensure_ascii=False)
            group_segments[segment.get("group")].append(segment)
        for group, segments in group_segments.items():
            self._impl.upsert(self._gen_collection_name(group), segments)
        return

    def query(self, query: str, group_name: str, similarity: str, similarity_cut_off: Union[float, Dict[str, float]],
              topk: Optional[int] = 10, embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[DocNode]:
        self._validate_query_params(group_name, similarity, embed_keys)
        # temporary, when search in map store, use default index
        if isinstance(self._impl, MapStore):
            return self.get_index("default").query(query, group_name, similarity, similarity_cut_off,
                                                   topk, embed_keys, filters, **kwargs)
        nodes = []
        uid_score = {}
        if embed_keys:
            # vector search
            for embed_key in embed_keys:
                query_embedding = self._embed.get(embed_key)(query)
                search_res = self._impl.search(self._gen_collection_name(group_name), query_embedding,
                                               topk, filters, embed_key=self._gen_embed_key(embed_key))
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
            search_res = self._impl.search(self._gen_collection_name(group_name), query, topk, filters)
            if search_res:
                uid_score = {res['uid']: res['score'] for res in search_res}
        uids = list(uid_score.keys())
        if not uids: return []
        nodes = self.get_nodes(uids=uids)
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
            f"[DocumentStore - {self._algo_name}] Embed keys {embed_keys} are not supported"
        assert embed_keys and self._impl.capability != StoreCapability.SEGMENT, \
            f"[DocumentStore - {self._algo_name}] Embed keys {embed_keys} are not supported for segment store"
        return True

    def find(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        pass

    def find_parent(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        pass

    def find_children(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        pass

    def clear_cache(self, groups: Optional[List[str]] = None) -> None:
        if groups is None:
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
            meta=json.dumps(node.metadata, ensure_ascii=False),
            global_meta=json.dumps(node.global_metadata, ensure_ascii=False),
            kb_id=node.global_metadata.get(RAG_KB_ID, DEFAULT_KB_ID),
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            parent=node.parent,
        )
        if isinstance(node, QADocNode):
            segment.type = SegmentType.QA
            segment.answer = node.answer
        elif isinstance(node, ImageDocNode):
            segment.type = SegmentType.IMAGE
            segment.image_keys = [node.image_path] if node.image_path else []
        res = segment.model_dump()
        # For speed up, add embedding after serialization
        if node.embedding:
            res["embedding"] = {self._gen_embed_key(k): v for k, v in node.embedding.items()}
        return res

    def _deserialize_node(self, data: dict) -> DocNode:
        if data["type"] == SegmentType.QA:
            node = QADocNode(query=data["content"], answer=data["answer"], uid=data["uid"],
                             group=data["group"], parent=data["parent"],
                             metadata=json.loads(data["meta"]),
                             global_metadata=json.loads(data["global_meta"]))
        elif data["type"] == SegmentType.IMAGE:
            node = ImageDocNode(image_path=data["image_path"][0] if data["image_path"] else "",
                                uid=data["uid"], group=data["group"], parent=data["parent"],
                                metadata=json.loads(data["meta"]),
                                global_metadata=json.loads(data["global_meta"]))
        else:
            node = DocNode(uid=data["uid"], group=data["group"], content=data["content"],
                           parent=data["parent"], metadata=json.loads(data["meta"]),
                           global_metadata=json.loads(data["global_meta"]))
        node.excluded_embed_metadata_keys = data["excluded_embed_metadata_keys"]
        node.excluded_llm_metadata_keys = data["excluded_llm_metadata_keys"]
        if "embedding" in data:
            node.embedding = {k[len(EMBED_PREFIX):]: v for k, v in data.get("embedding", {}).items()}
        return node

    def _gen_embed_key(self, key: str) -> str:
        return f"{EMBED_PREFIX}_{key}".lower()

    def _gen_collection_name(self, group: str) -> str:
        return f"col_{self._algo_name}_{group}".lower()
