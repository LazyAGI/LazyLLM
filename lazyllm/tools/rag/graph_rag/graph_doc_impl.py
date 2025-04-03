import json
import ast
from functools import wraps
from typing import Callable, Dict, List, Optional, Union, Any, TypedDict
from lazyllm import LOG, once_wrapper, globals
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from ..doc_node import DocNode
from ..data_loaders import DirectoryReader
from ..utils import DocListManager, is_sparse
from .graph_network_store import BaseGraphNetworkStore
from .graph_er_store import BaseGraphERStore
from .graph_chunk_store import BaseGraphChunkStore
from .graph_node import GraphChunkNode, GraphEntityNode, GraphRelationNode, EntityDict, RelationDict
from .prompt import PROMPTS as GraphRAGPrompts
from lazyllm.tools.rag.transform import SentenceSplitter
import re
import tiktoken


def embed_wrapper(func):
    if not func:
        return None

    @wraps(func)
    def wrapper(*args, **kwargs) -> List[float]:
        result = func(*args, **kwargs)
        return ast.literal_eval(result) if isinstance(result, str) else result

    return wrapper


class GraphDocReaderConf(TypedDict):
    chunk_size: int  # Default 1200
    chunk_overlap: int  # Default 100


class GraphDocStoreConf(TypedDict):
    root_path: str
    name_space: str
    chunk_store_type: str
    chunk_store_config: Dict[str, Any]
    er_store_type: str
    er_store_config: Dict[str, Any]
    network_store_type: str
    network_store_config: Dict[str, Any]


class GraphDocImpl:
    _registered_file_reader: Dict[str, Callable] = {}
    EXAMPLE_NUMBER = 1
    TRUNKCATE_MAX_TOKEN_NUM = 4000

    def __init__(
        self,
        embed: Callable,
        llm: ModuleBase,
        group_name: str,
        reader_conf: GraphDocReaderConf,
        store_conf: GraphDocStoreConf,
        document_module_id: str,
        dlm: Optional[DocListManager] = None,
        doc_files: Optional[str] = None,
    ):
        super().__init__()
        self._local_file_reader: Dict[str, Callable] = {}
        self._embed = embed_wrapper(embed)
        assert store_conf is not None
        # self.group_name = group_name
        self._root_path = store_conf["root_path"]
        self._name_space = store_conf["name_space"]
        self._reader_conf = reader_conf
        self._store_conf = store_conf  # NOTE: will be used in _lazy_init()
        self._graph_er_store = None
        self._graph_network_store = None
        self._graph_chunk_store = None
        self._document_module_id = document_module_id
        self._tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self._text_splitter = SentenceSplitter(
            chunk_size=reader_conf.get("chunk_size", 1200), chunk_overlap=reader_conf.get("chunk_overlap", 100)
        )
        self._activated_embeddings = {}
        examples = "\n".join(GraphRAGPrompts["keywords_extraction_examples"][: self.EXAMPLE_NUMBER])
        self._kws_extract_prompter = ChatPrompter(
            instruction=GraphRAGPrompts["keywords_extraction"].format(
                examples=examples, history="{history}", query="{query}"
            )
        ).pre_hook(self._kws_extract_prompt_hook)
        self._reader = DirectoryReader(None, self._local_file_reader, GraphDocImpl._registered_file_reader)
        # self.node_groups = {self.group_name}
        self._llm_kws_extractor = llm.share(prompt=self._kws_extract_prompter).used_by(document_module_id)

    def _kws_extract_prompt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        if not isinstance(input, str):
            raise ValueError(f"Unexpected type for input: {type(input)}")
        history_info = ""
        if self._document_module_id in globals["chat_history"]:
            history_info = chat_history_to_str(globals["chat_history"][self._document_module_id])
        return (
            dict(history=history_info, query=input, additional_query=""),
            history,
            tools,
            label,
        )

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self) -> None:
        # TODO add processing for doc_files
        embedding = self._embed('a')
        assert is_sparse(embedding) is False
        embedding_dim = len(embedding)

        # graph_er_store and graph_network_store must be initialized
        self._graph_er_store = BaseGraphERStore.create_instance(
            self._store_conf['er_store_type'],
            embed=self._embed,
            root_path=self._root_path,
            name_space=self._name_space,
            config=dict(self._store_conf['er_store_config'], embedding_dim=embedding_dim),
        )
        self._graph_network_store = BaseGraphNetworkStore.create_instance(
            self._store_conf['network_store_type'],
            root_path=self._root_path,
            name_space=self._name_space,
            config=self._store_conf['network_store_config'],
        )
        self._graph_chunk_store = BaseGraphChunkStore.create_instance(
            self._store_conf['chunk_store_type'],
            root_path=self._root_path,
            name_space=self._name_space,
            config=self._store_conf['chunk_store_config'],
        )

    def _add_doc_to_kg(
        self, input_files: List[str], ids: Optional[List[str]] = None, metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        # TODO: 1. Process chunk 2. Extract entity/relation/graph 3. Update kg graph
        raise NotImplementedError

    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        assert callable(func), 'func for reader should be callable'
        self._local_file_reader[pattern] = func

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        if func is not None:
            cls._registered_file_reader[pattern] = func
            return None

        def decorator(klass):
            if callable(klass):
                cls._registered_file_reader[pattern] = klass
            else:
                raise TypeError(f"The registered object {klass} is not a callable object.")
            return klass

        return decorator

    def extract_keywrods(self, query: str):
        result = self._llm_kws_extractor(query)

        # 6. Parse out JSON from the LLM response
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if not match:
            LOG.warning("No JSON-like structure found in the LLM respond.")
            return [], []
        try:
            keywords_data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            LOG.warning(f"JSON parsing error: {e}")
            return [], []

        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        LOG.info(f"Extracted result: high level keywords: {hl_keywords}, low level keywords: {ll_keywords}")
        return hl_keywords, ll_keywords

    def truncate_list_by_token_size(self, list_data: list, key: callable):
        if self.TRUNKCATE_MAX_TOKEN_NUM <= 0:
            return []
        tokens = 0
        for i, data in enumerate(list_data):
            tokens += len(self._tiktoken_tokenizer.encode(key(data)))
            if tokens > self.TRUNKCATE_MAX_TOKEN_NUM:
                return list_data[:i]
        return list_data

    def _find_related_entities(self, query: str, topk: int, similarity_cut_off: float) -> List[GraphEntityNode]:
        entities_dict: EntityDict = self._graph_er_store.query_on_entity(query, topk, similarity_cut_off)
        related_entities = [
            self._graph_network_store.get_node(entity_dict["entity_name"]) for entity_dict in entities_dict
        ]
        related_entities = [ele for ele in related_entities if ele]
        return related_entities

    def _find_related_relations(
        self, query: str, topk: int = 30, similarity_cut_off: float = 0.3
    ) -> List[GraphRelationNode]:
        relation_keys: List[RelationDict] = self._graph_er_store.query_on_relationship(query, topk, similarity_cut_off)
        related_relations = [self._graph_network_store.get_edge(key["src_id"], key["tgt_id"]) for key in relation_keys]
        related_relations = [ele for ele in related_relations if ele]
        return related_relations

    def _find_most_related_chunkids_from_entities(self, entities: List[GraphEntityNode]) -> List[GraphChunkNode]:
        ed_ent = 0
        acc_token_num = 0
        selected_chunkids = set()
        while ed_ent < len(entities):
            token_num_cur_entity = 0
            for cid in entities[ed_ent].source_chunk_ids:
                if cid not in selected_chunkids:
                    chunk = self._graph_chunk_store.get_chunk(cid)
                    if not chunk:
                        continue
                    token_num_cur_entity += chunk.tokens
            if acc_token_num + token_num_cur_entity <= self.TRUNKCATE_MAX_TOKEN_NUM:
                acc_token_num += token_num_cur_entity
                selected_chunkids.update(entities[ed_ent].source_chunk_ids)
            else:
                break
            ed_ent += 1
        # If all entities are selected return without truncating
        if ed_ent == len(entities):
            return list(selected_chunkids)
        # Else truncate entities[ed].chunk_ids before adding to selected chunkids
        sorted_chunkids = self._graph_network_store.sort_entitity_chunkids(entities[ed_ent])
        ed_sub_chunk = 0
        while ed_sub_chunk < len(sorted_chunkids):
            cid = sorted_chunkids[ed_sub_chunk]
            if cid not in selected_chunkids:
                chunk = self._graph_chunk_store.get_chunk(cid)
                acc_token_num += chunk.tokens
                if acc_token_num > self.TRUNKCATE_MAX_TOKEN_NUM:
                    break
            ed_sub_chunk += 1
        selected_chunkids.update(sorted_chunkids[:ed_sub_chunk])
        return list(selected_chunkids)

    def _find_most_related_relations_from_entities(self, entities: List[str]) -> List[GraphRelationNode]:
        related_relations = self._graph_network_store.get_sorted_relations_from_entities(entities)
        truncated_related_relations = self.truncate_list_by_token_size(
            related_relations,
            key=lambda x: x.description,
        )
        return truncated_related_relations

    def _find_most_related_entities_from_relationships(
        self, relations: List[GraphRelationNode]
    ) -> List[GraphEntityNode]:
        pass

    def _find_related_chunks_from_relationships(self, relationships: List[GraphRelationNode]) -> List[GraphChunkNode]:
        pass

    def _generate_doc_nodes(
        self,
        related_eneities: List[GraphEntityNode],
        related_relations: List[GraphRelationNode],
        related_chunks: List[GraphChunkNode],
    ) -> List[DocNode]:
        str_content = ""
        # 1. Format entities
        str_content += '\n-----Entities-----\n```csv\n"entity","type","description"\n'
        for entity in related_eneities:
            str_content += f'"{entity.entity_name}","{entity.entity_type}","{entity.description}"\n'
        str_content += '\n```'
        # 2. Format relations
        str_content += '\n-----Relationships-----\n```csv\n"source_entity","target_entity","description","keywords"\n'
        for relation in related_relations:
            str_content += f'"{relation.src_id}","{relation.tgt_id}","{relation.description}","{relation.keywords}"\n'
        str_content += '\n```'
        # 3. Format chunks
        str_content += '\n-----Source Chunks-----\n```csv\n"content"\n'
        for chunk in related_chunks:
            str_content += f'"{chunk.content}"\n'
        str_content += '\n```\n'
        return [DocNode(content=str_content)]

    def _retrieve_local(self, ll_keywords: str, topk: int, similarity_cut_off: float) -> List[DocNode]:
        LOG.info("RUNNING LOCAL MODE")
        related_eneities: List[GraphEntityNode] = self._find_related_entities(ll_keywords, topk, similarity_cut_off)
        related_chunkids = self._find_most_related_chunkids_from_entities(related_eneities)
        related_chunks: List[GraphChunkNode] = self._graph_chunk_store.get_chunks(related_chunkids)
        related_relations: List[GraphRelationNode] = self._find_most_related_relations_from_entities(related_eneities)
        return self._generate_doc_nodes(related_eneities, related_relations, related_chunks)

    def _retrieve_global(self, hl_keywords: str):
        raise NotImplementedError

    def retrieve(self, query: str, topk: int = 30, similarity_cut_off: float = 0.3, **kwargs) -> List[DocNode]:
        self._lazy_init()

        hl_keywords, ll_keywords = self.extract_keywrods(query)
        str_ll_keywords = ", ".join(ll_keywords)
        str_hl_keywords = ", ".join(hl_keywords)
        if ll_keywords:
            return self._retrieve_local(str_ll_keywords, topk=topk, similarity_cut_off=similarity_cut_off)
        else:
            raise NotImplementedError(f"high level retrieval not implemented yet, str_hl_keywords: {str_hl_keywords}")
            # return self._retrieve_global(str_hl_keywords, topk=topk, similarity_cut_off=similarity_cut_off)

    def __call__(self, func_name: str, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)
