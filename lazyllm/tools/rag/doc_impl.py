from functools import partial
from typing import Dict, List, Optional, Set
from lazyllm import ModuleBase, LOG
from lazyllm.common import LazyLlmRequest
from .parser import FuncNodeParser, SentenceSplitter
from .store import MapStore, DocNode
from .data_loaders import DirectoryReader
import ast
from .index import DefaultIndex


class DocImplV2:
    def __init__(self, embed, doc_files=Optional[List[str]], **kwargs):
        super().__init__()
        self.embed = embed
        self.directory_reader = DirectoryReader(input_files=doc_files)
        self.parser_dict: Dict[str, Dict] = {}
        self.create_node_group_default()
        self.store = MapStore()
        self.index = DefaultIndex()

    def create_node_group_default(self):
        self.create_node_group(
            name="CoarseChunk",
            transform=SentenceSplitter,
            chunk_size=1024,
            chunk_overlap=100,
        )
        self.create_node_group(
            name="MediumChunk",
            transform=SentenceSplitter,
            chunk_size=256,
            chunk_overlap=25,
        )
        self.create_node_group(
            name="FineChunk",
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=12,
        )
        # TODO: SentenceDivider is deprecated and will be removed in the future
        self.create_node_group(
            name="SentenceDivider",
            transform=SentenceSplitter,
            chunk_size=1024,
            chunk_overlap=20,
        )
        self.create_node_group(
            name="SentenceSplitter",
            transform=SentenceSplitter,
            chunk_size=1024,
            chunk_overlap=20,
        )

    def create_node_group(self, name, transform, parent="root", **kwargs) -> None:
        if name in self.parser_dict:
            LOG.warning(f"Duplicate parser rule: {name}")
        assert callable(transform), "transform should be callable"
        self.parser_dict[name] = dict(
            parser=transform, parser_kwargs=kwargs, parent_name=parent
        )

    def _get_transform(self, name):
        parser_dict = self.parser_dict.get(name)
        if parser_dict is None:
            raise ValueError(
                f"Parser '{name}' does not exist. "
                "Please check the parser name or add a new one through 'create_node_group'."
            )

        transform = parser_dict["parser"]
        return (
            transform(**parser_dict["parser_kwargs"])
            if isinstance(transform, type)
            else FuncNodeParser(transform)
        )

    def _dynamic_create_nodes(self, parser_name) -> None:
        parser_dict = self.parser_dict.get(parser_name)
        if self.store.has_nodes(parser_name):
            return
        transform = self._get_transform(parser_name)
        parent_name = parser_dict["parent_name"]
        if parent_name:
            self._dynamic_create_nodes(parent_name)

        parent_nodes = self.store.traverse_nodes(parent_name)

        sub_nodes = transform(parent_nodes, parser_name)
        self.store.add_nodes(parser_name, sub_nodes)
        LOG.debug(f"building {parser_name} nodes: {sub_nodes}")

    def retrieve(self, query, parser_name, similarity, index, topk, similarity_kws):
        if isinstance(query, LazyLlmRequest):
            query = query.input

        # lazy load files
        if not self.store.has_nodes("root"):
            docs = self.directory_reader.load_data()
            self.store.add_nodes("root", docs)
            LOG.debug(f"building root nodes: {docs}")

        self._dynamic_create_nodes(parser_name)

        nodes = self.store.traverse_nodes(parser_name)

        similarity_func, use_embedding = self.index.registered_similarity[similarity]
        if use_embedding:
            assert self.embed, "Chosen similarity needs embed model."
            assert len(query) > 0, "query empty!"
            query_embedding = ast.literal_eval(self.embed(query))
            for node in nodes:
                if not node.embedding:
                    node.embedding = ast.literal_eval(self.embed(node.text))
            similar_nodes = similarity_func(
                query_embedding, nodes, topk=topk, **similarity_kws
            )
        else:
            similar_nodes = similarity_func(query, nodes, topk=topk, **similarity_kws)

        return similar_nodes

    def _find_parent(self, nodes: List[DocNode], name: str) -> List[DocNode]:
        def recurse_parents(node: DocNode, visited: Set[DocNode]) -> None:
            if node.parent:
                if node.parent.ntype == name:
                    visited.add(node.parent)
                recurse_parents(node.parent, visited)

        result = set()
        for node in nodes:
            recurse_parents(node, result)
        if not result:
            LOG.warning(
                f"We can not find any nodes for name `{name}`, please check your input"
            )
        LOG.debug(f"Found parent node for {name}: {result}")
        return list(result)

    def find_parent(self, name: str) -> List[DocNode]:
        return partial(self._find_parent, name=name)

    def _find_children(self, nodes: List[DocNode], name: str) -> List[DocNode]:
        def recurse_children(node: DocNode, visited: Set[DocNode]) -> bool:
            if name in node.children:
                visited.update(node.children[name])
                return True

            found_in_any_child = False

            for children_list in node.children.values():
                for child in children_list:
                    if recurse_children(child, visited):
                        found_in_any_child = True
                    else:
                        break

            return found_in_any_child

        result = set()

        if not self.store.has_nodes(name):
            self._dynamic_create_nodes(name)

        for node in nodes:
            if name in node.children:
                result.update(node.children[name])
            else:
                LOG.log_once(
                    f"Fetching children that are not in direct relationship might be slower. "
                    f"We recommend first fetching through direct children {list(node.children.keys())}, "
                    f"then using `find_children()` again for deeper levels.",
                    level="warning",
                )
                # Note: the input nodes are the same type
                if not recurse_children(node, result):
                    LOG.warning(
                        f"Node {node} and its children do not contain any nodes with the name `{name}`. "
                        "Skipping further search in this branch."
                    )
                    break

        if not result:
            LOG.warning(
                f"We cannot find any nodes for name `{name}`, please check your input."
            )

        LOG.debug(f"Found children nodes for {name}: {result}")
        return list(result)

    def find_children(self, name: str) -> List[DocNode]:
        return partial(self._find_children, name=name)


class RetrieverV2(ModuleBase):
    __enable_request__ = False

    def __init__(
        self,
        doc,
        parser: str,
        similarity: str = "dummy_similarity",
        index: str = "default",
        topk: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.doc = doc
        self.parser_name = parser
        self.similarity = similarity  # similarity function str
        self.index = index
        self.topk = topk
        self.similarity_kw = kwargs  # kw parameters

    def forward(self, query):
        # TODO(ywt): self.doc._impl._impl.retrieve should be updated
        # if we've developed all of the components
        return self.doc._impl._impl.retrieve(
            query,
            self.parser_name,
            self.similarity,
            self.index,
            self.topk,
            self.similarity_kw,
        )
