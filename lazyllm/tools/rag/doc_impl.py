from functools import partial
from typing import Dict, List, Optional, Set
from lazyllm import ModuleBase, LOG
from lazyllm.common import LazyLlmRequest
from .transform import FuncNodeTransform, SentenceSplitter
from .store import MapStore, DocNode
from .data_loaders import DirectoryReader
from .index import DefaultIndex


class DocImplV2:
    def __init__(self, embed, doc_files=Optional[List[str]], **kwargs):
        super().__init__()
        self.directory_reader = DirectoryReader(input_files=doc_files)
        self.node_groups: Dict[str, Dict] = {}
        self.create_node_group_default()
        self.store = MapStore()
        self.index = DefaultIndex(embed)

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

    def create_node_group(
        self, name, transform, parent="_lazyllm_root", **kwargs
    ) -> None:
        if name in self.node_groups:
            LOG.warning(f"Duplicate group name: {name}")
        assert callable(transform), "transform should be callable"
        self.node_groups[name] = dict(
            transform=transform, transform_kwargs=kwargs, parent_name=parent
        )

    def _get_transform(self, name):
        node_group = self.node_groups.get(name)
        if node_group is None:
            raise ValueError(
                f"Node group '{name}' does not exist. "
                "Please check the group name or add a new one through `create_node_group`."
            )

        transform = node_group["transform"]
        return (
            transform(**node_group["transform_kwargs"])
            if isinstance(transform, type)
            else FuncNodeTransform(transform)
        )

    def _dynamic_create_nodes(self, group_name) -> None:
        node_group = self.node_groups.get(group_name)
        if self.store.has_nodes(group_name):
            return
        transform = self._get_transform(group_name)
        parent_name = node_group["parent_name"]
        self._dynamic_create_nodes(parent_name)

        parent_nodes = self.store.traverse_nodes(parent_name)

        sub_nodes = transform(parent_nodes, group_name)
        self.store.add_nodes(group_name, sub_nodes)
        LOG.debug(f"building {group_name} nodes: {sub_nodes}")

    def _get_nodes(self, group_name: str) -> List[DocNode]:
        # lazy load files, if group isn't set, create the group
        if not self.store.has_nodes("_lazyllm_root"):
            docs = self.directory_reader.load_data()
            self.store.add_nodes("_lazyllm_root", docs)
            LOG.debug(f"building _lazyllm_root nodes: {docs}")
        self._dynamic_create_nodes(group_name)
        return self.store.traverse_nodes(group_name)

    def retrieve(self, query, group_name, similarity, index, topk, similarity_kws):
        if index:
            assert index == "default", "we only support default index currently"
        if isinstance(query, LazyLlmRequest):
            query = query.input

        nodes = self._get_nodes(group_name)
        return self.index.query(query, nodes, similarity, topk, **similarity_kws)

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

        # case when user hasn't used the group before.
        _ = self._get_nodes(name)

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
        group_name: str,
        similarity: str = "dummy",
        index: str = "default",
        topk: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.doc = doc
        self.group_name = group_name
        self.similarity = similarity  # similarity function str
        self.index = index
        self.topk = topk
        self.similarity_kw = kwargs  # kw parameters

    def forward(self, query):
        # TODO(ywt): self.doc._impl._impl.retrieve should be updated
        # if we've developed all of the components
        return self.doc._impl._impl.retrieve(
            query,
            self.group_name,
            self.similarity,
            self.index,
            self.topk,
            self.similarity_kw,
        )
