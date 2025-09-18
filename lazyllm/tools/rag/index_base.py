from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List, Optional

class IndexBase(ABC):
    """An abstract base class for implementing indexing systems that support updating, removing, and querying document nodes.
`class IndexBase(ABC)`
This abstract base class defines the interface for an indexing system. It requires subclasses to implement methods for updating, removing, and querying document nodes.


Examples:
    >>> from mymodule import IndexBase, DocNode
    >>> class MyIndex(IndexBase):
    ...     def __init__(self):
    ...         self.nodes = []
    ...     def update(self, nodes):
    ...         self.nodes.extend(nodes)
    ...         print(f"Updated nodes: {nodes}")
    ...     def remove(self, uids, group_name=None):
    ...         self.nodes = [node for node in self.nodes if node.uid not in uids]
    ...         print(f"Removed nodes with uids: {uids}")
    ...     def query(self, *args, **kwargs):
    ...         print("Querying nodes...")
    ...         return self.nodes
    >>> index = MyIndex()
    >>> doc1 = DocNode(uid="1", content="Document 1")
    >>> doc2 = DocNode(uid="2", content="Document 2")
    >>> index.update([doc1, doc2])
    Updated nodes: [DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
    >>> index.query()
    Querying nodes...
    [DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
    >>> index.remove(["1"])
    Removed nodes with uids: ['1']
    >>> index.query()
    Querying nodes...
    [DocNode(uid="2", content="Document 2")]
    """
    # TODO(chenjiahao): change params `nodes` to `segments`, index should be able to handle segments
    @abstractmethod
    def update(self, nodes: List[DocNode]) -> None:
        """Update index contents.

This method receives a list of document nodes and updates or inserts them into the index structure. Typically used for incremental indexing or refreshing data.

Args:
    nodes (List[DocNode]): A list of document nodes to update or insert.
"""
        pass

    @abstractmethod
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        """Remove specific document nodes from the index.

Removes document nodes based on their unique identifiers, optionally scoped by group name.

Args:
    uids (List[str]): List of unique IDs corresponding to the document nodes to remove.
    group_name (Optional[str]): Optional group name to scope the removal operation.
"""
        pass

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        """Execute a query over the index.

Performs a query based on the given arguments and returns matching document nodes.  
**Note:** This method is a placeholder and should be implemented by subclasses.

Args:
    *args: Positional arguments for the query.
    **kwargs: Keyword arguments for the query.
"""
        pass
