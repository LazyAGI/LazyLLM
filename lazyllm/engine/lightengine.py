from .engine import Engine, Node
from lazyllm import once_wrapper
from typing import List, Dict, Optional, overload, Set
import uuid


class LightEngine(Engine):

    _instance = None

    def __new__(cls):
        if not LightEngine._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @once_wrapper
    def __init__(self):
        super().__init__()
        self.node_graph: Set[str, List[str]] = dict()

    def build_node(self, node):
        if not isinstance(node, Node):
            if isinstance(node, str):
                return self._nodes.get(node)
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        if node.id not in self._nodes:
            self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def update_node(self, node):
        if not isinstance(node, Node):
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    @overload
    def start(self, nodes: str) -> None:
        ...

    @overload
    def start(self, nodes: List[Dict] = [], edges: List[Dict] = [], resources: List[Dict] = [],
              gid: Optional[str] = None, name: Optional[str] = None) -> str:
        ...

    def start(self, nodes=[], edges=[], resources=[], gid=None, name=None):
        if isinstance(nodes, str):
            self.build_node(nodes).func.start()
        else:
            gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
            node = Node(id=gid, kind='Graph', name=name, args=dict(nodes=nodes, edges=edges, resources=resources))
            self.build_node(node).func.start()
            return gid

    def status(self, node_id: str, task_name: Optional[str] = None):
        node = self.build_node(node_id)
        if not node: return 'unknown'
        elif node.id in self.node_graph:
            return {n.id: self.status(n.id) for n in self.node_graph[node.id]}
        elif node.kind in ('LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA'):
            return node.func.status(task_name=task_name)
        else:
            return 'started'

    def stop(self, node_id: str, task_name: Optional[str] = None):
        node = self.build_node(node_id)
        if task_name:
            assert node.kind in ('LocalLLM')
            node.func.stop(task_name=task_name)
        else:
            assert node.kind in ('Graph', 'LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA')
            node.func.stop()

    def update(self, nodes: List[Dict] = [], changed_nodes: List[Dict] = [],
               edges: List[Dict] = [], changed_resources: List[Dict] = [],
               gid: Optional[str] = None, name: Optional[str] = None):
        for r in changed_resources:
            if r['kind'] in ('server', 'web'):
                raise NotImplementedError('Web and Api server are not allowed now')
            self.update_node(r)
        for n in changed_nodes: self.update_node(n)
        gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
        node = Node(id=gid, kind='Graph', name=name, args=dict(nodes=nodes, edges=edges))
        self.update_node(node).func.start()

    def run(self, id: str, *args, **kw):
        return self.build_node(id).func(*args, **kw)
