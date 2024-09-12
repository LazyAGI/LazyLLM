from .engine import Engine, Node
from lazyllm import ActionModule, once_wrapper
from typing import List, Dict, Optional
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

    def build_node(self, node):
        if not isinstance(node, Node):
            if isinstance(node, str):
                return self._nodes[node]
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        if node.id not in self._nodes:
            self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def update_node(self, node):
        if not isinstance(node, Node):
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def start(self, nodes: List[Dict] = [], edges: List[Dict] = [], resources: List[Dict] = [],
              gid: Optional[str] = None, name: Optional[str] = None):
        node = Node(id=gid or str(uuid.uuid4().hex), kind='Graph',
                    name=name or str(uuid.uuid4().hex), args=dict(nodes=nodes, edges=edges, resources=resources))
        self.graph = self.build_node(node).func
        self.graph.start()

    def update(self, nodes: List[Dict] = [], changed_nodes: List[Dict] = [],
               edges: List[Dict] = [], changed_resources: List[Dict] = [],
               gid: Optional[str] = None, name: Optional[str] = None):
        for r in changed_resources:
            if r['kind'] in ('server', 'web'):
                raise NotImplementedError('Web and Api server are not allowed now')
            self.update_node(r)
        for n in changed_nodes: self.update_node(n)
        node = Node(id=gid or str(uuid.uuid4().hex), kind='Graph',
                    name=name or str(uuid.uuid4().hex), args=dict(nodes=nodes, edges=edges))
        self.graph = self.update_node(node).func
        ActionModule(self.graph).start()

    def run(self, *args, **kw):
        return self.graph(*args, **kw)
