from .engine import Engine, Node
from lazyllm import once_wrapper
from typing import List, Dict, Optional, Set, Any
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

    def release_node(self, nodeid: str):
        self.stop(nodeid)
        # TODO(wangzhihong): Analyze dependencies and only allow deleting nodes without dependencies
        self._nodes.pop(nodeid)

    def update_node(self, node):
        if not isinstance(node, Node):
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def start(self, nodes, edges=[], resources=[], gid=None, name=None):
        if isinstance(nodes, str):
            assert not edges and not resources and not gid and not name
            self.build_node(nodes).func.start()
        elif isinstance(nodes, dict):
            Engine().build_node(nodes)
        else:
            gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
            node = Node(id=gid, kind='Graph', name=name, args=dict(nodes=nodes, edges=edges, resources=resources))
            self.build_node(node).func.start()
            return gid

    def status(self, node_id: str, task_name: Optional[str] = None):
        node = self.build_node(node_id)
        if not node:
            return 'unknown'
        elif task_name:
            assert node.kind in ('LocalLLM')
            return node.func.status(task_name=task_name)
        elif subs := node.subitems:
            return {n: self.status(n) for n in subs}
        elif node.kind in ('LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA', 'web', 'server'):
            return node.func.status()
        else:
            return 'running'

    def stop(self, node_id: Optional[str] = None, task_name: Optional[str] = None):
        if not node_id:
            for node in self._nodes:
                self.release_node(node)
        elif node := self.build_node(node_id):
            if task_name:
                assert node.kind in ('LocalLLM')
                node.func.stop(task_name=task_name)
            elif node.kind in ('Graph', 'LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA'):
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
