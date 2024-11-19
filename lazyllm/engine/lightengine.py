from .engine import Engine, Node, ServerGraph
import lazyllm
from lazyllm import once_wrapper
from typing import List, Dict, Optional, Set, Union
import copy
import uuid
from contextlib import contextmanager


@contextmanager
def set_resources(resource):
    lazyllm.globals['engine_resource'] = {r['id']: r for r in resource}
    try:
        yield
    finally:
        lazyllm.globals.pop('engine_resource', None)


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
                if node not in self._nodes and (resource := lazyllm.globals.get('engine_resource', {}).get(node)):
                    node = resource
                else:
                    return self._nodes.get(node)
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        if node.id not in self._nodes:
            self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def release_node(self, *node_ids: Union[str, List[str]]):
        if len(node_ids) == 1 and isinstance(node_ids[0], (tuple, list)): node_ids = node_ids[0]
        for nodeid in node_ids:
            self.stop(nodeid)
            # TODO(wangzhihong): Analyze dependencies and only allow deleting nodes without dependencies
            [self._nodes.pop(id) for id in self.subnodes(nodeid, recursive=True)]
            self._nodes.pop(nodeid)

    def update_node(self, node):
        if not isinstance(node, Node):
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def start(self, nodes, edges=[], resources=[], gid=None, name=None, _history_ids=None):
        if isinstance(nodes, str):
            assert not edges and not resources and not gid and not name
            self.build_node(nodes).func.start()
        elif isinstance(nodes, dict):
            Engine().build_node(nodes)
        else:
            gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
            node = Node(id=gid, kind='Graph', name=name, args=dict(
                nodes=copy.copy(nodes), edges=copy.copy(edges),
                resources=copy.copy(resources), _history_ids=_history_ids))
            with set_resources(resources):
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

    def update(self, gid_or_nodes: Union[str, Dict, List[Dict]], nodes: List[Dict],
               edges: List[Dict] = [], resources: List[Dict] = []) -> str:
        if isinstance(gid_or_nodes, str):
            assert (gid := gid_or_nodes) in self._nodes
            name = self._nodes[gid].name
            self.release_node(gid)
            self.start(nodes, edges, resources, gid_or_nodes, name=name)
        else:
            for node in gid_or_nodes: self.update_node(node)

    def run(self, id: str, *args, _lazyllm_files: Optional[Union[str, List[str]]] = None,
            _file_resources: Optional[Dict[str, Union[str, List[str]]]] = None,
            _lazyllm_history: Optional[List[List[str]]] = None, **kw):
        if files := _lazyllm_files:
            assert len(args) <= 1 and len(kw) == 0, 'At most one query is enabled when file exists'
            args = [lazyllm.formatter.file(formatter='encode')(dict(query=args[0] if args else '', files=files))]
        if _file_resources:
            lazyllm.globals['lazyllm_files'] = _file_resources
        f = self.build_node(id).func
        lazyllm.FileSystemQueue().dequeue()
        if history := _lazyllm_history:
            assert isinstance(f, ServerGraph) and (ids := f._history_ids), 'Only graph can support history'
            if not isinstance(history, list) and all([isinstance(h, list) for h in history]):
                raise RuntimeError('History shoule be [[str, str], ..., [str, str]] (list of list of str)')
            lazyllm.globals['chat_history'] = {Engine().build_node(i).func._module_id: history for i in ids}
        result = self.build_node(id).func(*args, **kw)
        lazyllm.globals['lazyllm_files'] = {}
        lazyllm.globals['chat_history'] = {}
        return result
