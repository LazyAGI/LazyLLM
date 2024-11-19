import copy
import uuid
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import List, Dict, Optional, Set, Union

import lazyllm
from lazyllm import once_wrapper
from .engine import Engine, Node
from lazyllm.tools.train_service.serve import TrainServer
from lazyllm.tools.train_service.client import LocalTrainClient, OnlineTrainClient


@contextmanager
def set_resources(resource):
    lazyllm.globals['engine_resource'] = {r['id']: r for r in resource}
    try:
        yield
    finally:
        lazyllm.globals.pop('engine_resource', None)


class LightEngine(Engine):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not LightEngine._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @once_wrapper
    def __init__(self, launch_local_train_serve=False):
        super().__init__()
        self.node_graph: Set[str, List[str]] = dict()
        self._launch_local_train = launch_local_train_serve
        if self._launch_local_train:
            train_server = TrainServer()
            self._local_serve = lazyllm.ServerModule(train_server, launcher=lazyllm.launcher.EmptyLauncher(sync=False))
            self._local_serve.start()()
            parsed_url = urlparse(self._local_serve._url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            self.local_train_client = LocalTrainClient(base_url)
        self.online_train_client = OnlineTrainClient()

    # Local
    def local_train(self, train_config, token='default'):
        if not self._launch_local_train:
            raise RuntimeError('Please set launch_local_train_serve=True when instanche a LightEngine.')
        return self.local_train_client.train(train_config, token)

    def local_cancel_finetuning(self, token, job_id):
        if not self._launch_local_train:
            raise RuntimeError('Please set launch_local_train_serve=True when instanche a LightEngine.')
        return self.local_train_client.cancel_finetuning(token, job_id)

    def local_get_train_status(self, token, job_id):
        if not self._launch_local_train:
            raise RuntimeError('Please set launch_local_train_serve=True when instanche a LightEngine.')
        return self.local_train_client.get_train_status(token, job_id)

    def local_get_log(self, token, job_id):
        if not self._launch_local_train:
            raise RuntimeError('Please set launch_local_train_serve=True when instanche a LightEngine.')
        return self.local_train_client.get_log(token, job_id)

    def local_get_all_finetuned_models(self, token):
        if not self._launch_local_train:
            raise RuntimeError('Please set launch_local_train_serve=True when instanche a LightEngine.')
        return self.local_train_client.get_all_finetuned_models(token)

    # Online
    def online_train(self, train_config, token, source='glm'):
        return self.online_train_client.train(train_config, token, source)

    def online_cancel_finetuning(self, token, job_id, source='glm'):
        return self.online_train_client.cancel_finetuning(token, job_id, source)

    def online_get_train_status(self, token, job_id, source='glm'):
        return self.online_train_client.get_train_status(token, job_id, source)

    def online_get_log(self, token, job_id, source='glm', target_path=None):
        return self.online_train_client.get_log(token, job_id, source=source, target_path=target_path)

    def online_get_all_finetuned_models(self, token, source='glm'):
        return self.online_train_client.get_all_finetuned_models(token, source)

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

    def start(self, nodes, edges=[], resources=[], gid=None, name=None):
        if isinstance(nodes, str):
            assert not edges and not resources and not gid and not name
            self.build_node(nodes).func.start()
        elif isinstance(nodes, dict):
            Engine().build_node(nodes)
        else:
            gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
            node = Node(id=gid, kind='Graph', name=name, args=dict(
                nodes=copy.copy(nodes), edges=copy.copy(edges), resources=copy.copy(resources)))
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
            _file_resources: Optional[Dict[str, Union[str, List[str]]]] = None, **kw):
        if files := _lazyllm_files:
            assert len(args) <= 1 and len(kw) == 0, 'At most one query is enabled when file exists'
            args = [lazyllm.formatter.file(formatter='encode')(dict(query=args[0] if args else '', files=files))]
        if _file_resources:
            lazyllm.globals['lazyllm_files'] = _file_resources
        result = self.build_node(id).func(*args, **kw)
        lazyllm.globals['lazyllm_files'] = {}
        return result
