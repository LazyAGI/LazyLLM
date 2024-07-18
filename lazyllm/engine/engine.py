from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from lazyllm import graph
from .node import all_nodes


@dataclass
class Node(object):
    id: int
    kind: str
    name: str
    args: Optional[Dict] = None
    func: Optional[Callable] = None

@dataclass
class Edge(object):
    iid: int
    oid: int
    formatter: Optional[str] = None


class NodeBuilder(object):
    builder_methods = dict()

    @classmethod
    def register(cls, name):
        def impl(f):
            cls.builder_methods[name] = f
        return impl

    def build(self, node):
        if node.kind in NodeBuilder.builder_methods:
            return NodeBuilder.builder_methods[node.kind](**node.args)

        builder_args = all_nodes[node.kind]
        module = NodeBuilder.get_module(builder_args.module)
        init_args, build_args, other_args = dict(), dict(), dict()
        if node.kind.startswith('__') and node.kind.endswith('__'):
            return None
        for key, value in node.args.items():
            if key in builder_args.init_arguments:
                getf = builder_args.init_arguments[key].getattr_f
                init_args[key] = getf(value) if getf else value
            elif key in build_args.builder_argument:
                getf = builder_args.builder_argument[key].getattr_f
                build_args[key] = getf(value) if getf else value
            elif '.' in key:
                builder_key, key = key.split('.')
                if builder_key not in other_args: other_args[builder_key] = dict()
                getf = builder_args.other_arguments[builder_key][key].getattr_f
                other_args[builder_key][key] = value
            else:
                raise KeyError(f'Invalid key `{key}` found')

        module = module(**init_args)
        for key, value in build_args.items():
            module = getattr(module, key)(value, **other_args.get(key, dict()))
        return module


_builder = NodeBuilder()

# Each session will have a separate engine
class Engine(object):
    def __init__(self):
        self.nodes = dict()

    def start(self, nodes=[]):
        raise NotImplementedError

    def update(self, changes=[]):
        raise NotImplementedError

    @staticmethod
    def make_graph(self, nodes, edges):
        with graph() as g:
            for _, node in nodes.items():
                setattr(g, node.name)(node.func)
        for edge in edges:
            g.add_edge(nodes[edge.iid].name, nodes[edge.oid].name)
        return g


class LightEngine(Engine):
    def start(self, nodes: List[dict] = [], edges: List[dict] = []):
        nodes = [Node(id=n['id'], kind=n['kind'], name=n['name'], args=n['args']) for n in nodes]
        nodes.extend([Node(id='__start__', kind='__start__', name='__start__'),
                      Node(id='__end__', kind='__end__', name='__end__')])
        edges = [Edge(iid=e['iid'], oid=e['oid'], formatter=e.get('formatter')) for e in edges]
        for node in nodes:
            self.nodes[node.id].func = _builder.build(node)
        self.graph = self.make_graph(self.nodes, edges)

    def run(self, *args, **kw):
        return self.graph(*args, **kw)
