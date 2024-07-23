from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from lazyllm import graph, ActionModule, switch, pipeline
from .node import all_nodes
import re
import ast
import inspect


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


class CodeBlock(object):
    def __init__(self, code):
        pass


class NodeBuilder(object):
    builder_methods = dict()

    def __init__(self):
        self.funcs = dict()

    @classmethod
    def register(cls, name):
        def impl(f):
            cls.builder_methods[name] = f
            return f
        return impl

    # build node recursively
    def build(self, node):
        if node.kind.startswith('__') and node.kind.endswith('__'):
            return None
        if node.id in self.funcs:
            return self.funcs[node.id]
        if node.kind in NodeBuilder.builder_methods:
            createf = NodeBuilder.builder_methods[node.kind]
            r = inspect.getfullargspec(createf)
            if isinstance(node.args, dict) and set(r.args) == set(node.args.keys()):
                return NodeBuilder.builder_methods[node.kind](**node.args)
            else:
                return NodeBuilder.builder_methods[node.kind](node.args)
        node_msgs = all_nodes[node.kind]
        init_args, build_args, other_args = dict(), dict(), dict()

        for key, value in node.args.items():
            if key in node_msgs['init_arguments']:
                getf = node_msgs['init_arguments'][key].getattr_f
                init_args[key] = getf(value) if getf else value
            elif key in node_msgs['builder_argument']:
                getf = node_msgs['builder_argument'][key].getattr_f
                build_args[key] = getf(value) if getf else value
            elif '.' in key:
                builder_key, key = key.split('.')
                if builder_key not in other_args: other_args[builder_key] = dict()
                getf = node_msgs['other_arguments'][builder_key][key].getattr_f
                other_args[builder_key][key] = getf(value) if getf else value
            else:
                raise KeyError(f'Invalid key `{key}` found')

        module = node_msgs['module'](**init_args)
        for key, value in build_args.items():
            module = getattr(module, key)(value, **other_args.get(key, dict()))
        self.funcs[node.id] = module
        return module


_builder = NodeBuilder()


# Each session will have a separate engine
class Engine(object):
    def __init__(self):
        self._nodes = {'__start__': Node(id='__start__', kind='__start__', name='__start__'),
                       '__end__': Node(id='__end__', kind='__end__', name='__end__')}

    def start(self, nodes=[]):
        raise NotImplementedError

    def update(self, changes=[]):
        raise NotImplementedError


class LightEngine(Engine):

    _instance = None

    def __new__(cls):
        if not LightEngine._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__()

    def make_graph(self, nodes: List[dict], edges: List[dict]):
        nodes = {n['id']: Node(id=n['id'], kind=n['kind'], name=n['name'], args=n['args']) for n in nodes}
        edges = [Edge(iid=e['iid'], oid=e['oid'], formatter=e.get('formatter')) for e in edges]
        for node in nodes.values():
            node.func = _builder.build(node)

        with graph() as g:
            for _, node in nodes.items():
                setattr(g, node.name, node.func)

        self._nodes.update(nodes)
        for edge in edges:
            g.add_edge(self._nodes[edge.iid].name, self._nodes[edge.oid].name)
        return g

    def start(self, nodes: List[dict] = [], edges: List[dict] = []):
        self.graph = self.make_graph(nodes, edges)
        ActionModule(self.graph).start()

    def run(self, *args, **kw):
        return self.graph(*args, **kw)


@NodeBuilder.register('SubGroup')
def make_subgraph(nodes: List[dict], edges: List[dict]):
    return LightEngine().make_graph(nodes, edges)


@NodeBuilder.register('App')
def make_subapp(nodes: List[dict], edges: List[dict]):
    return LightEngine().make_graph(nodes, edges)


@NodeBuilder.register('Code')
def make_code(code):
    fname = re.search(r'def\s+(\w+)\s*\(', code).group(1)
    module = ast.parse(code)
    code = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(code, {}, local_dict)
    return local_dict[fname]


@NodeBuilder.register('Switch')
def make_switch(judge_on_full_input: bool, nodes: Dict[str, List[dict]]):
    with switch(judge_on_full_input=judge_on_full_input) as sw:
        for cond, nodes in nodes.items():
            if len(nodes) > 1:
                node = pipeline([LightEngine()._build(node) for node in nodes])
            else:
                node = LightEngine()._build(nodes[cond])
            sw.case[cond, node]
    return sw
