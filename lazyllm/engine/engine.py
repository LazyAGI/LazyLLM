from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Type
import lazyllm
from lazyllm import graph, switch, pipeline
from lazyllm.tools import IntentClassifier
from .node import all_nodes
import re
import ast
import inspect


@dataclass
class Node():
    id: int
    kind: str
    name: str
    args: Optional[Dict] = None
    func: Optional[Callable] = None


class CodeBlock(object):
    def __init__(self, code):
        pass


# Each session will have a separate engine
class Engine(object):
    __default_engine__ = None

    def __init__(self):
        self._nodes = {'__start__': Node(id='__start__', kind='__start__', name='__start__'),
                       '__end__': Node(id='__end__', kind='__end__', name='__end__')}

    def __new__(cls):
        if cls is not Engine:
            return super().__new__(cls)
        return Engine.__default_engine__()

    @classmethod
    def set_default(cls, engine: Type):
        cls.__default_engine__ = engine

    def start(self, nodes=[]):
        raise NotImplementedError

    def update(self, changes=[]):
        raise NotImplementedError

    def build_node(self, node) -> Callable:
        return _constructor.build(node)


class NodeConstructor(object):
    builder_methods = dict()

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
        if node.kind in NodeConstructor.builder_methods:
            createf = NodeConstructor.builder_methods[node.kind]
            r = inspect.getfullargspec(createf)
            if isinstance(node.args, dict) and set(r.args) == set(node.args.keys()):
                node.func = NodeConstructor.builder_methods[node.kind](**node.args)
            else:
                node.func = NodeConstructor.builder_methods[node.kind](node.args)
            return node

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
        node.func = module
        return node


_constructor = NodeConstructor()


@NodeConstructor.register('Graph')
@NodeConstructor.register('SubGraph')
def make_graph(nodes: List[dict], edges: List[dict]):
    engine = Engine()
    nodes = [engine.build_node(node) for node in nodes]

    with graph() as g:
        for node in nodes:
            setattr(g, node.name, node.func)

    for edge in edges:
        if formatter := edge.get('formatter'):
            assert formatter.startswith('[') and formatter.endswith(']')
            formatter = lazyllm.formatter.JsonLike(formatter)
        g.add_edge(engine._nodes[edge['iid']].name, engine._nodes[edge['oid']].name, formatter)

    return g


@NodeConstructor.register('App')
def make_subapp(nodes: List[dict], edges: List[dict]):
    return make_graph(nodes, edges)


@NodeConstructor.register('Code')
def make_code(code):
    fname = re.search(r'def\s+(\w+)\s*\(', code).group(1)
    module = ast.parse(code)
    code = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(code, {}, local_dict)
    return local_dict[fname]


@NodeConstructor.register('Switch')
def make_switch(judge_on_full_input: bool, nodes: Dict[str, List[dict]]):
    with switch(judge_on_full_input=judge_on_full_input) as sw:
        for cond, nodes in nodes.items():
            if isinstance(nodes, list) and len(nodes) > 1:
                f = pipeline([Engine().build_node(node).func for node in nodes])
            else:
                f = Engine().build_node(nodes[0] if isinstance(nodes, list) else nodes).func
            sw.case[cond::f]
    return sw

@NodeConstructor.register('Intention')
def make_intention(base_model: str, nodes: Dict[str, List[dict]]):
    with IntentClassifier(Engine().build_node(base_model)) as ic:
        for cond, nodes in nodes.items():
            if isinstance(nodes, list) and len(nodes) > 1:
                f = pipeline([Engine().build_node(node).func for node in nodes])
            else:
                f = Engine().build_node(nodes[0] if isinstance(nodes, list) else nodes).func
            ic.case[cond::f]
    return ic
