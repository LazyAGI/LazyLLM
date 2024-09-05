from typing import List, Callable, Dict, Type, Optional, Union
import lazyllm
from lazyllm import graph, switch, pipeline
from lazyllm.tools import IntentClassifier
from .node import all_nodes, Node
import re
import ast
import inspect


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

    def start(self, nodes: List[Dict], edges: List[Dict], resources: List[Dict],
              gid: Optional[str], name: Optional[str]):
        raise NotImplementedError

    def update(self, nodes: List[Dict], changed_nodes: List[Dict], edges: List[Dict],
               changed_resources: List[Dict], gid: Optional[str], name: Optional[str]):
        raise NotImplementedError

    def build_node(self, node) -> Callable:
        return _constructor.build(node)

    def reset(self):
        self.__init__.flag.reset()
        self.__init__()


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
            if isinstance(node.args, dict) and set(node.args.keys()).issubset(set(r.args)):
                node.func = NodeConstructor.builder_methods[node.kind](**node.args)
            else:
                node.func = NodeConstructor.builder_methods[node.kind](node.args)
            return node

        node_msgs = all_nodes[node.kind]
        init_args, build_args, other_args = dict(), dict(), dict()

        def get_args(cls, key, value, builder_key=None):
            node_args = node_msgs[cls][builder_key][key] if builder_key else node_msgs[cls][key]
            if node_args.type == Node:
                return Engine().build_node(value).func
            return node_args.getattr_f(value) if node_args.getattr_f else value

        for key, value in node.args.items():
            if key in node_msgs['init_arguments']:
                init_args[key] = get_args('init_arguments', key, value)
            elif key in node_msgs['builder_argument']:
                build_args[key] = get_args('builder_argument', key, value)
            elif '.' in key:
                builder_key, key = key.split('.')
                if builder_key not in other_args: other_args[builder_key] = dict()
                other_args[builder_key][key] = get_args('other_arguments', key, value, builder_key=builder_key)
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
def make_graph(nodes: List[dict], edges: List[dict], resources: List[dict] = []):
    engine = Engine()
    resources = [engine.build_node(resource) for resource in resources]
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


# Note: It will be very dangerous if provided to C-end users as a SAAS service
@NodeConstructor.register('Code')
def make_code(code):
    fname = re.search(r'def\s+(\w+)\s*\(', code).group(1)
    module = ast.parse(code)
    code = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(code, {}, local_dict)
    return local_dict[fname]


def _build_pipeline(nodes):
    if isinstance(nodes, list) and len(nodes) > 1:
        return pipeline([Engine().build_node(node).func for node in nodes])
    else:
        return Engine().build_node(nodes[0] if isinstance(nodes, list) else nodes).func


@NodeConstructor.register('Switch')
def make_switch(judge_on_full_input: bool, nodes: Dict[str, List[dict]]):
    with switch(judge_on_full_input=judge_on_full_input) as sw:
        for cond, nodes in nodes.items():
            sw.case[cond::_build_pipeline(nodes)]
    return sw


@NodeConstructor.register('Warp')
def make_warp(nodes: List[dict], edges: List[dict], resources: List[dict] = []):
    return lazyllm.warp(make_graph(nodes, edges, resources))


@NodeConstructor.register('Loop')
def make_loop(stop_condition: str, judge_on_full_input: bool, nodes: List[dict],
              edges: List[dict], resources: List[dict] = []):
    stop_condition = make_code(stop_condition)
    return lazyllm.loop(make_graph(nodes, edges, resources), stop_condition=stop_condition,
                        judge_on_full_input=judge_on_full_input)


@NodeConstructor.register('Ifs')
def make_ifs(cond: str, judge_on_full_input: bool, true: List[dict], false: List[dict]):
    return lazyllm.ifs(make_code(cond), tpath=_build_pipeline(true), fpath=_build_pipeline(false))


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


@NodeConstructor.register('Document')
def make_document(dataset_path: str, embed: Node = None, create_ui: bool = False, node_group: List = []):
    document = lazyllm.tools.rag.Document(dataset_path, Engine().build_node(embed) if embed else None, create_ui)
    for group in node_group:
        if group['transform'] == 'LLMParser': group['llm'] = Engine().build_node(group['llm']).func
        elif group['transform'] == 'FuncNode': group['function'] = make_code(group['function'])
        document.create_node_group(**group)
    return document

@NodeConstructor.register('Reranker')
def make_reranker(type: str = 'ModuleReranker', target: Optional[str] = None,
                  output_format: Optional[str] = None, join: Union[bool, str] = False, arguments: Dict = {}):
    return lazyllm.tools.Reranker(type, target=target, output_format=output_format, join=join, **arguments)

@NodeConstructor.register('JoinFormatter')
def make_join_formatter(method='sum'):
    def impl(*args):
        assert len(args) > 0, 'Cannot sum empty inputs'
        return sum(args, type(args[0])())
    return impl
