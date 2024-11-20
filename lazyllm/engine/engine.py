from typing import List, Dict, Type, Optional, Union, Any, overload
import lazyllm
from lazyllm import graph, switch, pipeline, package
from lazyllm.tools import IntentClassifier
from lazyllm.common import compile_func
from .node import all_nodes, Node
from .node_meta_hook import NodeMetaHook
import inspect
import functools


# Each session will have a separate engine
class Engine(object):
    __default_engine__ = None
    REPORT_URL = ""

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

    @overload
    def start(self, nodes: str) -> None:
        ...

    @overload
    def start(self, nodes: Dict[str, Any]) -> None:
        ...

    @overload
    def start(self, nodes: List[Dict] = [], edges: List[Dict] = [], resources: List[Dict] = [],
              gid: Optional[str] = None, name: Optional[str] = None, _history_ids: Optional[List[str]] = None) -> str:
        ...

    @overload
    def update(self, nodes: List[Dict]) -> None:
        ...

    @overload
    def update(self, gid: str, nodes: List[Dict], edges: List[Dict] = [],
               resources: List[Dict] = []) -> str:
        ...

    def release_node(self, nodeid: str): pass
    def stop(self, node_id: Optional[str] = None, task_name: Optional[str] = None): pass

    def build_node(self, node) -> Node:
        return _constructor.build(node)

    def set_report_url(self, url) -> None:
        Engine.REPORT_URL = url

    def reset(self):
        for node in self._nodes:
            self.stop(node)
        self.__init__.flag.reset()
        self.__init__()

    def __del__(self):
        self.stop()
        self.reset()

    def subnodes(self, nodeid: str, recursive: bool = False):
        def _impl(nid, recursive):
            for id in self._nodes[nid].subitems:
                yield id
                if recursive: yield from self.subnodes(id, True)
        return list(_impl(nodeid, recursive))


class NodeConstructor(object):
    builder_methods = dict()

    @classmethod
    def register(cls, *names: Union[List[str], str], subitems: Optional[Union[str, List[str]]] = None):
        if len(names) == 1 and isinstance(names[0], (tuple, list)): names = names[0]

        def impl(f):
            for name in names:
                cls.builder_methods[name] = (f, subitems)
            return f
        return impl

    # build node recursively
    def build(self, node: Node):
        if node.kind.startswith('__') and node.kind.endswith('__'):
            return None
        node.arg_names = node.args.pop('_lazyllm_arg_names', None) if isinstance(node.args, dict) else None
        node.enable_data_reflow = (node.args.pop('_lazyllm_enable_report', False)
                                   if isinstance(node.args, dict) else False)
        if node.kind in NodeConstructor.builder_methods:
            createf, node.subitem_name = NodeConstructor.builder_methods[node.kind]
            node.func = createf(**node.args) if isinstance(node.args, dict) and set(node.args.keys()).issubset(
                set(inspect.getfullargspec(createf).args)) else createf(node.args)
            self._process_hook(node, node.func)
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
        self._process_hook(node, module)
        return node

    def _process_hook(self, node, module):
        if not node.enable_data_reflow:
            return
        if isinstance(module, (lazyllm.ModuleBase, lazyllm.LazyLLMFlowsBase)):
            node.func.register_hook(NodeMetaHook(node.func, Engine.REPORT_URL, node.id))


_constructor = NodeConstructor()


class ServerGraph(lazyllm.ModuleBase):
    def __init__(self, g: lazyllm.graph, server: Node, web: Node, _history_ids: Optional[List[str]] = None):
        super().__init__()
        self._g = lazyllm.ActionModule(g)
        self._history_ids = _history_ids
        if server:
            if server.args.get('port'): raise NotImplementedError('Port is not supported now')
            self._g = lazyllm.ServerModule(g)
        if web:
            port = self._get_port(web.args['port'])
            self._web = lazyllm.WebModule(g, port=port, title=web.args['title'], audio=web.args['audio'],
                                          history=[Engine().build_node(h).func for h in web.args.get('history', [])])

    def forward(self, *args, **kw):
        return self._g(*args, **kw)

    # TODO(wangzhihong)
    def _update(self, *, mode=None, recursive=True):
        super(__class__, self)._update(mode=mode, recursive=recursive)
        if hasattr(self, '_web'): self._web.start()
        return self

    def _get_port(self, port):
        if not port: return None
        elif ',' in port:
            return list(int(p.strip()) for p in port.split(','))
        elif '-' in port:
            left, right = tuple(int(p.strip()) for p in port.split('-'))
            assert left < right
            return range(left, right)
        return int(port)

    @property
    def api_url(self):
        if isinstance(self._g, lazyllm.ServerModule):
            return self._g._url
        return None

    @property
    def web_url(self):
        if hasattr(self, '_web'):
            return self._web.url
        return None

    def __repr__(self):
        return repr(self._g)


class ServerResource(object):
    def __init__(self, graph: ServerGraph, kind: str, args: Dict):
        self._graph = graph
        self._kind = kind
        self._args = args

    def status(self):
        return self._graph._g.status if self._kind == 'server' else self._graph._web.status


@NodeConstructor.register('web', 'server')
def make_server_resource(kind: str, graph: ServerGraph, args: Dict[str, Any]):
    return ServerResource(graph, kind, args)


@NodeConstructor.register('Graph', 'SubGraph', subitems=['nodes', 'resources'])
def make_graph(nodes: List[dict], edges: List[Union[List[str], dict]] = [],
               resources: List[dict] = [], enable_server: bool = True, _history_ids: Optional[List[str]] = None):
    engine = Engine()
    server_resources = dict(server=None, web=None)
    for resource in resources:
        if resource['kind'] in server_resources:
            assert enable_server, 'Web and Api server are not allowed outside graph and subgraph'
            assert server_resources[resource['kind']] is None, f'Duplicated {resource["kind"]} resource'
            server_resources[resource['kind']] = Node(id=resource['id'], kind=resource['kind'],
                                                      name=resource['name'], args=resource['args'])

    resources = [engine.build_node(resource) for resource in resources if resource['kind'] not in server_resources]
    nodes: List[Node] = [engine.build_node(node) for node in nodes]

    with graph() as g:
        for node in nodes:
            setattr(g, node.name, node.func)
    g.set_node_arg_name([node.arg_names for node in nodes])

    if not edges:
        edges = ([dict(iid='__start__', oid=nodes[0].id)] + [
            dict(iid=nodes[i].id, oid=nodes[i + 1].id) for i in range(len(nodes) - 1)] + [
            dict(iid=nodes[-1].id, oid='__end__')])

    for edge in edges:
        if isinstance(edge, (tuple, list)): edge = dict(iid=edge[0], oid=edge[1])
        if formatter := edge.get('formatter'):
            assert formatter.startswith(('*[', '[', '}')) and formatter.endswith((']', '}'))
            formatter = lazyllm.formatter.JsonLike(formatter)
        if 'constant' in edge:
            g.add_const_edge(edge['constant'], engine._nodes[edge['oid']].name)
        else:
            g.add_edge(engine._nodes[edge['iid']].name, engine._nodes[edge['oid']].name, formatter)

    sg = ServerGraph(g, server_resources['server'], server_resources['web'], _history_ids=_history_ids)
    for kind, node in server_resources.items():
        if node:
            node.args = dict(kind=kind, graph=sg, args=node.args)
            engine.build_node(node)
    return sg


@NodeConstructor.register('App')
def make_subapp(nodes: List[dict], edges: List[dict], resources: List[dict] = []):
    return make_graph(nodes, edges, resources)


# Note: It will be very dangerous if provided to C-end users as a SAAS service
@NodeConstructor.register('Code')
def make_code(code: str, vars_for_code: Optional[Dict[str, Any]] = None):
    ori_func = compile_func(code, vars_for_code)

    def cls_method(self, *args, **kwargs):
        return ori_func(*args, **kwargs)

    CodeBlock = type("CodeBlock", (lazyllm.ModuleBase,), {"forward": cls_method})
    code_block = CodeBlock()
    code_block.__doc__ = ori_func.__doc__
    code_block.__name__ = ori_func.__name__
    code_block._ori_func = ori_func
    return code_block


def _build_pipeline(nodes):
    if isinstance(nodes, list) and len(nodes) > 1:
        return pipeline([Engine().build_node(node).func for node in nodes])
    else:
        return Engine().build_node(nodes[0] if isinstance(nodes, list) else nodes).func


@NodeConstructor.register('Switch', subitems=['nodes:dict'])
def make_switch(judge_on_full_input: bool, nodes: Dict[str, List[dict]]):
    with switch(judge_on_full_input=judge_on_full_input) as sw:
        for cond, nodes in nodes.items():
            sw.case[cond::_build_pipeline(nodes)]
    return sw


@NodeConstructor.register('Warp', subitems=['nodes', 'resources'])
def make_warp(nodes: List[dict], edges: List[dict] = [], resources: List[dict] = []):
    return lazyllm.warp(make_graph(nodes, edges, resources, enable_server=False))


@NodeConstructor.register('Loop', subitems=['nodes', 'resources'])
def make_loop(stop_condition: str, nodes: List[dict], edges: List[dict] = [],
              resources: List[dict] = [], judge_on_full_input: bool = True):
    stop_condition = make_code(stop_condition)
    return lazyllm.loop(make_graph(nodes, edges, resources, enable_server=False),
                        stop_condition=stop_condition, judge_on_full_input=judge_on_full_input)


@NodeConstructor.register('Ifs', subitems=['true', 'false'])
def make_ifs(cond: str, true: List[dict], false: List[dict], judge_on_full_input: bool = True):
    assert judge_on_full_input, 'judge_on_full_input only support True now'
    return lazyllm.ifs(make_code(cond), tpath=_build_pipeline(true), fpath=_build_pipeline(false))


@NodeConstructor.register('Intention', subitems=['nodes:dict'])
def make_intention(base_model: str, nodes: Dict[str, List[dict]],
                   prompt: str = '', constrain: str = '', attention: str = ''):
    with IntentClassifier(Engine().build_node(base_model).func,
                          prompt=prompt, constrain=constrain, attention=attention) as ic:
        for cond, nodes in nodes.items():
            if isinstance(nodes, list) and len(nodes) > 1:
                f = pipeline([Engine().build_node(node).func for node in nodes])
            else:
                f = Engine().build_node(nodes[0] if isinstance(nodes, list) else nodes).func
            ic.case[cond::f]
    return ic


@NodeConstructor.register('Document')
def make_document(dataset_path: str, embed: Node = None, create_ui: bool = False,
                  server: bool = False, node_group: List = []):
    document = lazyllm.tools.rag.Document(
        dataset_path, Engine().build_node(embed).func if embed else None, server=server, manager=create_ui)
    for group in node_group:
        if group['transform'] == 'LLMParser': group['llm'] = Engine().build_node(group['llm']).func
        elif group['transform'] == 'FuncNode': group['function'] = make_code(group['function'])
        document.create_node_group(**group)
    return document

@NodeConstructor.register('Reranker')
def make_reranker(type: str = 'ModuleReranker', target: Optional[str] = None,
                  output_format: Optional[str] = None, join: Union[bool, str] = False, arguments: Dict = {}):
    return lazyllm.tools.Reranker(type, target=target, output_format=output_format, join=join, **arguments)

class JoinFormatter(lazyllm.components.FormatterBase):
    def __init__(self, type, *, names=None, symbol=None):
        self.type = type
        self.names = names
        self.symbol = symbol

    def _parse_py_data_by_formatter(self, data):
        if self.type == 'sum':
            assert len(data) > 0, 'Cannot sum empty inputs'
            if isinstance(data[0], str): return ''.join(data)
            return sum(data, type(data[0])())
        elif self.type == 'stack':
            return list(data) if isinstance(data, package) else [data,]
        elif self.type == 'to_dict':
            assert self.names and len(self.names) == len(data)
            return {k: v for k, v in zip(self.names, data)}
        elif self.type == 'join':
            symbol = self.symbol or ''
            return symbol.join([str(d) for d in data])
        else:
            raise TypeError('type should be one of sum/stack/to_dict/join')

@NodeConstructor.register('JoinFormatter')
def make_join_formatter(type='sum', names=None, symbol=None):
    if type == 'file': return make_formatter('file', rule='merge')
    return JoinFormatter(type, names=names, symbol=symbol)

@NodeConstructor.register('Formatter')
def make_formatter(ftype, rule):
    return getattr(lazyllm.formatter, ftype)(formatter=rule)

def return_a_wrapper_func(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper_func

def _get_tools(tools):
    callable_list = []
    for rid in tools:  # `tools` is a list of ids in engine's resources
        node = Engine().build_node(rid)
        if type(node.func).__name__ == "CodeBlock":
            wrapper_func = return_a_wrapper_func(node.func._ori_func)
        else:
            wrapper_func = return_a_wrapper_func(node.func)
        wrapper_func.__name__ = node.name
        callable_list.append(wrapper_func)
    return callable_list

@NodeConstructor.register('ToolsForLLM')
def make_tools_for_llm(tools: List[str]):
    return lazyllm.tools.ToolManager(_get_tools(tools))

@NodeConstructor.register('FunctionCall')
def make_fc(llm: str, tools: List[str], algorithm: Optional[str] = None):
    f = lazyllm.tools.PlanAndSolveAgent if algorithm == 'PlanAndSolve' else \
        lazyllm.tools.ReWOOAgent if algorithm == 'ReWOO' else \
        lazyllm.tools.ReactAgent if algorithm == 'React' else lazyllm.tools.FunctionCallAgent
    return f(Engine().build_node(llm).func, _get_tools(tools))

@NodeConstructor.register('HttpTool')
def make_http_tool(method: Optional[str] = None,
                   url: Optional[str] = None,
                   params: Optional[Dict[str, str]] = None,
                   headers: Optional[Dict[str, str]] = None,
                   body: Optional[str] = None,
                   timeout: int = 10,
                   proxies: Optional[Dict[str, str]] = None,
                   code_str: Optional[str] = None,
                   vars_for_code: Optional[Dict[str, Any]] = None,
                   doc: Optional[str] = None):
    instance = lazyllm.tools.HttpTool(method, url, params, headers, body, timeout, proxies,
                                      code_str, vars_for_code)
    if doc:
        instance.__doc__ = doc
    return instance


class VQA(lazyllm.Module):
    def __init__(self, base_model: Union[str, lazyllm.TrainableModule], file_resource_id: Optional[str]):
        super().__init__()
        self.vqa = self._vqa = (lazyllm.TrainableModule(base_model).deploy_method(lazyllm.deploy.LMDeploy)
                                if not isinstance(base_model, lazyllm.TrainableModule) else base_model)
        self._file_resource_id = file_resource_id
        if file_resource_id:
            with pipeline() as self.vqa:
                self.vqa.file = Engine().build_node(file_resource_id).func
                self.vqa.vqa = self._vqa | lazyllm.bind(self.vqa.input, lazyllm._0)

    def status(self, task_name: Optional[str] = None):
        return self._vqa.status(task_name)

    def share(self, prompt: str):
        shared_vqa = self._vqa.share(prompt=prompt)
        return VQA(shared_vqa, self._file_resource_id)

    def forward(self, *args, **kw):
        return self.vqa(*args, **kw)

    @property
    def stream(self):
        return self._vqa._stream

    @stream.setter
    def stream(self, v: bool):
        self._vqa._stream = v


@NodeConstructor.register('VQA')
def make_vqa(base_model: str, file_resource_id: Optional[str] = None):
    return VQA(base_model, file_resource_id)


@NodeConstructor.register('SharedLLM')
def make_shared_llm(llm: str, prompt: Optional[str] = None, stream: Optional[bool] = None,
                    file_resource_id: Optional[str] = None):
    llm = Engine().build_node(llm).func
    if file_resource_id: assert isinstance(llm, VQA), 'file_resource_id is only supported in VQA'
    r = VQA(llm._vqa.share(prompt=prompt), file_resource_id) if file_resource_id else llm.share(prompt=prompt)
    if stream is not None: r.stream = stream
    return r


@NodeConstructor.register('STT')
def make_stt(base_model: str):
    # TODO: support multi-files with pictures
    def cond(x):
        if '<lazyllm-query>' in x:
            for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']:
                if ext in x or ext.upper() in x:
                    return True
        return False

    return lazyllm.ActionModule(lazyllm.ifs(cond, tpath=lazyllm.TrainableModule(base_model), fpath=lazyllm.Identity()))


@NodeConstructor.register('Constant')
def make_constant(value: Any):
    return (lambda *args, **kw: value)


class FileResource(object):
    def __init__(self, id) -> None:
        self.id = id

    def __call__(self, *args, **kw) -> Union[str, List[str]]:
        return lazyllm.globals['lazyllm_files'].get(self.id)


@NodeConstructor.register('File')
def make_file(id: str):
    return FileResource(id)
