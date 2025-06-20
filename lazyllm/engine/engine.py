from typing import List, Tuple, Dict, Type, Optional, Union, Any, overload
import lazyllm
from lazyllm import graph, switch, pipeline, package
from lazyllm.tools import IntentClassifier, SqlManager
from lazyllm.common import compile_func
from lazyllm.components.formatter.formatterbase import _lazyllm_get_file_list
from .node import all_nodes, Node
from .node_meta_hook import NodeMetaHook
import inspect
import functools
from itertools import repeat
import copy
from abc import ABC, abstractclassmethod
from enum import Enum
from datetime import datetime, timedelta
import requests
import json
import sys

# Each session will have a separate engine
class Engine(ABC):
    __default_engine__ = None
    REPORT_URL = ""

    class DefaultLockedDict(dict):
        def __init__(self, default_data, *args, **kwargs):
            self._default_keys = set(default_data.keys())  # 记录默认 key
            super().__init__(default_data)
            self.update(*args, **kwargs)

        def __setitem__(self, key, value):
            if key in self._default_keys: return
            super(__class__, self).__setitem__(key, value)

        def update(self, __other=None, **kw):
            if __other:
                has_kv = hasattr(__other, 'keys') and callable(__other.keys) and hasattr(__other, 'items')
                [self.__setitem__(k, v) for k, v in (__other.items() if has_kv else __other)
                 if k not in self._default_keys]
            if kw:
                [self.__setitem__(k, v) for k, v in kw.items() if k not in self._default_keys]

        def __delitem__(self, key):
            if key in self._default_keys: return
            super(__class__, self).__delitem__(key)

        def pop(self, key, __default=None):
            if key in self._default_keys: return __default
            return super(__class__, self).pop(key, __default)

    def __init__(self):
        self._nodes: Engine.DefaultLockedDict[str, Node] = Engine.DefaultLockedDict({
            '__start__': Node(id='__start__', kind='__start__', name='__start__'),
            '__end__': Node(id='__end__', kind='__end__', name='__end__')})
        self._key_db_connect_message = None

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

    def set_db_connect_message(self, key_db_connect_message: Optional[Union[Dict, str]]) -> None:
        if isinstance(key_db_connect_message, str):
            key_db_connect_message = json.loads(key_db_connect_message)

        if not isinstance(key_db_connect_message, dict):
            raise TypeError("The database connection information only supports dict and str, "
                            f"not {type(key_db_connect_message)}.")

        self._key_db_connect_message = key_db_connect_message

    @property
    def key_db_connect_message(self):
        return self._key_db_connect_message

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

    @abstractclassmethod
    def launch_localllm_train_service(self): pass

    @abstractclassmethod
    def launch_localllm_infer_service(self): pass

    @abstractclassmethod
    def get_infra_handle(self, token, mid) -> lazyllm.TrainableModule: pass


class NodeConstructor(object):
    builder_methods = dict()

    @classmethod
    def register(cls, *names: Union[List[str], str], subitems: Optional[Union[str, List[str]]] = None,
                 need_id: bool = False):
        if len(names) == 1 and isinstance(names[0], (tuple, list)): names = names[0]

        def impl(f):
            for name in names:
                cls.builder_methods[name] = (f, subitems, need_id)
            return f
        return impl

    # build node recursively
    def build(self, node: Node):
        if node.kind.startswith('__') and node.kind.endswith('__'):
            return None
        node_args = copy.copy(node.args)
        node.arg_names = node_args.pop('_lazyllm_arg_names', None) if isinstance(node_args, dict) else None
        node.enable_data_reflow = (node_args.pop('_lazyllm_enable_report', False)
                                   if isinstance(node_args, dict) else False)
        if node.kind in NodeConstructor.builder_methods:
            createf, node.subitem_name, need_id = NodeConstructor.builder_methods[node.kind]
            kw = {'_node_id': node.id} if need_id else {}
            node.func = createf(**node_args, **kw) if isinstance(node_args, dict) and set(node_args.keys()).issubset(
                set(inspect.getfullargspec(createf).args)) else createf(node_args, **kw)
            self._process_hook(node, node.func)
            return node

        node_msgs = all_nodes[node.kind]
        init_args, build_args, other_args = dict(), dict(), dict()

        def get_args(cls, key, value, builder_key=None):
            node_args = node_msgs[cls][builder_key][key] if builder_key else node_msgs[cls][key]
            if node_args.type == Node:
                return Engine().build_node(value).func
            return node_args.getattr_f(value) if node_args.getattr_f else value

        for key, value in node_args.items():
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


@NodeConstructor.register('Diverter', subitems=['nodes:list'])
def make_diverter(nodes: List[dict]):
    return lazyllm.diverter([_build_pipeline(node) for node in nodes])


@NodeConstructor.register('Warp', subitems=['nodes', 'resources'])
def make_warp(nodes: List[dict], edges: List[dict] = [], resources: List[dict] = [],
              batch_flags: Optional[List[int]] = None):
    wp = lazyllm.warp(make_graph(nodes, edges, resources, enable_server=False))
    if batch_flags and len(batch_flags) > 1:
        def transform(*args):
            args = [a if b else repeat(a) for a, b in zip(args, batch_flags)]
            args = [lazyllm.package(a) for a in zip(*args)]
            return args
        wp = lazyllm.pipeline(transform, wp)
    return wp


@NodeConstructor.register('Loop', subitems=['nodes', 'resources'])
def make_loop(nodes: List[dict], edges: List[dict] = [], resources: List[dict] = [],
              stop_condition: Optional[str] = None, judge_on_full_input: bool = True, count=sys.maxsize):
    assert stop_condition is not None or count > 1, 'stop_condition or count is required'
    if stop_condition is not None:
        stop_condition = make_code(stop_condition)
    return lazyllm.loop(make_graph(nodes, edges, resources, enable_server=False),
                        stop_condition=stop_condition, judge_on_full_input=judge_on_full_input, count=count)


@NodeConstructor.register('Ifs', subitems=['true', 'false'])
def make_ifs(cond: str, true: List[dict], false: List[dict], judge_on_full_input: bool = True):
    assert judge_on_full_input, 'judge_on_full_input only support True now'
    return lazyllm.ifs(make_code(cond), tpath=_build_pipeline(true), fpath=_build_pipeline(false))


@NodeConstructor.register('LocalLLM')
def make_local_llm(base_model: str, target_path: str = '', prompt: str = '', stream: bool = False,
                   return_trace: bool = False, deploy_method: str = 'auto', url: Optional[str] = None,
                   history: Optional[List[List[str]]] = None):
    if history and not (isinstance(history, list) and all(len(h) == 2 and isinstance(h, list) for h in history)):
        raise TypeError('history must be List[List[str, str]]')
    deploy_method = getattr(lazyllm.deploy, deploy_method)
    m = lazyllm.TrainableModule(base_model, target_path, stream=stream, return_trace=return_trace)
    m.prompt(prompt, history=history)
    if deploy_method is lazyllm.deploy.AutoDeploy:
        m.deploy_method(deploy_method)
    else:
        m.deploy_method(deploy_method, url=url)
    return m


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


@NodeConstructor.register('Document', need_id=True)
def make_document(dataset_path: str, _node_id: str, embed: Node = None, create_ui: bool = False, server: bool = False,
                  node_group: List[Dict] = [], activated_groups: List[Tuple[str, Optional[List[Node]]]] = []):
    groups = [[g, None] if isinstance(g, str) else g for g in activated_groups]
    groups += [[g['name'], g.pop('embed', None)] for g in node_group]
    groups = [[g, e] if (not e or isinstance(e, list)) else [g, [e]] for g, e in groups]
    embed = {e: Engine().build_node(e).func for e in set(sum([g[1] for g in groups if g[1]], []))}
    document = lazyllm.tools.rag.Document(dataset_path, embed or None, server=server, manager=create_ui, name=_node_id)

    for group in node_group:
        if group['transform'] == 'LLMParser':
            group['transform'] = 'llm'
            group['llm'] = Engine().build_node(group['llm']).func
        elif group['transform'] == 'FuncNode':
            group['transform'] = 'function'
            group['function'] = make_code(group['function'])
        document.create_node_group(**group)

    [document.activate_group(g, e) for g, e in groups]
    return document


@NodeConstructor.register('Retriever')
def make_retriever(doc: str, group_name: str, similarity: str = 'cosine', similarity_cut_off: float = float("-inf"),
                   index: str = 'default', topk: int = 6, target: str = None, output_format: str = None,
                   join: bool = False):
    return lazyllm.tools.Retriever(Engine().build_node(doc).func, group_name=group_name, similarity=similarity,
                                   similarity_cut_off=similarity_cut_off, index=index, topk=topk, embed_keys=[],
                                   target=target, output_format=output_format, join=join)


@NodeConstructor.register('Reranker')
def make_reranker(type: str = 'ModuleReranker', target: Optional[str] = None,
                  output_format: Optional[str] = None, join: Union[bool, str] = False, arguments: Dict = {}):
    if type == 'ModuleReranker' and (node := Engine().build_node(arguments['model'])):
        arguments['model'] = node.func
    return lazyllm.tools.Reranker(type, target=target, output_format=output_format, join=join, **arguments)

class JoinFormatter(lazyllm.components.FormatterBase):
    def __init__(self, type, *, names=None, symbol=None):
        self.type = type
        self.names = names
        self.symbol = symbol

    def _load(self, msg: str):
        return lazyllm.components.formatter.decode_query_with_filepaths(msg)

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

@NodeConstructor.register('ToolsForLLM', subitems=['tools'])
def make_tools_for_llm(tools: List[str]):
    return lazyllm.tools.ToolManager(_get_tools(tools))

@NodeConstructor.register('FunctionCall', subitems=['tools'])
def make_fc(llm: str, tools: List[str], algorithm: Optional[str] = None):
    f = lazyllm.tools.PlanAndSolveAgent if algorithm == 'PlanAndSolve' else \
        lazyllm.tools.ReWOOAgent if algorithm == 'ReWOO' else \
        lazyllm.tools.ReactAgent if algorithm == 'React' else lazyllm.tools.FunctionCallAgent
    return f(Engine().build_node(llm).func, _get_tools(tools))

class AuthenticationFailedError(Exception):
    def __init__(self, message="Authentication failed for the given user and tool."):
        self._message = message
        super().__init__(self._message)

class TokenExpiredError(Exception):
    """Access token expired"""
    pass

class TokenRefreshError(Exception):
    """Access key request failed"""
    pass

class AuthType(Enum):
    SERVICE_API = "service_api"
    OAUTH = "oauth"
    OIDC = "oidc"

class SharedHttpTool(lazyllm.tools.HttpTool):
    def __init__(self,
                 method: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict[str, str]] = None,
                 code_str: Optional[str] = None,
                 vars_for_code: Optional[Dict[str, Any]] = None,
                 outputs: Optional[List[str]] = None,
                 extract_from_result: Optional[bool] = None,
                 authentication_type: Optional[str] = None,
                 tool_api_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 share_key: bool = False):
        super().__init__(method, url, params, headers, body, timeout, proxies,
                         code_str, vars_for_code, outputs, extract_from_result)
        self._token_type = authentication_type
        self._tool_api_id = tool_api_id
        self._user_id = user_id
        self._share_key = share_key
        self._key_db_connect_message = Engine().key_db_connect_message
        if self._key_db_connect_message:
            self._sql_manager = SqlManager(
                db_type=self._key_db_connect_message['db_type'],
                user=self._key_db_connect_message.get('user', None),
                password=self._key_db_connect_message.get('password', None),
                host=self._key_db_connect_message.get('host', None),
                port=self._key_db_connect_message.get('port', None),
                db_name=self._key_db_connect_message['db_name'],
                options_str=self._key_db_connect_message.get('options_str', None),
                tables_info_dict=self._key_db_connect_message.get('tables_info_dict', None),
            )
        self._default_expired_days = 3

    def _process_api_key(self, headers, params):
        if not self._token_type:
            return headers, params
        if self._token_type == AuthType.SERVICE_API.value:
            if self._location == "header":
                headers[self._param_name] = self._token if self._token.startswith("Bearer") \
                    else "Bearer " + self._token
            elif self._location == "query":
                params = params or {}
                params[self._param_name] = self._token
            else:
                raise TypeError("The Service API authentication type only supports ['header', 'query'], "
                                f"not {self._location}.")
        elif self._token_type == AuthType.OAUTH.value:
            headers['Authorization'] = f"Bearer {self._token}"
        else:
            raise TypeError("Currently, tool authentication only supports ['service_api', 'oauth'] types, "
                            f"and does not support {self._token_type} type.")
        return headers, params

    def valid_key(self):
        if not self._token_type:
            return True
        table_name = self._key_db_connect_message.get('tables_info_dict', {}).get('tables', [])[0]['name']
        SQL_SELECT = (
            f"SELECT id, tool_api_id, endpoint_url, client_id, client_secret, user_id, location, param_name, token, "
            f"refresh_token, token_type, expires_at FROM {table_name} "
            f"WHERE tool_api_id = {self._tool_api_id} AND is_auth_success = True AND token_type = '{self._token_type}'"
        )
        if self._share_key:
            ret = self._fetch_valid_key(SQL_SELECT + " AND is_share = True")
            if not ret:
                raise AuthenticationFailedError(f"Authentication failed for share_key=True and "
                                                f"tool_api_id='{self._tool_api_id}'")
        else:
            ret = self._fetch_valid_key(SQL_SELECT + f" AND user_id = '{self._user_id}'")
            if not ret:
                raise AuthenticationFailedError(f"Authentication failed for user_id='{self._user_id}' and "
                                                f"tool_api_id='{self._tool_api_id}'")

        if self._token_type == AuthType.SERVICE_API.value:
            self._token = ret['token']
            self._location = ret['location']
            self._param_name = ret['param_name']
        elif self._token_type == AuthType.OAUTH.value:
            try:
                expires_at = datetime.strptime(ret['expires_at'], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                expires_at = datetime.strptime(ret['expires_at'], "%Y-%m-%d %H:%M:%S.%f")
            self._token = self._validate_and_refresh_token(
                id=ret['id'],
                client_id=ret['client_id'],
                client_secret=ret['client_secret'],
                endpoint_url=ret['endpoint_url'],
                token=ret['token'],
                refresh_token=ret['refresh_token'],
                expires_at=expires_at,
                table_name=table_name)
        elif self._token_type == AuthType.OIDC.value:
            raise TypeError("OIDC authentication is not currently supported.")
        else:
            raise TypeError("The authentication type only supports ['no authentication', 'service_api', "
                            f"'oauth', 'oidc'], and does not support type {self._token_type}.")

    def _fetch_valid_key(self, query: str):
        ret = self._sql_manager.execute_query(query)
        ret = json.loads(ret)
        return ret[0] if ret else None

    def _validate_and_refresh_token(self, id: int, client_id: str, client_secret: str, endpoint_url: str,
                                    token: str, refresh_token: str, expires_at: datetime, table_name):
        now = datetime.now()
        # 1、Access token has not expired
        if now < expires_at:
            if not refresh_token:
                # Update only the expiration time
                new_expires_at = now + timedelta(days=self._default_expired_days)
                self._sql_manager.execute_commit(f"UPDATE {table_name} SET expires_at = "
                                                 f"'{new_expires_at}' WHERE id = {id}")
            return token

        # 2、Access token expired
        if not refresh_token:
            raise TokenExpiredError("Access key has expired, and no refresh key was provided.")

        # 3、Request a new access token with the refresh_token
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {client_secret}"}
        data = {"client_id": client_id, "grant_type": "refresh_token", "refresh_token": refresh_token}
        with requests.post(endpoint_url, json=data, headers=headers) as r:
            if r.status_code != 200:
                raise TokenRefreshError(f"Request failed, status code: {r.status_code}, message: {r.text}")

            data = r.json()
            new_token = data.get("access_token")
            new_refresh_token = data.get("refresh_token")
            new_expires_at = data.get("expires_in")

            # update db
            self._sql_manager.execute_commit(
                f"UPDATE {table_name} SET token = '{new_token}', refresh_token = '{new_refresh_token}', "
                f"expires_at = '{datetime.fromtimestamp(new_expires_at)}' where id = {id}")
            return new_token

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
                   doc: Optional[str] = None,
                   outputs: Optional[List[str]] = None,
                   extract_from_result: Optional[bool] = None,
                   authentication_type: Optional[str] = None,
                   tool_api_id: Optional[str] = None,
                   user_id: Optional[str] = None,
                   share_key: bool = False):
    instance = SharedHttpTool(method, url, params, headers, body, timeout, proxies,
                              code_str, vars_for_code, outputs, extract_from_result, authentication_type,
                              tool_api_id, user_id, share_key)
    if doc:
        instance.__doc__ = doc
    return instance


class VQA(lazyllm.Module):
    def __init__(self, base_model: Union[str, lazyllm.TrainableModule], file_resource_id: Optional[str],
                 prompt: Optional[str] = None):
        super().__init__()
        self.vqa = self._vqa = (lazyllm.TrainableModule(base_model).deploy_method(lazyllm.deploy.LMDeploy)
                                if not isinstance(base_model, lazyllm.TrainableModule) else base_model)
        if prompt: self._vqa.prompt(prompt=prompt)
        self._file_resource_id = file_resource_id
        if file_resource_id:
            with pipeline() as self.vqa:
                self.vqa.file = Engine().build_node(file_resource_id).func
                self.vqa.vqa = self._vqa | lazyllm.bind(self.vqa.input, lazyllm._0)

    def status(self, task_name: Optional[str] = None):
        return self._vqa.status(task_name)

    def share(self, prompt: str, history: Optional[List[List[str]]] = None):
        shared_vqa = self._vqa.share(prompt=prompt, history=history)
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
def make_vqa(base_model: str, file_resource_id: Optional[str] = None, prompt: Optional[str] = None):
    return VQA(base_model, file_resource_id, prompt)


@NodeConstructor.register('SharedLLM')
def make_shared_llm(llm: str, local: bool = True, prompt: Optional[str] = None, token: str = None,
                    stream: Optional[bool] = None, file_resource_id: Optional[str] = None,
                    history: Optional[List[List[str]]] = None):
    if local:
        llm = Engine().build_node(llm).func
        if file_resource_id: assert isinstance(llm, VQA), 'file_resource_id is only supported in VQA'
        r = (VQA(llm._vqa.share(prompt=prompt, history=history), file_resource_id)
             if file_resource_id else llm.share(prompt=prompt, history=history))
    else:
        assert Engine().launch_localllm_infer_service.flag, 'Infer service should start first!'
        r = Engine().get_infra_handle(token, llm)
        if prompt: r.prompt(prompt, history=history)
    if stream is not None: r.stream = stream
    return r


@NodeConstructor.register('OnlineLLM')
def make_online_llm(source: str, base_model: Optional[str] = None, prompt: Optional[str] = None,
                    api_key: Optional[str] = None, secret_key: Optional[str] = None,
                    stream: bool = False, token: Optional[str] = None, base_url: Optional[str] = None,
                    history: Optional[List[List[str]]] = None):
    if source and source.lower() == 'lazyllm':
        return make_shared_llm(base_model, False, prompt, token, stream, history=history)
    else:
        return lazyllm.OnlineChatModule(base_model, source, base_url, stream,
                                        api_key=api_key, secret_key=secret_key).prompt(prompt, history=history)


class LLM(lazyllm.ModuleBase):
    def __init__(self, m: lazyllm.ModuleBase, keys: Optional[List[str]] = None):
        super().__init__()
        self._m = m
        self._m.used_by(self._module_id)
        self._keys = keys

    def forward(self, *args, **kw):
        if self._keys and len(self._keys) > 1:
            assert len(args) == len(self._keys)
            args = ({k: a for k, a in zip(self._keys, args)},)
        else:
            assert len(args) == 1
        return self._m(*args, **kw)

    def share(self, prompt: str, history: Optional[List[List[str]]] = None):
        return LLM(self._m.share(prompt=prompt, history=history), self._keys)


@NodeConstructor.register('LLM')
def make_llm(kw: dict):
    type: str = kw.pop('type')
    keys: Optional[List[str]] = kw.pop('keys', None)
    assert type in ('local', 'online'), f'Invalid type {type} given'
    if type == 'local': return LLM(make_local_llm(**kw), keys)
    elif type == 'online': return LLM(make_online_llm(**kw), keys)


class STT(lazyllm.Module):
    def __init__(self, base_model: Union[str, lazyllm.TrainableModule]):
        super().__init__()
        self._m = lazyllm.TrainableModule(base_model) if isinstance(base_model, str) else base_model.share()

    def forward(self, query: str):
        if '<lazyllm-query>' in query:
            for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']:
                if ext in query or ext.upper() in query:
                    return self._m(query)
        return query

    def share(self, prompt: str = None):
        assert prompt is None, 'STT has no promot'
        return STT(self._m)

    def status(self, task_name: Optional[str] = None):
        return self._m.status(task_name)

    @property
    def stream(self):
        return self._m._stream

    @stream.setter
    def stream(self, v: bool):
        self._m._stream = v


@NodeConstructor.register('STT')
def make_stt(base_model: str):
    return STT(base_model)


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


@NodeConstructor.register("Reader")
def make_simple_reader(file_resource_id: Optional[str] = None):
    if file_resource_id:
        def merge_input(input, extra_file: Union[str, List[str]]):
            if isinstance(input, package):
                input = input[0]
            input = _lazyllm_get_file_list(input)
            input = [input] if isinstance(input, str) else input
            if extra_file is not None:
                extra = [extra_file] if isinstance(extra_file, str) else extra_file
                return input + extra
            else:
                return input
        with pipeline() as ppl:
            ppl.extra_file = Engine().build_node(file_resource_id).func
            ppl.merge = lazyllm.bind(merge_input, ppl.input, lazyllm._0)
            ppl.reader = lazyllm.tools.rag.FileReader()
        return ppl
    else:
        return lazyllm.tools.rag.FileReader()


@NodeConstructor.register("OCR")
def make_ocr(model: Optional[str] = "PP-OCRv5_mobile"):
    if model is None:
        model = "PP-OCRv5_mobile"
    assert model in ["PP-OCRv5_server", "PP-OCRv5_mobile", "PP-OCRv4_server", "PP-OCRv4_mobile"]
    return lazyllm.TrainableModule(base_model=model).start()


@NodeConstructor.register("ParameterExtractor")
def make_parameter_extractor(base_model: str, param: list[str], type: list[str],
                             description: list[str], require: list[bool]):
    base_model = Engine().build_node(base_model).func
    return lazyllm.tools.ParameterExtractor(base_model, param, type, description, require)


@NodeConstructor.register("QustionRewrite")
def make_qustion_rewrite(base_model: str, rewrite_prompt: str = "", formatter: str = "str"):
    base_model = Engine().build_node(base_model).func
    return lazyllm.tools.QustionRewrite(base_model, rewrite_prompt, formatter)


@NodeConstructor.register("CodeGenerator")
def make_code_generator(base_model: str, prompt: str = ""):
    base_model = Engine().build_node(base_model).func
    return lazyllm.tools.CodeGenerator(base_model, prompt)
