import ast
import copy
from dataclasses import dataclass, replace
import json as std_json
import json5 as json
import lazyllm
import docstring_parser
import os
from lazyllm.module import ModuleBase
from lazyllm.common import LazyLLMRegisterMetaClass, kwargs, _change_exception_type
from lazyllm.common.utils import SecurityVisitor
from lazyllm.flow.flow import FlowException
from typing import Callable, Any, Union, Optional, get_type_hints, List, Dict, Type, Set
import inspect
import re
import uuid
from pydantic import create_model, BaseModel, ConfigDict, ValidationError
from lazyllm import LOG, locals as lazyllm_locals
from .toolError import exception_failure, tool_failure
from .tool_runtime import (
    PreparedToolCall,
    ResolvedToolAccess,
    ToolExecutionBatch,
    ToolExecutionDisposition,
    ToolExecutionRecord,
    ToolRuntimeMetadata,
    _accesses_conflict,
    _get_tool_runtime_metadata,
    _set_tool_runtime_metadata,
)
from typing import *  # noqa F403, to import all types for compile_func(), do not remove

# ---------------------------------------------------------------------------- #

_UNSET = object()


@dataclass(frozen=True)
class _ValidatedToolInput:
    value: Any


@dataclass(frozen=True)
class _PreparedToolInvocation:
    prepared: PreparedToolCall
    tool: Any
    validated_arguments: Optional[Dict[str, Any]]
    failure: Any = None


def _levenshtein(left, right):
    left, right = left.lower(), right.lower()
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for offset, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[offset] + 1,
                previous[offset - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _closest_tool_name(requested, available):
    ranked = sorted((_levenshtein(requested, name), name) for name in set(available))
    if not ranked:
        return None
    distance, name = ranked[0]
    threshold = min(3, max(1, len(requested) // 3))
    return name if distance <= threshold else None


class ModuleTool(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, verbose: bool = False, return_trace: bool = False, execute_in_sandbox: bool = True,
                 apply_func: Optional[Callable] = None, schema_func: Optional[Callable] = None,
                 metadata_func: Optional[Callable] = None):
        super().__init__(return_trace=return_trace)
        if apply_func is not None:
            self.apply = apply_func
        self._verbose = verbose
        func = getattr(self.apply, '__func__', self.apply)
        if callable(func) and getattr(func, '__closure__', None):
            self._name = getattr(func, '__name__', None)
        else:
            self._name = self.__class__.__name__
        self._description = self.apply.__doc__\
            if hasattr(self.apply, '__doc__') and self.apply.__doc__ is not None\
            else (_ for _ in ()).throw(ValueError('Function must have a docstring'))
        # strip space(s) and newlines before and after docstring, as RewooAgent requires
        self._description = self._description.strip(' \n')
        metadata = _get_tool_runtime_metadata(metadata_func or schema_func or apply_func or self.__class__.apply)
        self._runtime_metadata = metadata or ToolRuntimeMetadata(execute_in_sandbox=execute_in_sandbox)
        self._params_schema = self._load_function_schema(schema_func or self.__class__.apply)

    @staticmethod
    def _safe_eval_type(type_str: str, context: str) -> Any:
        try:
            tree = ast.parse(type_str, mode='eval')
            SecurityVisitor().visit(tree)
        except SyntaxError:
            return Any
        except ValueError as e:
            raise ValueError(f'Unsafe type expression in docstring ({context}): {e}')
        try:
            return eval(type_str, globals())  # noqa S307
        except Exception:
            raise NameError(f'Unknown type "{type_str}" in docstring ({context}).')

    @staticmethod
    def _parse_type_from_docstring(parsed_docstring) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}
        for param in parsed_docstring.params:
            if not param.type_name:
                continue
            hints[param.arg_name] = ModuleTool._safe_eval_type(
                param.type_name, f'parameter "{param.arg_name}"')
        if parsed_docstring.returns and parsed_docstring.returns.type_name:
            hints['return'] = ModuleTool._safe_eval_type(
                parsed_docstring.returns.type_name, 'return type')
        return hints

    def _load_function_schema(self, func: Callable) -> Type[BaseModel]:
        parsed_docstring = docstring_parser.parse(self._description)
        doc_type_hints = self._parse_type_from_docstring(parsed_docstring)

        # Resolve annotations in the callable's defining module. Host toolkits
        # commonly use structured request models that are not imported here.
        annotation_func = inspect.unwrap(func)
        func_type_hints = get_type_hints(
            annotation_func, getattr(annotation_func, '__globals__', globals()), locals())

        func_return = func_type_hints.get('return')
        doc_return = doc_type_hints.get('return')
        if func_return is not None and doc_return is not None and doc_return is not Any and func_return != doc_return:
            raise TypeError(
                f'return info in docstring ({doc_return}) is different from '
                f'function prototype ({func_return}) in {func.__name__}'
            )

        signature = inspect.signature(func)
        has_var_args = any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in signature.parameters.values())
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values())

        if has_var_args:
            self._type_hints = doc_type_hints
            param_names = [p.arg_name for p in parsed_docstring.params]
            fields = {name: (doc_type_hints.get(name, Any), ...) for name in param_names}
        else:
            self._type_hints = func_type_hints
            for name, t in doc_type_hints.items(): self._type_hints.setdefault(name, t)
            fields = {
                name: (self._type_hints.get(name, Any), param.default
                       if param.default is not inspect.Parameter.empty else ...)
                for name, param in signature.parameters.items() if name != 'self'}

        self._return_type = self._type_hints.get('return') if self._type_hints else None
        return create_model(
            self._name,
            __config__=ConfigDict(extra='allow' if has_var_keyword else 'forbid'),
            **fields,
        )

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def execute_in_sandbox(self) -> bool:
        return self._runtime_metadata.execute_in_sandbox

    @execute_in_sandbox.setter
    def execute_in_sandbox(self, value: bool):
        self._runtime_metadata = replace(self._runtime_metadata, execute_in_sandbox=value)

    @property
    def runtime_metadata(self) -> ToolRuntimeMetadata:
        return self._runtime_metadata

    def _resolve_runtime_access(self, validated_arguments: Dict[str, Any]) -> ResolvedToolAccess:
        try:
            return self._runtime_metadata.resolve(validated_arguments)
        except Exception as error:
            LOG.warning(
                f'[ToolRuntime] tool={self.name!r} could not resolve resource keys; '
                f'using exclusive execution: {type(error).__name__}: {error}'
            )
            return ResolvedToolAccess(exclusive=True)

    @property
    def input_files_parm(self) -> str:
        return self._runtime_metadata.input_files_parm

    @input_files_parm.setter
    def input_files_parm(self, value: str):
        assert isinstance(value, str), f'input_files_parm must be a string, but got {type(value)}'
        self._runtime_metadata = replace(self._runtime_metadata, input_files_parm=value)

    @property
    def output_files_parm(self) -> str:
        return self._runtime_metadata.output_files_parm

    @output_files_parm.setter
    def output_files_parm(self, value: str):
        assert isinstance(value, str), f'output_files_parm must be a string, but got {type(value)}'
        self._runtime_metadata = replace(self._runtime_metadata, output_files_parm=value)

    @property
    def output_files(self) -> List[str]:
        return list(self._runtime_metadata.output_files)

    @output_files.setter
    def output_files(self, value: List[str]):
        assert isinstance(value, list) and all(isinstance(item, str) for item in value), \
            f'output_files must be a list of strings, but got {type(value)}'
        self._runtime_metadata = replace(self._runtime_metadata, output_files=tuple(value))

    @property
    def params_schema(self) -> Type[BaseModel]:
        return self._params_schema

    @property
    def args(self) -> Dict[str, Any]:
        return self._params_schema.model_json_schema()['properties']

    @property
    def required_args(self) -> Set[str]:
        return set(self._params_schema.model_json_schema().get('required', []))

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError('Implement apply function in subclass')

    @staticmethod
    def _rebuild_from_reduce(module_id, rebuild_object):
        if isinstance(rebuild_object, type):
            return rebuild_object()._set_mid(module_id)
        elif callable(rebuild_object):
            register('tool')(rebuild_object)
            cls_to_use = getattr(lazyllm.tool, rebuild_object.__name__)
            return cls_to_use()._set_mid(module_id)
        else:
            raise ValueError(f'Invalid rebuild object in ModuleTool: {rebuild_object}')

    @staticmethod
    def _get_orig_apply_func(apply_method):
        func = getattr(apply_method, '__func__', apply_method)
        if not callable(func) or not getattr(func, '__closure__', None):
            return None
        for cell in func.__closure__:
            try:
                c = cell.cell_contents
                if callable(c) and not isinstance(c, type) and getattr(c, '__name__', None):
                    return c
            except ValueError:
                pass
        return None

    def __reduce__(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON':
            orig = ModuleTool._get_orig_apply_func(self.apply)
            if orig is not None and orig.__module__ != '__main__':
                import types
                orig = types.FunctionType(
                    orig.__code__, orig.__globals__, orig.__name__,
                    orig.__defaults__, orig.__closure__)
                orig.__module__ = '__main__'
            return (ModuleTool._rebuild_from_reduce, (self._module_id, orig or self.__class__))
        return super().__reduce__()

    def _validate_input(self, tool_input: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        input_params = self._params_schema
        if isinstance(tool_input, dict):
            if input_params is not None:
                ret = input_params.model_validate(tool_input)
                return {key: getattr(ret, key) for key in ret.model_dump().keys() if key in tool_input}
            return tool_input
        elif isinstance(tool_input, str):
            if input_params is not None:
                key = next(iter(input_params.model_fields.keys()))
                input_params.model_validate({key: tool_input})
                arg_type = self._type_hints.get(key)
                if arg_type:
                    return {key: arg_type(tool_input)}
                return {key: tool_input}
            key = next(iter(self._type_hints.keys()))
            return {key: self._type_hints[key](tool_input)}
        else:
            raise TypeError(f'tool_input {tool_input} only supports dict and str.')

    def validate_parameters(self, arguments: Dict[str, Any]) -> bool:
        if len(self.required_args.difference(set(arguments.keys()))) == 0:
            # contains all required parameters
            try:
                self._validate_input(arguments)
                return True
            except ValidationError:
                return False
        return False

    def forward(self, tool_input: Union[str, Dict[str, Any]], verbose: bool = False) -> Any:
        validated = tool_input.value if isinstance(tool_input, _ValidatedToolInput) \
            else self._validate_input(tool_input)
        return self._execute_validated(validated, verbose=verbose)

    def _execute_validated(self, val_input: Any, verbose: bool = False) -> Any:
        if isinstance(val_input, dict):
            ret = self.apply(**val_input)
        else:
            ret = self.apply(val_input)
        if verbose or self._verbose:
            lazyllm.LOG.debug(f'The output of tool {self.name} is {ret}')

        return ret

    def to_sandbox_code(self, tool_arguments: Dict[str, Any]) -> str:
        from lazyllm.tools.sandbox.sandbox_base import SANDBOX_TOOL_RESULT_PREFIX
        args_dump = lazyllm.dump_obj(tool_arguments)
        tool_dump = lazyllm.dump_obj(self)
        return f'''
import base64
import cloudpickle
tool = cloudpickle.loads(base64.b64decode({repr(tool_dump)}.encode('utf-8')))
kwargs = cloudpickle.loads(base64.b64decode({repr(args_dump)}.encode('utf-8')))
from lazyllm.tools.agent.toolsManager import _ValidatedToolInput
result = tool(_ValidatedToolInput(kwargs))
print(f'{SANDBOX_TOOL_RESULT_PREFIX}{{result}}')  # noqa print
'''

_base_fc_register = lazyllm.Register(ModuleTool, ['apply'], default_group='tool')
_RUNTIME_METADATA_FIELDS = {
    'execute_in_sandbox',
    'input_files_parm',
    'output_files_parm',
    'output_files',
    'read_keys',
    'write_keys',
    'exclusive',
    'polling',
}


def _runtime_metadata_from_options(options):
    unknown = set(options) - _RUNTIME_METADATA_FIELDS
    assert not unknown, f'Only allowed parameters: {_RUNTIME_METADATA_FIELDS}, but got {unknown}'
    normalized = dict(options)
    if 'output_files' in normalized:
        output_files = normalized['output_files']
        assert isinstance(output_files, list) and all(isinstance(item, str) for item in output_files), \
            f'output_files must be a list of strings, but got {type(output_files)}'
        normalized['output_files'] = tuple(output_files)
    ToolRuntimeMetadata(**normalized)
    return normalized


class _FunctionCallRegister:
    def __call__(self, group=_UNSET, *, rewrite_func=None, **options):
        if group is not _UNSET and not isinstance(group, (str, type)):
            if options or rewrite_func is not None:
                raise TypeError('bare fc_register does not accept registration options')
            return _base_fc_register(group)

        metadata = _runtime_metadata_from_options(options) if options else None
        if group is _UNSET:
            if metadata is None:
                raise TypeError('fc_register requires a callable, group, or runtime metadata')

            def metadata_decorator(func):
                _set_tool_runtime_metadata(func, metadata)
                return func

            return metadata_decorator

        registrar = _base_fc_register(group, rewrite_func=rewrite_func)
        if metadata is None:
            return registrar

        def register_decorator(func, *args):
            _set_tool_runtime_metadata(func, metadata)
            return registrar(func, *args)

        return register_decorator

    def __getattr__(self, name):
        return getattr(_base_fc_register, name)

    def new_group(self, group_name):
        return _base_fc_register.new_group(group_name)


fc_register = _FunctionCallRegister()
register = fc_register
if 'tool' not in LazyLLMRegisterMetaClass.all_clses:
    fc_register.new_group('tool')
if 'builtin_tools' not in LazyLLMRegisterMetaClass.all_clses:
    fc_register.new_group('builtin_tools')
if 'tmp_tool' not in LazyLLMRegisterMetaClass.all_clses:
    fc_register.new_group('tmp_tool')


class MethodModuleTool(ModuleTool):
    def __init__(self, instance: Any, method_name: str, schema_func: Optional[Callable] = None,
                 input_adapter: Optional[Callable] = None):
        object.__setattr__(self, '_instance', instance)
        object.__setattr__(self, '_method_name', method_name)
        object.__setattr__(self, '_input_adapter', input_adapter)
        bound = getattr(instance, method_name)

        def _apply(**kwargs):
            return bound(**kwargs)
        _apply.__doc__ = bound.__doc__ or self._find_inherited_docstring(instance, method_name)
        _apply.__name__ = method_name

        super().__init__(
            execute_in_sandbox=False,
            apply_func=_apply,
            schema_func=schema_func or bound,
            metadata_func=bound,
        )
        self.execute_in_sandbox = False
        self._name = instance.__class__.__name__ if method_name == '__call__' \
            else f'{instance.__class__.__name__}_{method_name}'

    def _validate_input(self, tool_input: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        if self._input_adapter:
            tool_input = self._input_adapter(tool_input)
        return super()._validate_input(tool_input)

    @staticmethod
    def _find_inherited_docstring(instance: Any, method_name: str) -> Optional[str]:
        for cls in type(instance).__mro__[1:]:
            member = cls.__dict__.get(method_name)
            if member is not None and getattr(member, '__doc__', None):
                return member.__doc__
        return None


def _gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring):
    tool_args = tool.args
    args_description = {param.arg_name: param.description for param in parsed_docstring.params}
    args = {}
    for k, v in tool_args.items():
        val = copy.deepcopy(v)
        val.pop('title', None)
        val.pop('default', None)
        args[k] = val if val else {'type': 'string'}
        desc = args_description.get(k, None)
        if desc:
            args[k].update({'description': desc})
    return args


def _build_tool_desc(tool: 'ModuleTool') -> Dict:
    parsed_docstring = docstring_parser.parse(tool.description)
    args = _gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring)
    model_schema = tool.params_schema.model_json_schema()
    parameters = {
        key: copy.deepcopy(value)
        for key, value in model_schema.items()
        if key not in {'title', 'properties'}
    }
    parameters['properties'] = args
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': parsed_docstring.description,
            'parameters': parameters,
        }
    }


class ToolContainer:
    def get_flat_tools(self) -> Dict[str, 'ModuleTool']: raise NotImplementedError
    def get_description(self, active_groups: Optional[Set[str]] = None) -> List[Dict]: raise NotImplementedError
    def get_leaf_names(self) -> List[str]: raise NotImplementedError
    def get_auto_active_groups(self, text: str) -> Set[str]: return set()
    def get_activation_path(self, group_name: str) -> Set[str]: return set()


class ToolGroup(ToolContainer):
    def __init__(self, tools: List[Any], name: str, desc: str = '', lazy: bool = True,
                 prefix: Optional[Union[bool, str]] = True, _outer_prefix: Optional[str] = None,
                 pick_first_valid: bool = False,
                 auto_activate: Optional[List[Union[str, Callable[[str], bool]]]] = None):
        self._name, self._desc, self._lazy = name, desc, lazy and not pick_first_valid
        self._pick_first_valid = pick_first_valid
        self._auto_activate = list(auto_activate) if isinstance(auto_activate, (list, tuple)) \
            else [auto_activate] if auto_activate else []
        for rule in self._auto_activate:
            if not callable(rule) and not isinstance(rule, str):
                raise TypeError('auto_activate rules must be regex strings or callables')
            if isinstance(rule, str): re.compile(rule)
        own_prefix: Optional[str] = name if prefix is True or prefix == '' \
            else None if prefix is False or prefix is None else prefix
        effective_prefix = f'{_outer_prefix}_{own_prefix}' if _outer_prefix and own_prefix \
            else (_outer_prefix or own_prefix)
        self._children: List[Union['ModuleTool', 'ToolGroup']] = [
            _build_tool_from_element(item, _outer_prefix=effective_prefix) for item in tools]
        self._precompute_descs(effective_prefix)
        self._gateway_tool: Optional['ModuleTool'] = self.make_gateway_tool() if self._lazy else None
        if self._gateway_desc is None and self._gateway_tool is not None:
            self._gateway_desc = _build_tool_desc(self._gateway_tool)

    def get_flat_tools(self) -> Dict[str, 'ModuleTool']:
        result: Dict[str, ModuleTool] = {}
        if self._gateway_tool is not None:
            result[self._gateway_tool.name] = self._gateway_tool
        for child, precomputed in zip(self._children, self._expanded_descs):
            if isinstance(child, ToolContainer):
                result.update(child.get_flat_tools())
            else:
                result[precomputed['function']['name']] = child
        return result

    def _precompute_descs(self, effective_prefix: Optional[str] = None):
        self._gateway_desc: Optional[Dict] = None
        self._expanded_descs: List[Union[Dict, 'ToolContainer']] = []
        self._leaf_names: List[str] = []
        for child in self._children:
            if isinstance(child, ToolContainer):
                self._expanded_descs.append(child)
                self._leaf_names.extend(child.get_leaf_names())
            else:
                desc = _build_tool_desc(child)
                final_name = f'{effective_prefix}_{child.name}' if effective_prefix else child.name
                desc['function']['name'] = final_name
                self._expanded_descs.append(desc)
                self._leaf_names.append(final_name)

    def get_description(self, active_groups: Optional[Set[str]] = None) -> List[Dict]:
        if self._pick_first_valid:
            for child, precomputed in zip(self._children, self._expanded_descs):
                if _child_is_valid(child):
                    return child.get_description(active_groups) if isinstance(child, ToolContainer) \
                        else [precomputed]
            return []
        if not self._lazy or (active_groups is not None and self._name in active_groups):
            return [x for item in self._expanded_descs
                    for x in (item.get_description(active_groups) if isinstance(item, ToolContainer) else [item])]
        return [self._gateway_desc]

    def get_leaf_names(self) -> List[str]:
        return [self._gateway_tool.name] if self._lazy else self._leaf_names

    def get_auto_active_groups(self, text: str) -> Set[str]:
        matched = any(
            bool(rule(text)) if callable(rule) else bool(re.search(rule, text, re.IGNORECASE))
            for rule in self._auto_activate
        )
        child_groups = {
            group_name
            for child in self._children if isinstance(child, ToolContainer)
            for group_name in child.get_auto_active_groups(text)
        }
        if matched or child_groups:
            return child_groups | ({self._name} if self._lazy else set())
        return set()

    def get_activation_path(self, group_name: str) -> Set[str]:
        if self._name == group_name:
            return {self._name} if self._lazy else set()
        for child in self._children:
            if not isinstance(child, ToolContainer): continue
            child_path = child.get_activation_path(group_name)
            if child_path:
                return child_path | ({self._name} if self._lazy else set())
        return set()

    def make_gateway_tool(self) -> 'ModuleTool':
        group_name = self._name
        child_names = self._leaf_names

        def _gateway_apply() -> str:
            workspace = lazyllm_locals['_lazyllm_agent'].get('workspace', {})
            active = workspace.setdefault('_active_groups', [])
            if group_name not in active:
                active.append(group_name)
            return (f'Activated Toolkit "{group_name}". '
                    f'Available tools: {", ".join(child_names)}')

        group_desc = docstring_parser.parse(self._desc).description if self._desc else ''
        desc = (
            f'Gateway to activate the "{group_name}" Toolkit '
            f'{"(usage: " + group_desc + ")" if group_desc else ""}'
            f'. You MUST call this tool before using any tool from {group_name}.'
        )
        _gateway_apply.__doc__ = (
            f'{desc}\n\nReturns:\n    str: List of available tool names in this group.'
        )
        _gateway_apply.__name__ = f'get_{group_name}_methods'

        register('tmp_tool')(_gateway_apply)
        tool_cls = lazyllm.tmp_tool.resolve(_gateway_apply.__name__)
        tool = tool_cls()
        lazyllm.tmp_tool.remove(_gateway_apply.__name__)
        return tool


class SkipMixin:
    @staticmethod
    def _normalize_source(src):
        if callable(src):
            try:
                sig = inspect.signature(src)
                params = [p for p in sig.parameters.values()
                          if p.default is inspect.Parameter.empty
                          and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            except (ValueError, TypeError):
                params = []
            if not params:
                return lambda _inst: src()
        return src

    def __init__(self, key_source: Union[str, Callable, List[Union[str, Callable]], None] = None):
        if isinstance(key_source, list):
            self._key_source = [SkipMixin._normalize_source(s) for s in key_source]
        else:
            self._key_source = SkipMixin._normalize_source(key_source) if key_source is not None else None

    def _resolve_key_by_string(self, source: str) -> Optional[str]:
        try:
            if '.' not in source:
                return lazyllm.globals.config[source] or None
            prefix, _, attr = source.partition('.')
            if prefix == 'env': return os.environ.get(attr) or None
            elif prefix == 'config': return lazyllm.config[attr] or None
            elif prefix == 'globals':
                if attr.startswith('config.'): return lazyllm.globals.config[attr[len('config.'):]] or None
                else: return lazyllm.globals[attr] or None
        except Exception: pass
        return None

    def _resolve_one_key(self, source: Union[str, Callable]) -> Optional[str]:
        if callable(source): return source(getattr(self, '_instance', None)) or None
        return self._resolve_key_by_string(source)

    def should_skip(self) -> bool:
        if self._key_source is None: return False
        sources = self._key_source if isinstance(self._key_source, list) else [self._key_source]
        return not any(bool(self._resolve_one_key(src)) for src in sources)

    def source_is_active(self, source: Union[str, Callable, List[Union[str, Callable]], None]) -> bool:
        '''Resolve an arbitrary dynamic source with the same semantics as key_source.'''
        if source is None:
            return False
        raw_sources = source if isinstance(source, list) else [source]
        sources = [self._normalize_source(item) for item in raw_sources]
        return any(bool(self._resolve_one_key(item)) for item in sources)


class InstanceToolGroup(SkipMixin, ToolGroup):
    def __init__(self, instance: Any,
                 key_source: Union[str, Callable, List[Union[str, Callable]], None] = None):
        if key_source is None: key_source = getattr(type(instance), '__key_source__', None)
        self._instance = instance
        SkipMixin.__init__(self, key_source)
        lazy_source = getattr(type(instance), '__lazy_source__', None)
        lazy = self.source_is_active(lazy_source) if lazy_source is not None else True
        public_apis = getattr(instance, '__tool_public_apis__', instance.__public_apis__)
        schema_overrides = getattr(instance, '__tool_schema_overrides__', {})
        input_adapters = getattr(instance, '__tool_input_adapters__', {})
        tools = [
            MethodModuleTool(
                instance,
                method_name,
                schema_func=schema_overrides.get(method_name),
                input_adapter=input_adapters.get(method_name),
            )
            for method_name in public_apis
        ]
        name = instance.__class__.__name__
        desc = getattr(type(instance), '__doc__', '') or ''
        auto_activate = getattr(type(instance), '__tool_auto_activate__', None)
        ToolGroup.__init__(self, tools=tools, name=name, desc=desc, lazy=lazy, prefix=False,
                           auto_activate=auto_activate)

    @property
    def _tools(self) -> Dict[str, 'ModuleTool']:
        return {child.name: child for child in self._children if isinstance(child, ModuleTool)}

    def get_description(self, active_groups: Optional[Set[str]] = None) -> List[Dict]:
        if self.should_skip(): return []
        return super().get_description(active_groups)


class ToolGroupWrapper(SkipMixin, ToolContainer):
    def __init__(self, tool: Any, key_source: Union[str, Callable, List[Union[str, Callable]], None]):
        SkipMixin.__init__(self, key_source)
        self._inner = _build_tool_from_element(tool)

    def get_flat_tools(self) -> Dict[str, 'ModuleTool']:
        return self._inner.get_flat_tools() if isinstance(self._inner, ToolContainer) \
            else {self._inner.name: self._inner}

    def get_leaf_names(self) -> List[str]:
        return self._inner.get_leaf_names() if isinstance(self._inner, ToolContainer) \
            else [self._inner.name]

    def get_description(self, active_groups: Optional[Set[str]] = None) -> List[Dict]:
        if self.should_skip(): return []
        return self._inner.get_description(active_groups) if isinstance(self._inner, ToolContainer) \
            else [_build_tool_desc(self._inner)]

    def get_auto_active_groups(self, text: str) -> Set[str]:
        if self.should_skip() or not isinstance(self._inner, ToolContainer): return set()
        return self._inner.get_auto_active_groups(text)

    def get_activation_path(self, group_name: str) -> Set[str]:
        if self.should_skip() or not isinstance(self._inner, ToolContainer): return set()
        return self._inner.get_activation_path(group_name)


def _child_is_valid(child) -> bool:
    if isinstance(child, SkipMixin):
        return not child.should_skip()
    return True


TOOL_CALL_FORMAT_EXAMPLE = (
    '{"function": {"name": "tool_name", "arguments": '
    '"{{"arg1": "value1", "arg2": "value2"}}"}}'
)


def _build_tool_from_element(
        element: Any, _outer_prefix: Optional[str] = None) -> Optional[Union['ModuleTool', 'ToolGroup']]:
    if isinstance(element, str):
        return _load_tool_by_name(element)
    if isinstance(element, (ToolGroup, ModuleTool)):
        return element
    if isinstance(element, (tuple, list)) and len(element) == 2:
        tool, key_source = element
        if isinstance(tool, dict) and 'key_source' in tool:
            raise ValueError(
                f'key_source is specified both in the dict (dict["key_source"]={tool["key_source"]!r}) '
                f'and as the second element of the tuple ({key_source!r}). Provide key_source in only one place.')
        if hasattr(tool, '__public_apis__'):
            return InstanceToolGroup(tool, key_source)
        return ToolGroupWrapper(tool, key_source)
    if hasattr(element, '__public_apis__') and not isinstance(element, (ToolGroup, ModuleTool)):
        return InstanceToolGroup(element)
    if isinstance(element, dict):
        assert 'name' in element, "ToolGroup dict must have a 'name' field"
        assert 'tools' in element, "ToolGroup dict must have a 'tools' field"
        assert 'desc' in element, "ToolGroup dict must have a 'desc' field"
        key_source = element.get('key_source', None)
        pick_first_valid = element.get('pick_first_valid', False)
        group = ToolGroup(tools=element['tools'], name=element['name'], desc=element['desc'],
                          lazy=element.get('lazy', True), prefix=element.get('prefix', None),
                          _outer_prefix=_outer_prefix, pick_first_valid=pick_first_valid,
                          auto_activate=element.get('auto_activate'))
        if key_source is not None:
            return ToolGroupWrapper(group, key_source)
        return group
    if callable(element):
        schema_func = element
        register('tmp_tool')(element)
        try:
            # The registry wraps callables in a generated method whose module globals
            # differ from the callable's defining module. Keep the original callable
            # as the schema source so forward references and structured models resolve.
            return lazyllm.tmp_tool.resolve(element.__name__)(schema_func=schema_func)
        finally:
            lazyllm.tmp_tool.remove(element.__name__)
    raise TypeError(f'ToolGroup child must be a ModuleTool, ToolGroup, dict, or callable, got {type(element)}')


def _load_tool_by_name(name: str) -> 'ModuleTool':
    name = name.strip()
    if '.' not in name: return getattr(lazyllm.tool, name)()
    target = lazyllm
    for part in name.split('.'):
        if not part: raise ValueError(f'invalid tool name: {name}')
        target = getattr(target, part)
    return target()


class ToolManager(ModuleBase):

    def __init__(self, tools: List[Union[str, Callable]], return_trace: bool = False, sandbox=None):
        super().__init__(return_trace=return_trace)
        self._tools = [_build_tool_from_element(element) for element in tools]
        self._format_tools()
        self._tools_desc = self._transform_to_openai_function()
        self._sandbox = sandbox

    @property
    def all_tools(self) -> List[ModuleTool]:
        return list(self._tool_call.values())

    @property
    def tools_description(self) -> List[Dict]:
        try:
            workspace = lazyllm_locals['_lazyllm_agent'].get('workspace', {})
        except Exception:
            workspace = {}
        active_groups = set(workspace.get('_active_groups', []))
        return [x for item in self._tools_desc
                for x in (item.get_description(active_groups=active_groups)
                          if isinstance(item, ToolContainer) else item() if callable(item) else [item])]

    @property
    def tools_info(self):
        return self._tool_call

    @staticmethod
    def normalize_tool_calls(tool_calls: Any) -> Any:
        '''Normalize provider tool-call arguments at the ToolManager boundary.'''
        for tool_call in tool_calls if isinstance(tool_calls, list) else [tool_calls]:
            if isinstance(tool_call, dict) and not str(tool_call.get('id') or '').strip():
                tool_call['id'] = str(uuid.uuid4())
            function = tool_call.get('function') if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get('arguments')
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                function['arguments'] = '{}'
            elif isinstance(arguments, dict):
                function['arguments'] = std_json.dumps(arguments, ensure_ascii=False)
        return tool_calls

    def sync_active_groups(self, input: Any = None, history: Optional[List[Dict[str, Any]]] = None) -> Set[str]:  # noqa C901
        '''Activate lazy Toolkits from registered input rules and structured gateway calls in history.'''
        try:
            workspace = lazyllm_locals['_lazyllm_agent'].setdefault('workspace', {})
        except Exception:
            return set()
        if history is not None:
            workspace.setdefault('history', list(history))
        active = set(workspace.get('_active_groups', []))
        text = input if isinstance(input, str) else str(input.get('content', input.get('input', ''))) \
            if isinstance(input, dict) else ''
        if text:
            for item in self._tools:
                if isinstance(item, ToolContainer):
                    active.update(item.get_auto_active_groups(text))
        gateway_prefix, gateway_suffix = 'get_', '_methods'
        for message in history or []:
            if not isinstance(message, dict): continue
            for tool_call in message.get('tool_calls') or []:
                function = tool_call.get('function') if isinstance(tool_call, dict) else None
                name = function.get('name', '') if isinstance(function, dict) else ''
                if name.startswith(gateway_prefix) and name.endswith(gateway_suffix) \
                        and name in self._tool_call:
                    group_name = name[len(gateway_prefix):-len(gateway_suffix)]
                    for item in self._tools:
                        if isinstance(item, ToolContainer):
                            active.update(item.get_activation_path(group_name))
        workspace['_active_groups'] = list(active)
        return active

    @property
    def sandbox(self):
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox):
        self._sandbox = sandbox

    def _validate_tool(self, tool_name: str, tool_arguments: Dict[str, Any]):
        entry = self._tool_call.get(tool_name)
        if not entry:
            LOG.error(f'cannot find tool named [{tool_name}]')
            return None, None
        try:
            return entry._validate_input(tool_arguments), None
        except ValidationError as error:
            summaries = []
            for item in error.errors(include_url=False, include_context=True, include_input=True):
                path = tuple(item.get('loc') or ())
                path_text = '.'.join(str(part) for part in path) or '$'
                summaries.append(f'{path_text}: {item.get("msg") or "Invalid value"}')
            message = 'Invalid arguments: ' + '; '.join(summaries)
            return None, tool_failure(message)

    def _format_tools(self):
        if isinstance(self._tools, List):
            self._tool_call: Dict[str, ModuleTool] = {}
            for item in self._tools:
                items = item.get_flat_tools() if isinstance(item, ToolContainer) else {item.name: item}
                for name, tool in items.items():
                    if name in self._tool_call:
                        raise ValueError(f'Duplicate tool name [{name}]. Tool names must be unique.')
                    self._tool_call[name] = tool

    def _transform_to_openai_function(self):
        if not isinstance(self._tools, List):
            raise TypeError(f'The tools type should be List instead of {type(self._tools)}')

        format_tools = []
        for item in self._tools:
            if isinstance(item, ToolContainer):
                format_tools.append(item)
            else:
                try:
                    format_tools.append(_build_tool_desc(item))
                except Exception:
                    self._raise_format_error()
        return format_tools

    @staticmethod
    def _raise_format_error():
        typehints_template = '''
                def myfunc(arg1: str, arg2: Dict[str, Any], arg3: Literal['aaa', 'bbb', 'ccc']='aaa'):
                    """
                    Function description ...

                    Args:
                        arg1 (str): arg1 description.
                        arg2 (Dict[str, Any]): arg2 description
                        arg3 (Literal['aaa', 'bbb', 'ccc']): arg3 description
                    """
                '''
        raise TypeError('Function description must include function description and '
                        f'parameter description, the format is as follows: {typehints_template}')

    @staticmethod
    def _ensure_list(value):
        if isinstance(value, str):
            return [value]
        return value if value else []

    def _build_sandbox_args(self, tool, arguments):
        input_files = self._ensure_list(arguments.get(tool.input_files_parm, []))
        output_files = self._ensure_list(arguments.get(tool.output_files_parm, [])) + tool.output_files
        return kwargs(code=tool.to_sandbox_code(arguments), input_files=input_files, output_files=output_files)

    @staticmethod
    def _safe_parse_json(raw):
        import re as _re
        try:
            return json.loads(raw)
        except Exception:
            pass
        cleaned = _re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        # last resort: close truncated JSON by counting unmatched braces
        s = cleaned.strip()
        opens = s.count('{') - s.count('}')
        opens_sq = s.count('[') - s.count(']')
        s = _re.sub(r'[,:\"\'\w\s]*$', '', s.rstrip())
        s = s + (']' * max(0, opens_sq)) + ('}' * max(0, opens))
        return json.loads(s)

    def _parse_tool_call(self, tc, allowed_tool_names=None):
        func = tc.get('function') if isinstance(tc, dict) else None
        name = func.get('name') if isinstance(func, dict) else None
        if not isinstance(func, dict) or not isinstance(name, str) or not name.strip() \
                or 'arguments' not in func:
            failure = tool_failure(
                f'Tool call format is invalid, expected: {TOOL_CALL_FORMAT_EXAMPLE}',
            )
            return None, None, None, failure
        name = name.strip()
        raw_args = func['arguments']
        visible_names = list(self._tool_call) if allowed_tool_names is None else sorted(
            item for item in allowed_tool_names if item
        )
        if name not in visible_names:
            suggested_tool = _closest_tool_name(name, visible_names)
            message = f'Tool [{name}] was not exposed in this turn.'
            if suggested_tool:
                message += f' Did you mean [{suggested_tool}]? Call it explicitly.'
            failure = tool_failure(message)
            lazyllm.LOG.warning(
                f'[ToolManager] tool {name!r} was not exposed. Visible: {visible_names}'
            )
            return None, raw_args, None, failure
        try:
            arguments = ToolManager._safe_parse_json(raw_args) if isinstance(raw_args, str) else raw_args
        except (TypeError, ValueError):
            failure = tool_failure(
                'Invalid arguments: arguments must be valid JSON.',
            )
            return None, raw_args.strip() if isinstance(raw_args, str) else raw_args, None, failure
        if not isinstance(arguments, dict):
            failure = tool_failure(
                f'Invalid arguments: Tool [{name}] arguments must be a JSON object, '
                f'got {type(arguments).__name__}.',
            )
            return None, arguments, None, failure
        tool = self._tool_call.get(name)
        if tool is None:
            failure = tool_failure(
                f'Tool [{name}] is no longer registered.',
            )
            return None, arguments, None, failure
        validated_arguments, validation_failure = self._validate_tool(name, arguments)
        if validated_arguments is None:
            return None, arguments, None, validation_failure
        return tool, arguments, validated_arguments, None

    @staticmethod
    def _call_tool(tool, arguments, exposed_name):
        try:
            value = tool(_ValidatedToolInput(arguments))
            return {'ok': True, 'value': value}
        except Exception as error:
            lazyllm.LOG.warning(
                f'[ToolCall] tool={exposed_name!r} raised: {type(error).__name__}: {error}')
            return exception_failure(exposed_name, error)

    def _prepare_tool_invocations(self, tools, allowed_tool_names=None):
        tool_calls = self.normalize_tool_calls([tools] if isinstance(tools, dict) else list(tools or []))
        invocations = []
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get('function') if isinstance(tool_call, dict) else None
            tool_name = str(function.get('name') or '') if isinstance(function, dict) else ''
            call_id = str(tool_call.get('id') or '') if isinstance(tool_call, dict) else ''
            tool, normalized_arguments, validated_arguments, failure = self._parse_tool_call(
                tool_call, allowed_tool_names,
            )
            if tool is None:
                view = PreparedToolCall(
                    index=index,
                    tool_call=copy.deepcopy(tool_call),
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=copy.deepcopy(normalized_arguments),
                    validated_arguments=None,
                )
                invocations.append(_PreparedToolInvocation(
                    prepared=view,
                    tool=None,
                    validated_arguments=None,
                    failure=failure,
                ))
                continue
            access = tool._resolve_runtime_access(validated_arguments)
            view = PreparedToolCall(
                index=index,
                tool_call=copy.deepcopy(tool_call),
                call_id=call_id,
                tool_name=tool_name,
                arguments=copy.deepcopy(normalized_arguments),
                validated_arguments=copy.deepcopy(validated_arguments),
                access=access,
                polling=tool.runtime_metadata.polling,
            )
            invocations.append(_PreparedToolInvocation(
                prepared=view,
                tool=tool,
                validated_arguments=validated_arguments,
            ))
        return invocations

    def _build_prepared_invocation(self, invocation: _PreparedToolInvocation):
        prepared = invocation.prepared
        if not prepared.ready:
            return (
                lambda *_, _failure=invocation.failure: _failure,
                {},
                prepared.access,
                ToolExecutionDisposition.PREPARATION_FAILED,
            )
        tool = invocation.tool
        if self._sandbox and tool.execute_in_sandbox:
            sandbox_arguments = self._build_sandbox_args(tool, invocation.validated_arguments)
            return (
                self._sandbox,
                sandbox_arguments,
                prepared.access,
                ToolExecutionDisposition.EXECUTED,
            )

        def _safe_call(args):
            return self._call_tool(tool, args, prepared.tool_name)

        return (
            _safe_call,
            invocation.validated_arguments,
            prepared.access,
            ToolExecutionDisposition.EXECUTED,
        )

    @staticmethod
    def _invoke_tool_callable(callable_, arguments):
        if isinstance(arguments, kwargs):
            return callable_(**arguments)
        return callable_(arguments)

    @staticmethod
    def _build_execution_segments(accesses):
        segments = []
        current_segment = []
        reserved = ResolvedToolAccess()

        def close_current_segment():
            nonlocal current_segment, reserved
            if current_segment:
                segments.append(current_segment)
                current_segment = []
                reserved = ResolvedToolAccess()

        for index, access in enumerate(accesses):
            if access.exclusive:
                close_current_segment()
                segments.append([index])
                continue
            if _accesses_conflict(access, reserved):
                close_current_segment()
            current_segment.append(index)
            reserved = ResolvedToolAccess(
                read_keys=reserved.read_keys | access.read_keys,
                write_keys=reserved.write_keys | access.write_keys,
            )
        close_current_segment()
        return segments

    @classmethod
    def _execute_segment(cls, indices, callables, call_arguments):
        branch_callables = []
        branch_arguments = []
        for index in indices:
            def _indexed_call(_, _index=index, _callable=callables[index],
                              _arguments=call_arguments[index]):
                try:
                    return _index, cls._invoke_tool_callable(_callable, _arguments), None
                except Exception as error:
                    flow_error = error if isinstance(error, FlowException) \
                        else _change_exception_type(error, FlowException)
                    return _index, None, flow_error

            branch_callables.append(_indexed_call)
            branch_arguments.append({})
        tool_diverter = lazyllm.diverter(tuple(branch_callables))
        return tool_diverter(tuple(branch_arguments))

    @classmethod
    def _execute_tool_calls(cls, count, callables, call_arguments, accesses):
        ordered_results = [None] * count
        errors = []
        for segment in cls._build_execution_segments(accesses):
            for index, result, error in cls._execute_segment(segment, callables, call_arguments):
                if error is not None:
                    errors.append((index, error))
                else:
                    ordered_results[index] = result
        if errors:
            raise min(errors, key=lambda item: item[0])[1]
        return lazyllm.package(ordered_results)

    def execute_with_records(self, tools: Union[Dict[str, Any], List[Dict[str, Any]]],
                             verbose: bool = False,
                             allowed_tool_names: Optional[Set[str]] = None,
                             *, dispatch_selector: Optional[Callable] = None):
        '''Prepare and execute tool calls once, returning results with execution records.'''
        del verbose
        invocations = self._prepare_tool_invocations(tools, allowed_tool_names)
        if not invocations:
            return ToolExecutionBatch(results=[], records=())
        prepared = tuple(invocation.prepared for invocation in invocations)
        selected_indices = tuple(
            dispatch_selector(prepared)
            if dispatch_selector is not None else range(len(invocations))
        )
        if any(type(index) is not int or index < 0 or index >= len(invocations)
               for index in selected_indices):
            raise IndexError('selected prepared-call index is out of range')
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError('selected prepared-call indices must be unique')
        selected = [invocations[index] for index in sorted(selected_indices)]
        if not selected:
            return ToolExecutionBatch(results=[], records=())
        execution_inputs = [self._build_prepared_invocation(item) for item in selected]
        callables, call_arguments, accesses, dispositions = map(list, zip(*execution_inputs))
        results = self._execute_tool_calls(
            len(selected), callables, call_arguments, accesses,
        )
        records = tuple(
            ToolExecutionRecord(item.prepared, result, disposition)
            for item, result, disposition in zip(selected, results, dispositions)
        )
        return ToolExecutionBatch(results=results, records=records)

    def forward(self, tools: Union[Dict[str, Any], List[Dict[str, Any]]], verbose: bool = False,
                allowed_tool_names: Optional[Set[str]] = None):
        return self.execute_with_records(
            tools,
            verbose=verbose,
            allowed_tool_names=allowed_tool_names,
        ).results
