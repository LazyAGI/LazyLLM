import copy
import json5 as json
import lazyllm
import docstring_parser
import os
from lazyllm.module import ModuleBase
from lazyllm.common import LazyLLMRegisterMetaClass, compile_func, kwargs
from typing import Callable, Any, Union, get_type_hints, List, Dict, Type, Set
import inspect
from pydantic import create_model, BaseModel, ValidationError
from lazyllm import LOG
from typing import *  # noqa F403, to import all types for compile_func(), do not remove
import time

# ---------------------------------------------------------------------------- #

def _gen_empty_func_str_from_parsed_docstring(parsed_docstring):
    '''
    returns a function prototype string
    '''

    func_name = 'f' + str(int(time.time()))
    s = 'def ' + func_name + '('
    for param in parsed_docstring.params:
        s += param.arg_name
        if param.type_name:
            s += ':' + param.type_name + ','
        else:
            s += ','
    s += ')'

    if parsed_docstring.returns and parsed_docstring.returns.type_name:
        s += '->' + parsed_docstring.returns.type_name
    s += ':\n    pass'

    return s

def _gen_func_from_str(func_str, orig_docstring, global_env=None):
    if not global_env:
        global_env = globals()
    f = compile_func(func_str, global_env)
    f.__doc__ = orig_docstring
    return f

def _check_return_type_is_the_same(doc_type_hints, func_type_hints) -> None:
    func_return_type = func_type_hints.get('return') if func_type_hints else None
    doc_return_type = doc_type_hints.get('return') if doc_type_hints else None
    if func_return_type is not None and doc_return_type is not None:
        if func_return_type != doc_return_type:
            raise TypeError('return info in docstring is different from that in function prototype.')

# ---------------------------------------------------------------------------- #

class ModuleTool(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, verbose: bool = False, return_trace: bool = True, execute_in_sandbox: bool = True):
        super().__init__(return_trace=return_trace)
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
        self._execute_in_sandbox = execute_in_sandbox
        self._input_files_parm = None
        self._output_files_parm = None
        self._output_files = []

        self._params_schema = self._load_function_schema(self.__class__.apply)

    def _load_function_schema(self, func: Callable) -> Type[BaseModel]:
        parsed_docstring = docstring_parser.parse(self._description)
        func_str_from_doc = _gen_empty_func_str_from_parsed_docstring(parsed_docstring)
        func_from_doc = _gen_func_from_str(func_str_from_doc, self._description)
        func_from_doc.__name__ = func.__name__
        doc_type_hints = get_type_hints(func_from_doc, globals(), locals())

        func_type_hints = get_type_hints(func, globals(), locals())

        _check_return_type_is_the_same(doc_type_hints, func_type_hints)

        signature = inspect.signature(func)
        has_var_args = False
        for _, param in signature.parameters.items():
            if param.kind == inspect.Parameter.VAR_POSITIONAL or\
               param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_args = True
                break

        if has_var_args:
            # we cannot get type hints from var args, so we get them from docstring
            self._type_hints = doc_type_hints
            signature = inspect.signature(func_from_doc)
        else:
            self._type_hints = func_type_hints
            # accomplish type_hints from docstring
            for name, type in doc_type_hints.items():
                self._type_hints.setdefault(name, type)

        self._return_type = self._type_hints.get('return') if self._type_hints else None

        fields = {
            name: (self._type_hints.get(name, Any), param.default
                   if param.default is not inspect.Parameter.empty
                   else ...)
            for name, param in signature.parameters.items()
            if name != 'self'
        }

        return create_model(self._name, **fields)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def execute_in_sandbox(self) -> bool:
        return self._execute_in_sandbox

    @execute_in_sandbox.setter
    def execute_in_sandbox(self, value: bool):
        self._execute_in_sandbox = value

    @property
    def input_files_parm(self) -> str:
        return self._input_files_parm

    @input_files_parm.setter
    def input_files_parm(self, value: str):
        assert isinstance(value, str), f'input_files_parm must be a string, but got {type(value)}'
        self._input_files_parm = value

    @property
    def output_files_parm(self) -> str:
        return self._output_files_parm

    @output_files_parm.setter
    def output_files_parm(self, value: str):
        assert isinstance(value, str), f'output_files_parm must be a string, but got {type(value)}'
        self._output_files_parm = value

    @property
    def output_files(self) -> List[str]:
        return self._output_files

    @output_files.setter
    def output_files(self, value: List[str]):
        assert isinstance(value, list) and all(isinstance(item, str) for item in value), \
            f'output_files must be a list of strings, but got {type(value)}'
        self._output_files = value

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

    def _validate_input(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
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

            if len(self._type_hints) != 1:
                return tool_input
            arg_type = self._type_hints.values()[0]
            return arg_type(tool_input)
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
        val_input = self._validate_input(tool_input)
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
result = tool(kwargs)
print(f'{SANDBOX_TOOL_RESULT_PREFIX}{{result}}')  # noqa print
'''

register = lazyllm.Register(ModuleTool, ['apply'], default_group='tool',
                            allowed_parameter=['execute_in_sandbox', 'input_files_parm',
                                               'output_files_parm', 'output_files'])
if 'tool' not in LazyLLMRegisterMetaClass.all_clses:
    register.new_group('tool')
if 'builtin_tools' not in LazyLLMRegisterMetaClass.all_clses:
    register.new_group('builtin_tools')

TOOL_CALL_FORMAT_EXAMPLE = (
    '{"function": {"name": "tool_name", "arguments": '
    '"{{"arg1": "value1", "arg2": "value2"}}"}}'
)


class ToolManager(ModuleBase):
    def __init__(self, tools: List[Union[str, Callable]], return_trace: bool = False, sandbox=None):
        super().__init__(return_trace=return_trace)
        self._tools = self._load_tools(tools)
        self._format_tools()
        self._tools_desc = self._transform_to_openai_function()
        self._sandbox = sandbox

    def _load_tools(self, tools: List[Union[str, Callable]]):
        if 'tmp_tool' not in LazyLLMRegisterMetaClass.all_clses:
            register.new_group('tmp_tool')

        _tools = []
        for element in tools:
            if isinstance(element, str):
                name = element.strip()
                if '.' in name:
                    target = lazyllm
                    for part in name.split('.'):
                        if not part:
                            raise ValueError(f'invalid tool name: {element}')
                        target = getattr(target, part)
                    _tools.append(target())
                else:
                    _tools.append(getattr(lazyllm.tool, name)())
            elif isinstance(element, ModuleTool):
                _tools.append(element)
            elif isinstance(element, Callable):
                # just to convert `element` to the internal type in `Register`
                register('tmp_tool')(element)
                _tools.append(getattr(lazyllm.tmp_tool, element.__name__)())
                lazyllm.tmp_tool.remove(element.__name__)

        return _tools

    @property
    def all_tools(self):
        return self._tools

    @property
    def tools_description(self):
        return self._tools_desc

    @property
    def tools_info(self):
        return self._tool_call

    @property
    def sandbox(self):
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox):
        self._sandbox = sandbox

    def _validate_tool(self, tool_name: str, tool_arguments: Dict[str, Any]):
        tool = self._tool_call.get(tool_name)
        if not tool:
            LOG.error(f'cannot find tool named [{tool_name}]')
            return False

        return tool.validate_parameters(tool_arguments)

    def _format_tools(self):
        if isinstance(self._tools, List):
            self._tool_call = {tool.name: tool for tool in self._tools}

    @staticmethod
    def _gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring):
        '''
        returns a dict of param names containing at least
          1. `type`
          2. `description` of params

        for example:
            args = {
                'foo': {
                    'enum': ['baz', 'bar'],
                    'type': 'string',
                    'description': 'a string',
                },
                'bar': {
                    'type': 'integer',
                    'description': 'an integer',
                }
            }
        '''
        tool_args = tool.args
        assert len(tool_args) == len(parsed_docstring.params), ('The parameter description and the actual '
                                                                'number of input parameters are inconsistent.')

        args_description = {}
        for param in parsed_docstring.params:
            args_description[param.arg_name] = param.description

        args = {}
        for k, v in tool_args.items():
            val = copy.deepcopy(v)
            val.pop('title', None)
            val.pop('default', None)
            args[k] = val if val else {'type': 'string'}
            desc = args_description.get(k, None)
            if desc:
                args[k].update({'description': desc})
            else:
                raise ValueError(f'The actual input parameter "{k}" is not found '
                                 f'in the parameter description of tool "{tool.name}".')
        return args

    def _transform_to_openai_function(self):
        if not isinstance(self._tools, List):
            raise TypeError(f'The tools type should be List instead of {type(self._tools)}')

        format_tools = []
        for tool in self._tools:
            try:
                parsed_docstring = docstring_parser.parse(tool.description)
                args = self._gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring)
                required_arg_list = tool.params_schema.model_json_schema().get('required', [])
                func = {
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': parsed_docstring.short_description,
                        'parameters': {
                            'type': 'object',
                            'properties': args,
                            'required': required_arg_list,
                        }
                    }
                }
                format_tools.append(func)
            except Exception:
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
        return format_tools

    @staticmethod
    def _ensure_list(value):
        if isinstance(value, str):
            return [value]
        return value if value else []

    def _build_sandbox_args(self, tool, arguments):
        input_files = self._ensure_list(arguments.get(tool.input_files_parm, []))
        output_files = self._ensure_list(arguments.get(tool.output_files_parm, [])) + tool.output_files
        return kwargs(code=tool.to_sandbox_code(arguments), input_files=input_files, output_files=output_files)

    def _parse_tool_call(self, tc):
        func = tc.get('function') if isinstance(tc, dict) else None
        if not func or 'name' not in func or 'arguments' not in func:
            return None, f'Tool call format is invalid, expected: {TOOL_CALL_FORMAT_EXAMPLE}'
        name = func['name']
        raw_args = func['arguments']
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(arguments, dict):
            return None, f'Tool [{name}] arguments format error.'
        tool = self._tool_call.get(name)
        if tool is None:
            return None, f'Tool [{name}] is not available. Please choose from the available tools.'
        if not self._validate_tool(name, arguments):
            return None, f'Tool [{name}] parameters error.'
        return tool, arguments

    def forward(self, tools: Union[Dict[str, Any], List[Dict[str, Any]]], verbose: bool = False):
        if not tools: return []
        tool_calls = [tools] if isinstance(tools, dict) else tools

        callables = []
        call_arguments = []
        for tc in tool_calls:
            tool, args_or_err = self._parse_tool_call(tc)
            if tool is None:
                callables.append(lambda *_, _e=args_or_err: _e)
                call_arguments.append({})
            elif self._sandbox and tool.execute_in_sandbox:
                callables.append(self._sandbox)
                call_arguments.append(self._build_sandbox_args(tool, args_or_err))
            else:
                callables.append(tool)
                call_arguments.append(args_or_err)

        tool_diverter = lazyllm.diverter(tuple(callables))
        return tool_diverter(tuple(call_arguments))
