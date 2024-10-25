import copy
import json5 as json
import lazyllm
import docstring_parser
from lazyllm.module import ModuleBase
from lazyllm.common import LazyLLMRegisterMetaClass, compile_func
from typing import Callable, Any, Union, get_type_hints, List, Dict, Type, Set
import inspect
from pydantic import create_model, BaseModel, ValidationError
from lazyllm import LOG
from typing import *  # noqa F403, to import all types for compile_func(), do not remove
import time

# ---------------------------------------------------------------------------- #

def _gen_empty_func_str_from_parsed_docstring(parsed_docstring):
    """
    returns a function prototype string
    """

    func_name = "f" + str(int(time.time()))
    s = "def " + func_name + "("
    for param in parsed_docstring.params:
        s += param.arg_name
        if param.type_name:
            s += ":" + param.type_name + ","
        else:
            s += ","
    s += ")"

    if parsed_docstring.returns and parsed_docstring.returns.type_name:
        s += '->' + parsed_docstring.returns.type_name
    s += ":\n    pass"

    return s

def _gen_func_from_str(func_str, orig_docstring, global_env=None):
    if not global_env:
        global_env = globals()
    f = compile_func(func_str, global_env)
    f.__doc__ = orig_docstring
    return f

def _gen_wrapped_moduletool(func):
    if "tmp_tool" not in LazyLLMRegisterMetaClass.all_clses:
        register.new_group('tmp_tool')
    register('tmp_tool')(func)
    wrapped_module = getattr(lazyllm.tmp_tool, func.__name__)()
    lazyllm.tmp_tool.remove(func.__name__)
    return wrapped_module

def _gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring):
    """
    returns a dict of param names containing at least
      1. `type`
      2. `description` of params

    for example:
        args = {
            "foo": {
                "enum": ["baz", "bar"],
                "type": "string",
                "description": "a string",
            },
            "bar": {
                "type": "integer",
                "description": "an integer",
            }
        }
    """
    tool_args = tool.args
    assert len(tool_args) == len(parsed_docstring.params), ("The parameter description and the actual "
                                                            "number of input parameters are inconsistent.")

    args_description = {}
    for param in parsed_docstring.params:
        args_description[param.arg_name] = param.description

    args = {}
    for k, v in tool_args.items():
        val = copy.deepcopy(v)
        val.pop("title", None)
        val.pop("default", None)
        args[k] = val if val else {"type": "string"}
        desc = args_description.get(k, None)
        if desc:
            args[k].update({"description": desc})
        else:
            raise ValueError(f"The actual input parameter {k} is not found "
                             "in the parameter description.")
    return args

# ---------------------------------------------------------------------------- #

class ModuleTool(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, verbose: bool = False, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._verbose = verbose
        self._name = self.apply.__name__\
            if hasattr(self.apply, "__name__") and self.apply.__name__ is not None\
            else (_ for _ in ()).throw(ValueError("Function must have a name."))
        self._description = self.apply.__doc__\
            if hasattr(self.apply, "__doc__") and self.apply.__doc__ is not None\
            else (_ for _ in ()).throw(ValueError("Function must have a docstring"))

        parsed_docstring = docstring_parser.parse(self._description)
        func_str_from_doc = _gen_empty_func_str_from_parsed_docstring(parsed_docstring)
        func_from_doc = _gen_func_from_str(func_str_from_doc, self._description)
        func_from_doc.__name__ = self._name
        self._params_schema = self._load_function_schema(self.__class__.apply, func_from_doc)

    def _load_function_schema(self, orig_func: Callable, func_from_doc: Callable) -> Type[BaseModel]:
        orig_type_hints = get_type_hints(orig_func, globals(), locals())
        doc_type_hints = get_type_hints(func_from_doc, globals(), locals())
        self._type_hints = doc_type_hints | orig_type_hints
        self._return_type = self._type_hints.get('return') if self._type_hints else None

        signature = inspect.signature(func_from_doc)
        fields = {
            name: (self._type_hints.get(name, Any), param.default
                   if param.default is not inspect.Parameter.empty
                   else ...)
            for name, param in signature.parameters.items()
        }

        return create_model(self._name, **fields)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def params_schema(self) -> Type[BaseModel]:
        return self._params_schema

    @property
    def args(self) -> Dict[str, Any]:
        return self._params_schema.model_json_schema()["properties"]

    @property
    def required_args(self) -> Set[str]:
        return set(self._params_schema.model_json_schema()["required"])

    def get_params_schema(self) -> [BaseModel]:
        return self._params_schema

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Implement apply function in subclass")

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
                key_type = self._type_hints.get(key)
                if not key_type or not isinstance(key_type, type):
                    raise TypeError(f'"{key_type} is not a type"')
                input_params.model_validate({key: tool_input})
                return {key: key_type(tool_input)}

            types = self._type_hints.values()
            if len(types) == 0:
                return tool_input
            return types[0](tool_input)
        else:
            raise TypeError(f"tool_input {tool_input} only supports dict and str.")

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
            lazyllm.LOG.debug(f"The output of tool {self.name} is {ret}")

        return ret

register = lazyllm.Register(ModuleTool, ["apply"])
if "tool" not in LazyLLMRegisterMetaClass.all_clses:
    register.new_group("tool")

class ToolManager(ModuleBase):
    def __init__(self, tools: List[Union[str, Callable]], return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._tools = self._load_tools(tools)
        self._format_tools()
        self._tools_desc = self._transform_to_openai_function()

    def _load_tools(self, tools: List[Union[str, Callable]]):
        if "tmp_tool" not in LazyLLMRegisterMetaClass.all_clses:
            register.new_group('tmp_tool')

        _tools = []
        for element in tools:
            if isinstance(element, str):
                _tools.append(getattr(lazyllm.tool, element)())
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
    def _check_return_info_is_the_same(func, tool) -> bool:
        type_hints = get_type_hints(func, globals(), locals())
        return_type = type_hints.get('return') if type_hints else None
        return return_type == tool._return_type

    def _transform_to_openai_function(self):
        if not isinstance(self._tools, List):
            raise TypeError(f"The tools type should be List instead of {type(self._tools)}")

        format_tools = []
        for tool in self._tools:
            try:
                parsed_docstring = docstring_parser.parse(tool.description)
                args = _gen_args_info_from_moduletool_and_docstring(tool, parsed_docstring)
                required_arg_list = tool.get_params_schema().model_json_schema().get("required", [])

                # check return values
                func_str_from_doc = _gen_empty_func_str_from_parsed_docstring(parsed_docstring)
                func_from_doc = _gen_func_from_str(func_str_from_doc, tool.description)
                if not self._check_return_info_is_the_same(func_from_doc, tool):
                    raise ValueError("return info in docstring is different from that in function prototype.")

                func = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": parsed_docstring.short_description,
                        "parameters": {
                            "type": "object",
                            "properties": args,
                            "required": required_arg_list,
                        }
                    }
                }
                format_tools.append(func)
            except Exception:
                typehints_template = """
                def myfunc(arg1: str, arg2: Dict[str, Any], arg3: Literal["aaa", "bbb", "ccc"]="aaa"):
                    '''
                    Function description ...

                    Args:
                        arg1 (str): arg1 description.
                        arg2 (Dict[str, Any]): arg2 description
                        arg3 (Literal["aaa", "bbb", "ccc"]): arg3 description
                    '''
                """
                raise TypeError("Function description must include function description and "
                                f"parameter description, the format is as follows: {typehints_template}")
        return format_tools

    def forward(self, tools: Union[Dict[str, Any], List[Dict[str, Any]]], verbose: bool = False):
        tool_calls = [tools,] if isinstance(tools, dict) else tools
        tool_calls = [{"name": tool['name'], "arguments": json.loads(tool['arguments'])
                      if isinstance(tool['arguments'], str) else tool['arguments']} for tool in tool_calls]
        output = []
        flag_val = [True if self._validate_tool(tool['name'], tool['arguments']) else False for tool in tool_calls]
        tool_inputs = [tool_calls[idx]['arguments'] for idx, val in enumerate(flag_val) if val]
        tools = [self._tool_call[tool_calls[idx]['name']] for idx, val in enumerate(flag_val) if val]
        tool_diverter = lazyllm.diverter(tuple(tools))
        rets = tool_diverter(tuple(tool_inputs))
        res = iter(rets)
        rets = [next(res) if ele else None for ele in flag_val]
        for idx, tool in enumerate(tool_calls):
            if flag_val[idx]:
                ret = rets[idx]
                output.append(json.dumps(ret, ensure_ascii=False) if not isinstance(ret, str) else ret)
            else:
                output.append(f"{tool} parameters error.")

        return output
