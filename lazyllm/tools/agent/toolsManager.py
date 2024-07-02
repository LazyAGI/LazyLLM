from __future__ import annotations

import lazyllm
import docstring_parser
from lazyllm.module import ModuleBase
from lazyllm.common import LazyLLMRegisterMetaClass
from typing import Callable, Any, Union, get_type_hints, List, Dict, Type, Tuple, Set
import inspect
from pydantic import validate_call, create_model, BaseModel, ValidationError


class ModuleTool(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, verbose: bool = False, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._verbose = verbose
        self._name = self.apply.__name__ if hasattr(self.apply, "__name__") and self.apply.__name__ is not None \
            else (_ for _ in ()).throw(ValueError("Function must has a name."))
        self._description = self.apply.__doc__ if hasattr(self.apply, "__doc__") and self.apply.__doc__ is not None \
            else (_ for _ in ()).throw(ValueError("Function must has a docstring"))

        self._params_schema = None
        if self._params_schema is None:
            self._params_schema = self.load_function_schema(getattr(type(self), "apply"))

    def load_function_schema(self, func: Callable) -> Type[BaseModel]:
        if func.__name__ is None or func.__doc__ is None:
            raise ValueError(f"Function {func} must have a name and docstring.")
        self._name = func.__name__
        self._description = func.__doc__
        func = validate_call(func)
        signature = inspect.signature(func)
        type_hints = get_type_hints(func, globals(), locals())

        fields = {
            name: (type_hints.get(name, Any), param.default if param.default is not inspect.Parameter.empty else ...)
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
        if self._params_schema is None:
            self._params_schema = self.load_function_schema(getattr(type(self), "apply"))
        return self._params_schema.model_json_schema()["properties"]

    @property
    def required_args(self) -> Set[str]:
        if self._params_schema is None:
            self._params_schema = self.load_function_schema(getattr(type(self), "apply"))
        return set(self._params_schema.model_json_schema()["required"])

    def get_params_schema(self) -> [BaseModel]:
        if self._params_schema is None:
            self._params_schema = self.load_function_schema(getattr(type(self), "apply"))
        return self._params_schema

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Implement apply function in subclass")

    def _validate_input(self, tool_input: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        input_params = self._params_schema
        if isinstance(tool_input, str):
            if input_params is not None:
                key_ = next(iter(input_params.model_fields.keys()))
                try:
                    input_params.model_validate({key_: tool_input})
                    return tool_input
                except ValidationError as e:
                    lazyllm.LOG.error(f"ValidationError: {e}")
                    # raise TypeError(f"The tool_input: {tool_input} type error.")
                    raise
        else:
            if input_params is not None:
                try:
                    ret = input_params.model_validate(tool_input)
                    return {key: getattr(ret, key) for key in ret.model_dump().keys() if key in tool_input}
                except ValidationError as e:
                    lazyllm.LOG.error(f"ValidationError: {e}")
                    # raise TypeError(f"The tool_input: {tool_input} type error.")
                    raise

        return tool_input

    def validate_parameters(self, arguments: Dict[str, Any]) -> bool:
        sz = len(self.required_args.difference(set(arguments.keys())))
        if sz == 0:
            # contains all required parameters
            try:
                self._validate_input(arguments)
                return True
            except ValidationError:
                return False
        else:
            return False

    def _arguments_spliter(self, tool_input: Union[str, Dict[str, Any]]) -> Tuple[Tuple, Dict[str, Any]]:
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def forward(self, tool_input: Union[str, Dict[str, Any]], verbose: bool = False) -> Any:
        if not self._verbose and verbose:
            verbose_ = verbose
        else:
            verbose_ = self._verbose

        val_input = self._validate_input(tool_input)
        tool_args, tool_kw = self._arguments_spliter(val_input)
        ret = self.apply(*tool_args, **tool_kw)
        if verbose_:
            lazyllm.LOG.info(f"The output of tool {self.name} is {ret}")

        return ret

register = lazyllm.Register(ModuleTool, ["apply"])
register.new_group("functionCall")

class ToolManager(ModuleBase):
    def __init__(self, tools: List[str], return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._tools = self._load_tools(tools)
        self._format_tools()
        self._tools_desc = self._transform_to_openai_function()

    def _load_tools(self, tools_str: List[str]):
        _tools = []
        for tool_str in tools_str:
            tool_all_str = tool_str + "functioncall".capitalize()
            t = lazyllm.functionCall.get(tool_all_str, None)
            if t:
                _tools.append(t())
            else:
                raise ValueError(f"Tool {tool_str} has not been registered yet.")
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

    def _format_tools(self):
        if isinstance(self._tools, List):
            self._tool_call = {tool.name: tool for tool in self._tools}

    def _transform_to_openai_function(self):
        if isinstance(self._tools, List):
            format_tools = []
            for tool in self._tools:
                parsed = docstring_parser.parse(tool.description)
                tool_args = tool.args
                assert len(tool_args) == len(parsed.params), "The parameter description and the actual \
                                                              number of input parameters are inconsistent."
                args_description = {}
                for param in parsed.params:
                    args_description[param.arg_name] = param.description
                args = {}
                for k, v in tool_args.items():
                    val = v.copy()
                    if "title" in val.keys():
                        del val["title"]
                    if "default" in val.keys():
                        del val["default"]
                    args[k] = val if val else {"type": "string"}
                    if k in args_description:
                        args[k].update({"description": args_description[k]})
                    else:
                        raise ValueError(f"The actual input parameter {k} is not found in the parameter description.")
                func = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": parsed.short_description,
                        "parameters": {
                            "type": "object",
                            "properties": args,
                            "required": tool.get_params_schema().model_json_schema().get("required", [])
                        }
                    }
                }
                format_tools.append(func)
            return format_tools
        else:
            raise TypeError(f"The tools type should be List instead of {type(self._tools)}")

    def forward(self, tool_name: str, tool_input: Union[str, Dict[str, Any]], verbose: bool = False):
        if tool_name in self._tool_call:
            return self._tool_call[tool_name](tool_input, verbose)
        else:
            raise ValueError(f"There is no tool {tool_name} in the toolset.")
