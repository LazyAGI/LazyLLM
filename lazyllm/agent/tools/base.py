from typing import  Any, Union
from abc import ABC, abstractmethod
try:
    import docstring_parser
except ImportError:
    raise ImportError("Please install docstring_parser using `pip install docstring_parser`.")
import re

class BaseTool(ABC):
    name: str

    @abstractmethod
    def call(*args, **kwages) -> Union[str, list, dict]:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwages: Any) -> Union[str, list, dict]:
        return self.call(*args, **kwages)

    @property
    def description(self):
        return transform_to_openai_function(self)


# ========== transform_to_openai_function start ============
def map_type(in_type:str):
    basic_type_map = {
        'str':'string',
        'float':'number',
        'list':'array',
        'tuple':'array',
        'dict':'object',
        'int':'integer',
        'bool':'boolean'
    }

    if in_type in basic_type_map:
        return {'type':basic_type_map[in_type]}
    
    match = re.search(r"\[(.*?)\]", in_type)
    sub_types = match.group(1).split(',') if match else ['string']

    if in_type.startswith('Literal'):
        return {'type':'string', 'enum':[x.replace('"','').replace("'",'').strip() for x in sub_types]}
    
    for i, sub_type in enumerate(sub_types):
        sub_type = sub_type.strip()
        if sub_type in basic_type_map:
            sub_type = basic_type_map[sub_type]
            sub_types[i] = sub_type
        if sub_type not in set(basic_type_map.values()):
            raise TypeError(f"The item {sub_type} form type {in_type} not basic type, it is not supported. ")

    if in_type.startswith(('List','Tuple')):
        return {'type':'array', 'items':{'type':sub_types[0]}}
    if in_type.startswith('Dict'):
        return {'type':'object'}
    raise TypeError(f"Type {in_type} not supported. ")
    

def get_args_without_defaults(func):
    num_args = func.__code__.co_argcount
    defaults = func.__defaults__ or ()
    args_without_defaults = func.__code__.co_varnames[:num_args - len(defaults)]
    required_args = list(args_without_defaults)
    if required_args[0] == "self":
        required_args = required_args[1:]
    return required_args

# 抽取 function 描述
def transform_to_openai_function(tool:BaseTool):
    parsed = docstring_parser.parse(tool.call.__doc__)

    # Extract descriptions, args
    description = parsed.short_description

    args = {}
    for param in parsed.params:
        args[param.arg_name] = {
            "description": param.description
        }
        args[param.arg_name].update(map_type(param.type_name))

    return {
        "type": "function",
        "function": {
            "name": tool.name if hasattr(tool, "name") else tool.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": args,
                "required": get_args_without_defaults(tool.call)
            },
        }
    }

# ========== transform_to_openai_function end ============