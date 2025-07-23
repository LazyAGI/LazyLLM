# noqa: E121
from typing import Optional, List, Callable, Dict, Union
from dataclasses import dataclass


def extract_all_ids(data) -> list[str]:
    """Recursively extract all 'id' fields in any nested structure"""
    result = []

    if isinstance(data, (list, tuple)):
        result.extend([x for item in data for x in extract_all_ids(item)])
    else:
        result.append(data['id'] if isinstance(data, dict) else data)

    return result

@dataclass
class Node():
    """Node(id: str, kind: str, name: str, args: Optional[Dict] = None, func: Optional[Callable] = None, arg_names: Optional[List[str]] = None, enable_data_reflow: bool = False, subitem_name: Union[List[str], str, NoneType] = None, hyperparameter: Optional[Dict] = None)"""
    id: str
    kind: str
    name: str
    args: Optional[Dict] = None
    func: Optional[Callable] = None
    arg_names: Optional[List[str]] = None
    enable_data_reflow: bool = False
    subitem_name: Optional[Union[List[str], str]] = None
    hyperparameter: Optional[Dict] = None

    @property
    def subitems(self) -> List[str]:
        if not self.subitem_name: return []
        names = [self.subitem_name] if isinstance(self.subitem_name, str) else self.subitem_name
        result = []
        for name in names:
            name, tp = name.split(':') if ':' in name else (name, None)
            source = self.args.get(name, {} if tp == 'dict' else [])
            if tp != 'dict': source = dict(key=source)
            for s in source.values():
                result.extend(extract_all_ids(s))
        return result
