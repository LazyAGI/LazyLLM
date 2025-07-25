from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar, Generic, List, Dict, Union

T = TypeVar('T')

def convert_to_integer(source: str) -> int:
    return int(source)

def convert_to_string(source: str) -> str:
    return source

def convert_to_bool(source: str) -> bool:
    if source in ['TRUE', 'True', 'true', '1', 'ON', 'on']:
        return True
    if source in ['FALSE', 'False', 'false', '0', 'OFF', 'off']:
        return False
    raise ValueError(f"'{source}' is not a valid bool")

def linear_search_exactly(values: List[Any], value: Any) -> Union[int, None]:
    return values.index(value) if value in values else None

def binary_search_exactly(values: List[Any], value: Any) -> Union[int, None]:
    index = binary_search_equal_or_greater(values, value)
    if index is not None and value < values[index]:
        index = None
    return index

def binary_search_equal_or_less(values: List[Any], value: Any) -> Union[int, None]:
    count = len(values)
    index = 0
    while count > 0:
        step = count // 2
        if value < values[index + step]:
            count = step
        elif step > 0:
            index += step
            count -= step
        else:
            break
    return None if index == 0 and value < values[0] else index

def binary_search_equal_or_greater(values: List[Any], value: Any) -> Union[int, None]:
    count = len(values)
    index = 0
    while count > 0:
        step = count // 2
        if values[index + step] < value:
            index += step + 1
            count -= step + 1
        else:
            count = step
    return index if index < len(values) else None


class SearchMode(Enum):
    """An enumeration."""
    LINEAR_EXACTLY = 0
    BINARY_EXACTLY = auto()
    BINARY_FLOOR = auto()
    BINARY_CEIL = auto()

    def __call__(self, values: List[Any], value: Any) -> Union[int, None]:
        mapping = [
            linear_search_exactly,
            binary_search_exactly,
            binary_search_equal_or_less,
            binary_search_equal_or_greater]
        return mapping[self.value](values, value)


@dataclass
class Options(Generic[T]):
    """Options(options: List[~T], mapping: bool = False, indexed: bool = False, matches: collections.abc.Callable[[typing.List[typing.Any], typing.Any], typing.Optional[int]] = <SearchMode.LINEAR_EXACTLY: 0>)"""
    options: List[T]
    mapping: bool = False
    indexed: bool = False
    matches: Callable[[List[Any], Any], Union[int, None]] = SearchMode.LINEAR_EXACTLY


class Rule(Generic[T]):
    @classmethod
    def from_indexed(cls, name: str, options: List[T], matches: SearchMode = SearchMode.LINEAR_EXACTLY):
        return cls(name, options=Options(options=options, mapping=True, indexed=True, matches=matches))

    @classmethod
    def from_options(cls, name: str, options: List[T], mapping: Union[bool, None] = None,
                     matches: SearchMode = SearchMode.LINEAR_EXACTLY):
        mapping = all(isinstance(o, str) for o in options) if mapping is None else mapping
        return cls(name, options=Options(options=options, mapping=mapping, matches=matches))

    @classmethod
    def from_type(cls, name: str, value_type: Union[type[bool], type[int], type[str]]):
        return cls(name, convert=Rule.__infer_converter_from_type(value_type))

    def __init__(self, name: str, *, options: Union[Options[T], None] = None,
                 convert: Union[Callable[[str], T], None] = None):
        if len(name) == 0:
            raise ValueError("empty name is invalid")
        if options is None and convert is None:
            raise ValueError("either 'options' or 'convert' should be provided")
        if options is not None and len(options.options) == 0:
            raise ValueError("empty options is invalid")
        if (options is not None and options.matches
           in [SearchMode.BINARY_EXACTLY, SearchMode.BINARY_FLOOR, SearchMode.BINARY_CEIL] and not
           Rule.__is_ordered(options.options)):
            raise ValueError("cannot perform binary-search on unordered options")

        self._name = name
        self._convert = convert if convert is not None else Rule.__infer_converter_from_options(options)
        self._options = options

    @property
    def name(self) -> str:
        return self._name

    @property
    def value_type(self) -> type[T]:
        try:
            return self._convert.__annotations__["return"]
        except KeyError as e:
            raise ValueError(f"missing return-type annotation in {self.name} convert function") from e

    @property
    def options(self) -> Union[List[T], None]:
        return None if self._options is None else self._options.options

    @property
    def mapping(self) -> bool:
        return False if self._options is None else self._options.mapping

    @property
    def indexed(self) -> bool:
        return False if self._options is None else self._options.indexed

    @property
    def matches(self) -> Union[Callable[[List[Any], Any], Union[int, None]], None]:
        return None if self._options is None else self._options.matches

    def convert_string_to_value(self, source: str) -> T:
        try:
            value = self._convert(source)
        except ValueError as e:
            raise ValueError(f"while parse value as {self._name}") from e

        if self.options is not None and value not in self.options:
            raise ValueError(f"value '{value}' is not in {self._name} options {self.options}")
        return value

    def convert_value_to_index(self, value: T) -> List[int]:
        assert self.options is not None, f"index is unavailable if rule {self.name} has no options"
        index = None
        if self.matches is not None:
            index = self.matches(self.options, value)
        if index is None and value in self.options:
            index = self.options.index(value)
        if index is None:
            raise ValueError(f"value {value} is out of {self._name} options {self.options}")
        return index, len(self.options)

    def convert_index_to_value(self, index: int) -> T:
        assert self.options is not None, f"index is unavailable if rule {self.name} has no options"
        if not 0 <= index < len(self.options):
            raise ValueError(f"index {index} out of range [0, {len(self.options)})")
        return self.options[index]

    @staticmethod
    def __infer_converter_from_options(options: Options[T]) -> Callable[[str], T]:
        assert options is not None and len(options.options) != 0
        return Rule.__infer_converter_from_type(type(options.options[0]))

    @staticmethod
    def __infer_converter_from_type(tp: type) -> Callable[[str], T]:
        if issubclass(tp, bool): return convert_to_bool  # type: ignore
        elif issubclass(tp, int): return convert_to_integer  # type: ignore
        elif issubclass(tp, str): return convert_to_string  # type: ignore
        else: raise ValueError(f"unknown type {tp} of options")

    @staticmethod
    def __is_ordered(options: List[Any]) -> bool:
        def is_list_monotonic_increasing(ops) -> bool:
            for i in range(len(ops) - 1):
                if not (hasattr(ops[i], '__lt__') and ops[i] < ops[i + 1]):
                    return False
            return True
        return len(options) > 0 and all(isinstance(o, int) for o in options) and is_list_monotonic_increasing(options)


class Configurations:
    def __init__(self, rules: List[Rule]):
        self._rules: dict[str, Rule] = {rule.name: rule for rule in rules}
        self._key_rules: list[Rule] = []
        self._ordered_rules: list[Rule] = []
        self._ordered_values: dict[int, list[list[Any]]] = {}
        if len(self._rules) != len(rules):
            raise ValueError("rule name should be unique")

    def parse_header(self, names: List[str]):
        if len(names) == 0:
            raise ValueError("header should not be empty")

        output: list[Rule] = []
        for name in names:
            rule = self._rules.get(name)
            if rule is None:
                raise ValueError(f"name {name} does not match any rules")
            output.append(rule)

        self._key_rules = [rule for rule in output if rule.indexed]
        self._ordered_rules = output
        return self

    def parse_values(self, values: Iterator[List[str]]):
        assert len(self._ordered_rules) != 0, "must call parse_header before parse_values"
        output: dict[int, List[List[Any]]] = {}
        for row in values:
            item_key = 0
            item_values = []
            for (word, rule) in zip(row, self._ordered_rules):
                value = rule.convert_string_to_value(word)
                if rule.mapping or rule.indexed:
                    (index, limit) = rule.convert_value_to_index(value)
                    if rule.mapping:
                        value = index
                    if rule.indexed:
                        item_key = self.__update_key(item_key, value, limit)
                item_values.append(value)
            output.setdefault(item_key, [])
            output[item_key].append(item_values)

        self._ordered_values = output
        return self

    def lookup(self, keys: List[Union[str, Any]]) -> List[Dict[str, Any]]:
        assert len(self._ordered_rules) != 0, "must invoke parse_header before lookup"
        assert len(self._ordered_values) != 0, "must invoke parse_values before lookup"

        get = sorted(keys.keys())
        expect = sorted([rule.name for rule in self._key_rules])
        if expect != get:
            raise ValueError(f"expect keys ({expect}) but get ({get})")

        key = 0
        for rule in self._key_rules:
            value = keys.get(rule.name)
            index = rule.convert_value_to_index(value)
            key = self.__update_key(key, *index)

        def to_value(value_or_index: Any, rule: Rule):
            return value_or_index if not rule.mapping else rule.convert_index_to_value(value_or_index)

        def to_dict(values: List[Any], rules: List[Rule]):
            return {rule.name: to_value(value, rule) for (value, rule) in zip(values, rules)}
        return [to_dict(values, self._ordered_rules) for values in self._ordered_values.get(key, [])]

    @staticmethod
    def __update_key(key: int, index: int, limit: int) -> int:
        shift = (limit - 1).bit_length()
        return (key << shift) | index
