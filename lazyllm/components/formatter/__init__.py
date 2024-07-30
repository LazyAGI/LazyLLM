from .formatterBase import LazyLLMFormatterBase, LazyLLMFormatterBase as FormatterBase, EmptyFormatter
from .jsonFormatter import JsonFormatter


class Formatter(object):
    __map__ = dict(
        empty=EmptyFormatter,
        json=JsonFormatter,
    )

    def __new__(cls, ftype, rule):
        return Formatter.__map__[ftype](rule)


__all__ = [
    'LazyLLMFormatterBase',
    'FormatterBase',
    'Formatter',
    'EmptyFormatter',
    'JsonFormatter'
]
