import ast
import json
import inspect
from typing import Any, Callable
from functools import update_wrapper


class _EmbedWrapper:
    SPARSE_FLOAT_VECTOR_DEFAULT_VALUE = {0: 0.0}

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        try:
            target = func if inspect.isroutine(func) else func.__call__
        except AttributeError:
            target = func
        update_wrapper(self, target)
        self.__wrapped__ = func

    def __getattr__(self, name: str) -> Any:
        return getattr(self.func, name)

    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        return self._normalize(res)

    def __reduce__(self):
        # NOTE avoid cloudpickle serialization error
        return (_EmbedWrapper, (self.func,))

    def _normalize(self, res: Any) -> Any:
        if isinstance(res, (bytes, bytearray, memoryview)):
            res = res.decode('utf-8', 'ignore')
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(res)
                except Exception:
                    raise ValueError('Embedding string is neither valid JSON nor'
                                     ' valid Python code for ast.literal_eval.')

        if isinstance(res, dict):
            return res or self.SPARSE_FLOAT_VECTOR_DEFAULT_VALUE
        if isinstance(res, list):
            return res
        # TODO (chenjiahao): support specific embedding item type check
        raise TypeError(f'Unexpected embedding type: {type(res)}')
