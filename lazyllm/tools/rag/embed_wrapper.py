import json
from typing import Any, Callable, Union, List, Dict
from functools import update_wrapper

Number = (int, float)
Vector = List[float]
Sparse = Dict[str, Any]
Matrix = List[Union[Vector, Sparse]]
EmbeddingRet = Union[Vector, Matrix, Sparse]


class _EmbedWrapper:
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        target = getattr(func, "__call__", func)
        update_wrapper(self, target)
        self.__wrapped__ = func

    def __getattr__(self, name: str) -> Any:
        return getattr(self.func, name)

    def __call__(self, *args, **kwargs) -> EmbeddingRet:
        res = self.func(*args, **kwargs)
        return self._normalize(res)

    def _normalize(self, res: Any) -> EmbeddingRet:
        if isinstance(res, (bytes, bytearray)):
            res = res.decode("utf-8", "ignore")
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except json.JSONDecodeError as e:
                raise ValueError("Embedding string is not valid JSON.") from e

        if isinstance(res, dict):
            return res

        if isinstance(res, list):
            if self._is_vector(res) or self._is_matrix(res):
                return res  # List[Dict[str, Any]] or List[List[float]] or List[float]
            raise TypeError("Embedding list must be List[float] or List[List[float]].")

        raise TypeError(f"Unexpected embedding type: {type(res)}")

    @staticmethod
    def _is_vector(x: list) -> bool:
        return all(isinstance(t, Number) for t in x)

    @staticmethod
    def _is_matrix(x: list) -> bool:
        return len(x) > 0 and all(
            (isinstance(r, list) and all(isinstance(t, Number) for t in r) for r in x)
            or (all(isinstance(r, dict) for r in x))
        )
