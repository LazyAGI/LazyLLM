from typing import Any
import copy
import multiprocessing
from .logger import LOG


class _OptionIterator(object):
    def __init__(self, m):
        self.m = m
        self.reset()

    def reset(self): self.m._idx = -1
    def __len__(self): return len(self.m._objs)
    def __deepcopy__(self, *args, **kw): return self

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        self.m._next()
        return self.m._obj


class Option(object):
    """
Option management class provided by LazyLLM, used for managing multiple option values and iterating between them. This class is primarily used for parameter grid search and hyperparameter tuning scenarios.

Args:
    *obj: One or more option values, which can be objects of any type. If a single list or tuple is passed, it will be automatically expanded. At least two options must be provided.

Key features:

- **Multi-option Management**: Can manage multiple different option values.
- **Iteration Support**: Supports standard Python iteration protocol, allowing traversal of all options.
- **Current Value Access**: Provides access to the currently selected option.
- **Deep Copy**: Supports obtaining a deep copy of the currently selected option.
- **Cyclic Iteration**: Options can be iterated over in a cyclic manner.

**Note**: This class is mainly used for LazyLLM's internal parameter search and trial management, especially in `TrialModule` for parameter grid search.


Examples:
    >>> import lazyllm
    >>> from lazyllm.common.option import Option
    >>> learning_rates = Option(0.001, 0.01, 0.1)
    >>> print(f"当前学习率: {learning_rates}")
    当前学习率: <Option options="(0.001, 0.01, 0.1)" curr="0.001">
    >>> print(f"所有选项: {list(learning_rates)}")
    所有选项: [0.001, 0.01, 0.1]
    """
    def __init__(self, *obj):
        if len(obj) == 1 and isinstance(obj[0], (tuple, list)): obj = obj[0]
        assert isinstance(obj, (tuple, list)) and len(obj) > 1, 'More than one option shoule be given'
        self._objs = obj
        self._idx = 0
        self._obj = self._objs[self._idx]

    def _next(self):
        self._idx += 1
        if self._idx == len(self._objs):
            self._idx = 0
            raise StopIteration

    def __setattr__(self, __name: str, __value: Any) -> None:
        object.__setattr__(self, __name, __value)
        if __name == '_idx' and 0 <= self._idx < len(self._objs):
            self._obj = self._objs[self._idx]

    def __deepcopy__(self, *args, **kw):
        return copy.deepcopy(self._obj)

    def __iter__(self):
        return _OptionIterator(self)

    def __repr__(self):
        return f'<Option options="{self._objs}" curr="{self._obj}">'


def rebuild(x): return x
def reduce(x): return rebuild, (x._obj,)
multiprocessing.reducer.ForkingPickler.register(Option, reduce)


def OptionIter(list_of_options, suboption_func=lambda x: []):
    LOG.info('Options:', list_of_options)

    def impl(cur, remain):
        for r in cur:
            new_remain = remain + suboption_func(r)
            if len(new_remain) == 0:
                yield [r]
            else:
                for r2 in impl(new_remain[0], new_remain[1:]):
                    yield [r] + r2
    return impl(list_of_options[0], list_of_options[1:])
