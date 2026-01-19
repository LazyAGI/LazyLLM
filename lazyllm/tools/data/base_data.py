import functools

from lazyllm import warp, LOG


class RegistryFactory:
    def __init__(self, wrapper=None):
        self._registry = {}
        self._tags = {}
        self._wrapper = wrapper

    def __getattr__(self, name):
        if name in self._registry:
            return self._registry[name]
        raise AttributeError(f'"{self.__class__.__name__}" has no attribute "{name}"')

    def register(self, obj=None, tag='default', **kwargs):
        if obj is None:
            # Support @register and @register(param=value) usages
            return functools.partial(self.register, tag=tag, **kwargs)

        if obj.__name__ in self._registry:
            raise ValueError(f'{obj.__name__} is already registered.')

        if self._wrapper:
            Wrapper = self._wrapper(obj, **kwargs)
        else:
            Wrapper = obj
        self._registry[obj.__name__] = Wrapper
        self._tags.setdefault(tag, []).append(obj.__name__)
        return Wrapper

def data_op_wrapper(func, one_item=True):  # noqa: C901
    @functools.wraps(func, updated=())
    class DataOpWrapper:
        def __init__(
                self, *args, _max_workers=32, _batch_size=None, _concurrency_mode='warp',
                _ignore_errors=False, **kwargs):
            self._obj = func(*args, **kwargs) if isinstance(func, type) else None
            self._args = args
            self._kwargs = kwargs
            self._max_workers = _max_workers
            self._batch_size = _batch_size or self._max_workers
            self._concurrency_mode = _concurrency_mode
            self._ignore_errors = _ignore_errors
            if one_item and self._concurrency_mode == 'warp':
                self._run = warp(self._run_one, _concurrent=self._max_workers).aslist

        def _run_one(self, data):
            try:
                if self._obj:
                    return self._obj(data)
                return func(data, *self._args, **self._kwargs)
            except Exception as e:
                if self._ignore_errors:
                    LOG.error(f'Error processing data in {func.__name__}: {e}')
                    return None
                raise e

        def _process_warp(self, data):
            results = []
            for i in range(0, len(data), self._batch_size):
                batch = data[i:i + self._batch_size]
                batch_res = self._run(batch)
                filtered_batch_res = []
                for one_res in batch_res:
                    if one_res is None:
                        if self._ignore_errors:
                            continue
                    if isinstance(one_res, list):
                        filtered_batch_res.extend(one_res)
                    else:
                        filtered_batch_res.append(one_res)
                # 后续加个filtered_batch_res的存储逻辑（逐批存）
                results.extend(filtered_batch_res)
            return results

        def _process_sequential(self, data):
            results = []
            for item in data:
                res = self._run_one(item)
                if res is None:
                    if self._ignore_errors:
                        continue
                if isinstance(res, list):
                    results.extend(res)
                else:
                    results.append(res)
                # 后续加个存储res的逻辑（逐个存）。
            return results

        def __call__(self, data):
            if not one_item:
                return self._run_one(data)
            assert isinstance(data, list)
            if self._concurrency_mode == 'warp':
                return self._process_warp(data)
            return self._process_sequential(data)

        def __repr__(self):
            return f'<DataOpWrapper for {func.__name__}>'
    return DataOpWrapper

DataOperatorRegistry = RegistryFactory(data_op_wrapper)
