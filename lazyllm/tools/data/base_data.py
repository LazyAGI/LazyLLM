import functools
import os
import json
import pickle
from tqdm import tqdm

from lazyllm import warp, LOG, config

config.add(
    'data_process_path', str, '', 'DATA_PROCESS_PATH',
    description='The path to store data files.')
config.add(
    'data_process_resume', bool, False, 'DATA_PROCESS_RESUME',
    description='Whether to resume data processing from saved state.')

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

class DataStateStore:
    def __init__(self, func_name, save_data=True):
        self.save_data = save_data
        self.resume = config['data_process_resume']
        self.save_path = None
        self.progress_path = None

        if self.save_data:
            root = config['data_process_path'] or os.path.join(os.getcwd(), 'data_pipeline_res')
            save_folder = os.path.join(root, func_name)
            os.makedirs(save_folder, exist_ok=True)
            self.save_path = os.path.join(save_folder, f'{func_name}_results.jsonl')
            self.progress_path = f'{self.save_path}.pkl'
            self._init_files()

    def _init_files(self):
        if self.save_path and not self.resume:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            if os.path.exists(self.progress_path):
                os.remove(self.progress_path)

    def load_progress(self):
        if self.save_path and self.resume and os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                LOG.warning(f'Failed to load progress from {self.progress_path}')
        return set()

    def save_results(self, results):
        if not self.save_path or not results:
            return
        with open(self.save_path, 'a', encoding='utf-8') as f:
            for res in results:
                try:
                    line = json.dumps(res, ensure_ascii=False)
                except Exception:
                    line = str(res)
                f.write(line + '\n')

    def save_progress(self, indices):
        if not self.save_path:
            return
        with open(self.progress_path, 'wb') as f:
            pickle.dump(indices, f)

    def load_results(self):
        results = []
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass
        return results

def data_op_wrapper(func, one_item=True):  # noqa: C901
    @functools.wraps(func, updated=())
    class DataOpWrapper:
        def __init__(
                self, *args, _max_workers=32, _batch_size=None, _concurrency_mode='warp',
                _ignore_errors=False, _save_data=True, **kwargs):
            self._obj = func(*args, **kwargs) if isinstance(func, type) else None
            self._args = args
            self._kwargs = kwargs
            self._max_workers = _max_workers
            self._batch_size = _batch_size or self._max_workers
            self._concurrency_mode = _concurrency_mode
            self._ignore_errors = _ignore_errors
            self._store = DataStateStore(func.__name__, _save_data)

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

        def _process_common(self, data, batch_size, process_func):
            processed_indices = self._store.load_progress()
            results = []
            pbar = tqdm(total=len(data), desc=f'Processing {func.__name__}', unit='item')

            for i in range(0, len(data), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(data))))
                pending_indices = [idx for idx in batch_indices if idx not in processed_indices]

                if not pending_indices:
                    pbar.update(len(batch_indices))
                    continue

                batch_data = [data[idx] for idx in pending_indices]
                batch_res = process_func(batch_data)

                filtered_batch_res = []
                for res in batch_res:
                    if res is None:
                        continue
                    if isinstance(res, list):
                        filtered_batch_res.extend(res)
                    else:
                        filtered_batch_res.append(res)

                if self._store.save_data:
                    self._store.save_results(filtered_batch_res)
                    processed_indices.update(pending_indices)
                    self._store.save_progress(processed_indices)
                else:
                    results.extend(filtered_batch_res)

                pbar.update(len(batch_indices))

            pbar.close()
            if self._store.save_data:
                return self._store.load_results()
            return results

        def __call__(self, data):
            if not one_item:
                if self._store.save_data and self._store.resume and 'Done' in self._store.load_progress():
                    LOG.warning(f'skip {func.__name__} and load data from {self._store.save_path}')
                    return self._store.load_results()
                res = self._run_one(data)
                if self._store.save_data and res is not None:
                    self._store.save_results(res if isinstance(res, list) else [res])
                    self._store.save_progress('Done')
                return res
            assert isinstance(data, list)
            if self._concurrency_mode == 'warp':
                return self._process_common(data, self._batch_size, self._run)
            # Fallback to single-threaded processing
            return self._process_common(data, 1, lambda batch: [self._run_one(item) for item in batch])

        def __repr__(self):
            return f'<DataOpWrapper for {func.__name__}>'
    return DataOpWrapper

DataOperatorRegistry = RegistryFactory(data_op_wrapper)
