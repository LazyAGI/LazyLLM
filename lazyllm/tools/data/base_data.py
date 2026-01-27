import os
import sys
import json
import time
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED

from lazyllm import LOG, config
import lazyllm
from lazyllm.common.registry import LazyLLMRegisterMetaClass

config.add(
    'data_process_path', str, '', 'DATA_PROCESS_PATH',
    description='The path to store data files.')
config.add(
    'data_process_resume', bool, False, 'DATA_PROCESS_RESUME',
    description='Whether to resume data processing from saved state.')

class DataStateStore:
    def __init__(self, func_name, save_data=True):
        self.save_data = save_data
        self.resume = config['data_process_resume']
        self.save_path = None
        self.progress_path = None
        self.buffer = []
        self.buffer_size_limit = 100
        self.last_save_time = time.time()

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

    def save_results(self, results, force=False):
        if not self.save_path:
            return

        if results:
            if isinstance(results, list):
                self.buffer.extend(results)
            else:
                self.buffer.append(results)

        current_time = time.time()
        time_diff = current_time - self.last_save_time

        # Intelligent storage: Adjust frequency based on buffer size and time
        if time_diff < 1.0:
            self.buffer_size_limit = min(self.buffer_size_limit * 2, 10000)
        elif time_diff > 10.0 and self.buffer_size_limit > 100:
            self.buffer_size_limit = max(self.buffer_size_limit // 2, 100)

        if force or (len(self.buffer) >= self.buffer_size_limit) or (time_diff > 5):
            self._flush()
            self.last_save_time = current_time

    def _flush(self):
        if not self.buffer: return
        try:
            with open(self.save_path, 'a', encoding='utf-8') as f:
                for res in self.buffer:
                    try:
                        line = json.dumps(res, ensure_ascii=False)
                    except Exception:
                        line = str(res)
                    f.write(line + '\n')
            self.buffer = []
        except Exception as e:
            LOG.error(f'Failed to save results to {self.save_path}: {e}')

    def save_progress(self, indices):
        if not self.save_path:
            return
        with open(self.progress_path, 'wb') as f:
            pickle.dump(indices, f)

    def load_results(self):
        # Ensure any remaining buffer is flushed before loading
        self._flush()
        results = []
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass
        return results

class LazyLLMDataBase(metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, _max_workers=None, _batch_size=None, _concurrency_mode='process',
                 _ignore_errors=False, _save_data=True, **kwargs):
        self._concurrency_mode = _concurrency_mode
        self._max_workers = _max_workers or (os.cpu_count() if _concurrency_mode == 'process' else 32)
        self._batch_size = _batch_size or self._max_workers
        self._ignore_errors = _ignore_errors
        self._store = DataStateStore(self.__class__.__name__, _save_data)
        self._lazyllm_kwargs = kwargs

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f) or \
            getattr(self.__class__, '__reg_overwrite__', None) == f

    def forward(self, input, **kwargs):
        raise NotImplementedError()

    def forward_batch_input(self, inputs, **kwargs):
        raise NotImplementedError()

    def _run_one(self, data):
        try:
            kwargs = getattr(self, '_lazyllm_kwargs', {})
            return self.forward(data, **kwargs)
        except Exception as e:
            if self._ignore_errors:
                LOG.error(f'Error processing data in {self.__class__.__name__}: {e}')
                return None
            raise e

    def _process_common(self, data):
        processed_indices = self._store.load_progress()
        results = []
        pbar = tqdm(total=len(data), desc=f'Processing {self.__class__.__name__}', unit='item')
        if len(processed_indices) > 0:
            pbar.update(len(processed_indices))

        pending_indices = [idx for idx in range(len(data)) if idx not in processed_indices]
        if not pending_indices:
            pbar.close()
            return self._store.load_results()

        if self._concurrency_mode == 'single':
            for idx in pending_indices:
                res = self._run_one(data[idx])
                self._handle_result(res, data[idx], results, processed_indices, [idx])
                pbar.update(1)
        else:
            self._process_parallel(data, pending_indices, results, processed_indices, pbar)

        pbar.close()
        # Flush remaining
        if self._store.save_data:
            self._store.save_results([], force=True)  # Flush
            self._store.save_progress(processed_indices)
            return self._store.load_results()
        return results

    def _ensure_picklable(self):
        # Fix dynamic class pickling issue for multiprocessing
        cls = self.__class__
        mod_name = cls.__module__
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            # Ensure the class is registered in the module with its name
            if not hasattr(mod, cls.__name__):
                setattr(mod, cls.__name__, cls)

            # Also check if qualname is used
            if '.' not in cls.__qualname__:
                if not hasattr(mod, cls.__qualname__):
                    setattr(mod, cls.__qualname__, cls)

    def _process_parallel(self, data, pending_indices, results, processed_indices, pbar):
        if self._concurrency_mode == 'process':
            self._ensure_picklable()

        executor_cls = ProcessPoolExecutor if self._concurrency_mode == 'process' else ThreadPoolExecutor
        idx_iter = iter(pending_indices)
        futures = {}

        with executor_cls(max_workers=self._max_workers) as executor:
            # 1. Submit initial batch
            for _ in range(self._max_workers):
                try:
                    idx = next(idx_iter)
                    fut = executor.submit(self._run_one, data[idx])
                    futures[fut] = idx
                except StopIteration:
                    break

            # 2. Loop
            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    idx = futures.pop(fut)
                    try:
                        res = fut.result()
                        self._handle_result(res, data[idx], results, processed_indices, [idx])
                    except Exception as e:
                        if not self._ignore_errors:
                            raise e
                        LOG.error(f'Task failed: {e}')

                    pbar.update(1)

                    # Submit next
                    try:
                        next_idx = next(idx_iter)
                        new_fut = executor.submit(self._run_one, data[next_idx])
                        futures[new_fut] = next_idx
                    except StopIteration:
                        pass

    def _handle_result(self, res, original_data, results, processed_indices, indices):
        # Logic to interpret return value
        final_res = []
        if res is None:
            final_res.append(original_data)  # Keep original
        elif isinstance(res, list):
            if res:  # Not empty
                final_res.extend(res)
            # Empty list means delete (do nothing)
        elif isinstance(res, dict):
            final_res.append(res)
        else:
            # Fallback or error? Assuming dict return for safety, but maybe primitive?
            # Doc says dict or List[dict].
            pass

        if self._store.save_data:
            self._store.save_results(final_res)
            processed_indices.update(indices)
            # Optimize progress saving frequency?
            # We can save progress periodically or just relies on memory set update
            # and save once at end? Or periodically?
            # To be safe for resume, we should save periodically.
            # Let's trust save_results handles data flush, we should handle progress flush.
            # But here we update the set. The set is in memory.
            # We can save pickle every X items.
            if len(processed_indices) % 100 == 0:
                self._store.save_progress(processed_indices)
        else:
            results.extend(final_res)

    def __call__(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]

        kwargs = getattr(self, '_lazyllm_kwargs', {})

        if self._overwrote('forward_batch_input'):
            if self._store.save_data and self._store.resume and 'Done' in self._store.load_progress():
                LOG.warning(f'skip {self.__class__.__name__} and load data from {self._store.save_path}')
                return self._store.load_results()

            res = self.forward_batch_input(inputs, **kwargs)

            if self._store.save_data and res is not None:
                self._store.save_results(res if isinstance(res, list) else [res], force=True)
                self._store.save_progress('Done')
            return res

        elif self._overwrote('forward'):
            return self._process_common(inputs)
        else:
            raise RuntimeError('Must implement forward or forward_batch_input')

data_register = lazyllm.Register(LazyLLMDataBase, ['forward', 'forward_batch_input'])
