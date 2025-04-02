import os
import abc
import json
import threading
from tqdm import tqdm
from datetime import datetime

import lazyllm
from lazyllm.module import ModuleBase
from lazyllm import warp


lazyllm.config.add('eval_result_dir', str, os.path.join(os.path.expanduser('~'), '.lazyllm', 'eval_res'),
                   'EVAL_RESULT_DIR')

class BaseEvaluator(ModuleBase):
    def __init__(self, concurrency=1, retry=3, log_base_name=None):
        super().__init__()
        self._concurrency = concurrency
        self._retry = retry
        self._lock = threading.Lock()
        self._warp = warp(self.process_one_data, _concurrent=self._concurrency)
        self._necessary_keys = []

    def _execute_with_retries(self, input_data, func, result_validator=None, post_processor=None):
        for attempt in range(1, self._retry + 1):
            try:
                result = func(input_data)
                if post_processor is not None:
                    result = post_processor(result)
                if result_validator is None or result_validator(result):
                    return result
                lazyllm.LOG.warning(f"Validation failed on attempt {attempt}/{self._retry}")
            except Exception as e:
                lazyllm.LOG.error(f"Attempt {attempt}/{self._retry} failed: {str(e)}")
        lazyllm.LOG.error(f"All {self._retry} attempts exhausted")
        return ''

    def forward(self, data):
        if not data:
            lazyllm.LOG.warning("Empty input data received")
            return 0.0

        with tqdm(total=len(data), desc=self.__class__.__name__.title()) as progress_bar:
            results = self.batch_process(data, progress_bar)

        if not results:
            return 0.0

        total_score = sum(item.get('final_score', 0) for item in results)
        return total_score / len(results)

    def process_one_data(self, data, progress_bar=None):
        res = self._process_one_data_impl(data)
        if progress_bar is not None:
            with self._lock:
                progress_bar.update(1)
        return res

    @abc.abstractmethod
    def _process_one_data_impl(self, data):
        pass

    def validate_inputs_key(self, data):
        if not isinstance(data, list):
            raise RuntimeError(f"The data should be a list, but got {type(data)}")
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(f"The item at index {i} should be a dict, but got {type(item)}")
            missing_keys = [key for key in self._necessary_keys if key not in item]
            if missing_keys:
                raise RuntimeError(
                    f"The dict at index {i} should contain "
                    f"keys: {self._necessary_keys}, but cannot find: {missing_keys}")

    def batch_process(self, data, progress_bar):
        self.validate_inputs_key(data)
        results = self._warp(data, progress_bar=progress_bar)
        self.save_res(results)
        return results

    def save_res(self, data, eval_res_save_name=None):
        save_dir = lazyllm.config['eval_result_dir']
        os.makedirs(save_dir, exist_ok=True)

        filename = eval_res_save_name or self.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{filename}_{timestamp}.json")
        try:
            with open(save_path, 'w') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            lazyllm.LOG.error(f"Dump Json error: {e}")
