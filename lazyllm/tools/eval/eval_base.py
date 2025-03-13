import os
import abc
import json
import time
import threading
from functools import wraps
from datetime import datetime

import lazyllm
from lazyllm.module import ModuleBase
from lazyllm import warp
from lazyllm.common.bind import _MetaBind


lazyllm.config.add('eval_result_dir', str, os.path.join(os.path.expanduser('~'), '.lazyllm', 'eval_res'),
                   'EVAL_RESULT_DIR')

class EvalMeta(abc.ABCMeta, _MetaBind):
    def __new__(cls, name, bases, namespace):
        if 'process_one_data' in namespace:
            method = namespace['process_one_data']
            if not getattr(method, '__isabstractmethod__', False):
                @wraps(method)
                def wrapped_method(self, *args, **kwargs):
                    res = method(self, *args, **kwargs)
                    with self.lock:
                        if self.progress.total and self.progress.progress <= self.progress.total:
                            self.progress.update()
                    return res
                namespace['process_one_data'] = wrapped_method
        return super().__new__(cls, name, bases, namespace)


class BaseEvaluator(ModuleBase, metaclass=EvalMeta):
    def __init__(self, concurrency=1, retry=3, log_base_name=None):
        super().__init__()
        self.concurrency = concurrency
        self.retry = retry
        self.lock = threading.Lock()
        self.progress = ProgressManager(show_name=self.__class__.__name__.title())
        with warp(_concurrent=self.concurrency)as self.warp:
            self.warp.func = self.process_one_data
        self.necessary_keys = []

    def llm_infer(self, query, model):
        n = 1
        while n <= self.retry:
            try:
                res = model(query)
                return res
            except Exception as e:
                lazyllm.LOG.error(f'Eval-Infer Error: {e}')
                n += 1
        return ''

    def forward(self, data):
        results = self.batch_process(data)
        total_score = 0
        for item in results:
            total_score += item['final_score']
        return total_score / len(results)

    @abc.abstractmethod
    def process_one_data(self):
        pass

    def validate_inputs_key(self, data):
        if not isinstance(data, list):
            raise RuntimeError(f"The data should be a list, but got {type(data)}")
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(f"The item at index {i} should be a dict, but got {type(item)}")
            no_contain = [key for key in self.necessary_keys if key not in item]
            if no_contain:
                raise RuntimeError(
                    f"The dict at index {i} should contain "
                    f"keys: {self.necessary_keys}, but cannot find: {no_contain}")

    def batch_process(self, data):
        self.validate_inputs_key(data)
        with self.progress.set_size(len(data)):
            results = self.warp(data)
        self.save_res(results, self.__class__.__name__)
        return results

    def save_res(self, data, eval_res_save_name):
        save_dir = lazyllm.config['eval_result_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{eval_res_save_name}_{timestamp}.json")
        with open(save_path, 'w') as file:
            try:
                json.dump(data, file, ensure_ascii=False, indent=4)
            except Exception as e:
                lazyllm.LOG.error(f"Dump Json error: {e}")


class ProgressManager:
    def __init__(self, inputs_size=0, show_name=None):
        self.inputs_size = inputs_size
        self.progress = 0
        self.total = 0
        self.bar_length = 80
        self.show_name = show_name
        self.start_time = time.time()

    def set_size(self, inputs_size):
        self.inputs_size = inputs_size
        return self

    def __enter__(self):
        self.progress = 0
        self.start_time = time.time()
        self.total = self.inputs_size
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress = 0
        self.total = 0

    def update(self):
        self.progress += 1
        elapsed_time = time.time() - self.start_time
        percentage = (self.progress / self.total)
        arrow = '>' * int(self.bar_length * percentage)
        spaces = ' ' * (self.bar_length - len(arrow))
        if self.progress == self.total:
            end_char = '\n'
        else:
            end_char = ''
        print(f"\r{self.show_name} Process: [{arrow}{spaces}] {percentage:.2%} "
              f"- Elapsed Time: {elapsed_time:.2f}s", end=end_char)
