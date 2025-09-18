import os
import abc
import json
import threading
from tqdm import tqdm
from datetime import datetime

import lazyllm
from lazyllm.module import ModuleBase
from lazyllm import warp


lazyllm.config.add('eval_result_dir', str, os.path.join(os.path.expanduser(lazyllm.config['home']), 'eval_res'),
                   'EVAL_RESULT_DIR')

class BaseEvaluator(ModuleBase):
    """Abstract base class for evaluation modules.

This class defines the standard interface and retry logic for evaluating model outputs. It supports concurrent processing, input validation, and automatic result saving.

Args:
    concurrency (int): Number of concurrent threads used during evaluation.
    retry (int): Number of retry attempts for each evaluation item.
    log_base_name (Optional[str]): Optional log file name prefix for saving results.


Examples:
    >>> from lazyllm.components import BaseEvaluator
    >>> class SimpleAccuracyEvaluator(BaseEvaluator):
    ...     def _process_one_data_impl(self, data):
    ...         return {
    ...             "final_score": float(data["pred"] == data["label"])
    ...         }
    >>> evaluator = SimpleAccuracyEvaluator()
    >>> score = evaluator([
    ...     {"pred": "yes", "label": "yes"},
    ...     {"pred": "no", "label": "yes"}
    ... ])
    >>> print(score)
    ... 0.5
    """
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
                lazyllm.LOG.warning(f'Validation failed on attempt {attempt}/{self._retry}')
            except Exception as e:
                lazyllm.LOG.error(f'Attempt {attempt}/{self._retry} failed: {str(e)}')
        lazyllm.LOG.error(f'All {self._retry} attempts exhausted')
        return ''

    def forward(self, data):
        if not data:
            lazyllm.LOG.warning('Empty input data received')
            return 0.0

        with tqdm(total=len(data), desc=self.__class__.__name__.title()) as progress_bar:
            results = self.batch_process(data, progress_bar)

        if not results:
            return 0.0

        total_score = sum(item.get('final_score', 0) for item in results)
        return total_score / len(results)

    def process_one_data(self, data, progress_bar=None):
        """Process a single data item.

Args:
    data: Data item to process.
    progress_bar (Optional[tqdm]): Progress bar object, defaults to None.

**Returns:**

- Any: Returns processing result.

Note:
    This method automatically updates the progress bar during processing and uses thread lock to ensure thread safety.
"""
        res = self._process_one_data_impl(data)
        if progress_bar is not None:
            with self._lock:
                progress_bar.update(1)
        return res

    @abc.abstractmethod
    def _process_one_data_impl(self, data):
        pass

    def validate_inputs_key(self, data):
        """Validate input data format and required keys.

Args:
    data: Data to validate.

Raises:
    RuntimeError: Raised when data format is incorrect or missing required keys.
        - If data is not a list
        - If items in the list are not dictionaries
        - If dictionaries are missing required keys
"""
        if not isinstance(data, list):
            raise RuntimeError(f'The data should be a list, but got {type(data)}')
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(f'The item at index {i} should be a dict, but got {type(item)}')
            missing_keys = [key for key in self._necessary_keys if key not in item]
            if missing_keys:
                raise RuntimeError(
                    f'The dict at index {i} should contain '
                    f'keys: {self._necessary_keys}, but cannot find: {missing_keys}')

    def batch_process(self, data, progress_bar):
        """Process data in batch.

Args:
    data: List of data to process.
    progress_bar (tqdm): Progress bar object.

**Returns:**

- List: Returns list of processing results.

Flow:
    1. Validates input data format and required keys
    2. Processes data using concurrent processor
    3. Saves processing results
"""
        self.validate_inputs_key(data)
        results = self._warp(data, progress_bar=progress_bar)
        self.save_res(results)
        return results

    def save_res(self, data, eval_res_save_name=None):
        """Save evaluation results.

Args:
    data: Data to save.
    eval_res_save_name (Optional[str]): Base name for the save file, defaults to class name.

Save Format:
    - Filename format: {filename}_{timestamp}.json
    - Timestamp format: YYYYMMDDHHmmSS
    - Save path: lazyllm.config['eval_result_dir']
    - JSON format with 4-space indentation
"""
        save_dir = lazyllm.config['eval_result_dir']
        os.makedirs(save_dir, exist_ok=True)

        filename = eval_res_save_name or self.__class__.__name__
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        save_path = os.path.join(save_dir, f'{filename}_{timestamp}.json')
        try:
            with open(save_path, 'w') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            lazyllm.LOG.error(f'Dump Json error: {e}')
