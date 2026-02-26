from typing import Tuple, Optional, Union

from lazyllm import LOG, JsonFormatter, ChatPrompter


class LLMDataJson:
    _default_prompt: Optional[Union[ChatPrompter, str]] = None
    _default_inference_kwargs = {
        'max_new_tokens': 512,
        'temperature': 0.2,
    }

    def __init__(self, model, prompt=None, max_retries=3, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        assert prompt is not None or self._default_prompt is not None, 'Prompt must be provided'
        prompt = prompt if prompt is not None else self._default_prompt
        self.model = model.share().prompt(prompt).formatter(JsonFormatter())
        self._max_retries = max_retries

    def preprocess(self, data: dict, **kwargs) -> Tuple[dict, dict]:
        raise NotImplementedError()

    def verify_output(self, output: dict, data: dict) -> bool:
        raise NotImplementedError()

    def postprocess(self, output: dict, data: dict) -> dict:
        raise NotImplementedError()

    def forward(self, data: dict, **kwargs) -> dict:
        prepared_data, infer_kwargs = self.preprocess(data, **kwargs)
        for key, default_val in self._default_inference_kwargs.items():
            infer_kwargs[key] = infer_kwargs.get(key, default_val)
        error_log = []
        for i in range(self._max_retries):
            try:
                res = self.model(prepared_data, **infer_kwargs)
                if self.verify_output(res, data):
                    return self.postprocess(res, data)
            except Exception as e:
                LOG.warning(f'LLM inference failed, try {i+1}/{self._max_retries}, Error: {e}')
                error_log.append(str(e))
                continue
        else:
            raise RuntimeError(f'LLM inference failed after {self._max_retries} retries. Errors: {"; ".join(error_log)}')
