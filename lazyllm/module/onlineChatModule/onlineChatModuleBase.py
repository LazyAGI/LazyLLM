from itertools import groupby
import json
import os
import requests
import re
from typing import Tuple, List, Dict, Union, Any
import time
import lazyllm
from lazyllm.components.prompter import PrompterBase, ChatPrompter
from lazyllm.components.formatter import FormatterBase, EmptyFormatter
from ..module import ModuleBase, Pipeline

class OnlineChatModuleBase(ModuleBase):

    def __init__(self,
                 model_type: str,
                 api_key: str,
                 base_url: str,
                 model_name: str,
                 stream: bool,
                 trainable_models: List[str],
                 return_trace: bool = False,
                 **kwargs):
        super().__init__(return_trace=return_trace)
        self._model_type = model_type
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._stream = stream
        self.trainable_mobels = trainable_models
        self._set_headers()
        self._set_chat_url()
        self.prompt()
        self._is_trained = False
        self.formatter()
        self.field_extractor()
        self._model_optional_params = {}

    def prompt(self, prompt=None):
        if prompt is None:
            self._prompt = ChatPrompter()
        elif isinstance(prompt, PrompterBase):
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt)
        else:
            raise TypeError(f"{prompt} type is not supported.")
        self._prompt._set_model_configs(system=self._get_system_prompt())
        return self

    def _get_system_prompt(self):
        raise NotImplementedError("_get_system_prompt is not implemented.")

    def _set_headers(self):
        self._headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self._api_key
        }

    @property
    def model_type(self):
        return self._model_type

    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'chat/completions')

    def _get_models_list(self):
        url = os.path.join(self._base_url, 'models')
        headers = {'Authorization': 'Bearer ' + self._api_key}
        with requests.get(url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            res_json = r.json()
            return res_json

    def _parse_output_by_key(self, key: str, data: Dict[str, Any]):
        if "choices" in data and isinstance(data["choices"], list):
            item = data['choices'][0]
            data = item.get("delta", {}) if "delta" in item else item.get("message", {})
            return data if not key or key == "." else data.get(key, "")
        else:
            raise ValueError(f"The response {data} does not contain a 'choices' field.")

    def formatter(self, format: FormatterBase = None):
        if isinstance(format, FormatterBase):
            self._formatter = format
        elif format is None:
            self._formatter = EmptyFormatter()
        else:
            raise TypeError("format must be a FormatterBase")

        return self

    def field_extractor(self, key: Union[str, List[str]] = None):
        if key is None:
            self._extractor_fields = ["content"]
        elif isinstance(key, str):
            self._extractor_fields = [key]
        elif isinstance(key, list):
            self._extractor_fields = key
        else:
            raise TypeError(f"Unsupported type: {type(key)}")

        return self

    def _convert_msg_format(self, msg: Dict[str, Any]):
        return msg

    def _str_to_json(self, msg: str):
        if isinstance(msg, bytes):
            pattern = re.compile(r"^data:\s*")
            msg = re.sub(pattern, "", msg.decode('utf-8'))
        try:
            chunk = json.loads(msg)
            return self._convert_msg_format(chunk)
        except Exception:
            return ""

    def _extract_specified_key_fields(self, response: Dict[str, Any]):
        if len(self._extractor_fields) == 1:
            key = self._extractor_fields[0]
            return self._parse_output_by_key(key, response) if key else ""
        elif len(self._extractor_fields) > 1:
            res = {}
            for key in self._extractor_fields:
                res[key] = self._parse_output_by_key(key, response) if key else ""
            return res
        else:
            return self._parse_output_by_key(".", response)

    def _merge_stream_result(self, src: List[str | int | list | dict]):
        assert len(src) > 0 and all(isinstance(ele, type(src[-1])) or ele is None for ele in src), \
               f"The elements in the list: {src} are of inconsistent types"
        if len(src) == 1:
            return src[0]
        if all(isinstance(ele, str) or ele is None for ele in src):
            if all(ele == src[-1] or ele is None for ele in src) or (self._model_optional_params
               and not self._model_optional_params.get("incremental_output", True)):
                return src[-1]
            else:
                return "".join(ele for ele in src if ele is not None)
        elif all(isinstance(ele, list) for ele in src):
            assert all(len(src[-1]) == len(ele) for ele in src), f"The lists of elements: {src} have different lengths."
            ret = [self._merge_stream_result([ele[idx] for ele in src]) for idx in range(len(src[-1]))]
            return ret[0] if isinstance(ret[0], list) else ret
        elif all(isinstance(ele, dict) for ele in src):
            if "index" in src[-1]:  # If there are multiple index values that need to be appended.
                data_sorted = sorted(src, key=lambda x: x['index'])
                grouped_data = [list(g) for k, g in groupby(data_sorted, key=lambda x: x['index'])]
                if len(grouped_data) > 1:
                    return [self._merge_stream_result(src) for src in grouped_data]
            return {k: "tool_calls" if k == "finish_reason" and "tool_calls" in [d[k] for d in src if k in d]
                    else self._merge_stream_result([d[k] for d in src if k in d]) for k in set().union(*src)}
        elif all(isinstance(ele, int) for ele in src):
            return src[-1] if all(ele == src[-1] for ele in src) else src[-1]
        else:
            raise TypeError(f"The elements in list {src} are of inconsistent types.")

    def forward(self, __input: Union[Dict, str] = None, llm_chat_history: List[List[str]] = None, tools: List[Dict[str, Any]] = None, **kw):  # noqa C901
        """LLM inference interface"""
        params = {"input": __input, "history": llm_chat_history}
        if tools:
            params["tools"] = tools
        params["return_dict"] = True
        data = self._prompt.generate_prompt(**params)

        data["model"] = self._model_name
        data["stream"] = self._stream
        if len(kw) > 0:
            data.update(kw)

        if len(self._model_optional_params) > 0:
            data.update(self._model_optional_params)

        with requests.post(self._url, json=data, headers=self._headers, stream=self._stream) as r:
            if r.status_code != 200:  # request error
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)])) \
                    if self._stream else requests.RequestException(r.text)

            resp = [line for line in r.iter_lines() if len(line)] if self._stream else [r.text]
            msg_json = list(filter(lambda x: x, [self._str_to_json(line) for line in resp]))
            extractor = self._extract_specified_key_fields(self._merge_stream_result(msg_json))

            return self._formatter.format(extractor) if extractor else ""

    def _set_template(self, template_message=None, keys_name_handle=None, template_headers=None):
        self.template_message = template_message
        self.keys_name_handle = keys_name_handle
        self.template_headers = template_headers

    def _upload_train_file(self, train_file) -> str:
        raise NotImplementedError(f"{self._model_type} not implemented _upload_train_file method in subclass")

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_type} not implemented _create_finetuning_job method in subclass")

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_type} not implemented _query_finetuning_job method in subclass")

    def set_train_tasks(self, train_file, **kw):
        self._train_file = train_file
        self._train_parameters = kw

    def _get_train_tasks(self):
        if not self._model_name or not self._train_file:
            raise ValueError("train_model and train_file is required")
        if self._model_name not in self.trainable_mobels:
            raise ValueError(f"{self._model_name} is not trainable")

        def _create_for_finetuning_job():
            """
            create for finetuning job to finish
            """
            file_id = self._upload_train_file(train_file=self._train_file)
            lazyllm.LOG.info(f"{os.path.basename(self._train_file)} upload success! file id is {file_id}")
            (fine_tuning_job_id, status) = self._create_finetuning_job(self._model_name,
                                                                       file_id,
                                                                       **self._train_parameters)
            lazyllm.LOG.info(f"fine tuning job {fine_tuning_job_id} created, status: {status}")

            if status.lower() == "failed":
                raise ValueError(f"Fine tuning job {fine_tuning_job_id} failed")
            import random
            while status.lower() != "succeeded":
                try:
                    # wait 10 seconds before querying again
                    time.sleep(random.randint(60, 120))
                    (fine_tuned_model, status) = self._query_finetuning_job(fine_tuning_job_id)
                    lazyllm.LOG.info(f"fine tuning job {fine_tuning_job_id} status: {status}")
                    if status.lower() == "failed":
                        raise ValueError(f"Finetuning job {fine_tuning_job_id} failed")
                except ValueError:
                    raise ValueError(f"Finetuning job {fine_tuning_job_id} failed")

            lazyllm.LOG.info(f"fine tuned model: {fine_tuned_model} finished")
            self._model_name = fine_tuned_model
            self._is_trained = True

        return Pipeline(_create_for_finetuning_job)

    def _create_deployment(self) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_type} not implemented _create_deployment method in subclass")

    def _query_deployment(self, deployment_id) -> str:
        raise NotImplementedError(f"{self._model_type} not implemented _query_deployment method in subclass")

    def _get_deploy_tasks(self):
        if not self._is_trained: None

        def _start_for_deployment():
            (deployment_id, status) = self._create_deployment()
            lazyllm.LOG.info(f"deployment {deployment_id} created, status: {status}")

            if status.lower() == "failed":
                raise ValueError(f"Deployment task {deployment_id} failed")
            status = self._query_deployment(deployment_id)
            while status.lower() != "running":
                # wait 10 seconds before querying again
                time.sleep(10)
                status = self._query_deployment(deployment_id)
                lazyllm.LOG.info(f"deployment {deployment_id} status: {status}")
                if status.lower() == "failed":
                    raise ValueError(f"Deployment task {deployment_id} failed")
            lazyllm.LOG.info(f"deployment {deployment_id} finished")
        return Pipeline(_start_for_deployment)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineChat', name=self._module_name, url=self._base_url,
                                 stream=self._stream, return_trace=self._return_trace)
