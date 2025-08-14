import copy
from itertools import groupby
import json
import os
import requests
import re
import random
from typing import Tuple, List, Dict, Union, Any, Optional, TypedDict
from urllib.parse import urljoin
import time
from operator import itemgetter as itemget

import lazyllm
from lazyllm import globals
from lazyllm.components.prompter import PrompterBase
from lazyllm.components.formatter import FormatterBase
from lazyllm.components.utils.file_operate import _delete_old_files, _image_to_base64
from ....servermodule import LLMBase
from ....module import Pipeline


class StaticParams(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    frequency_penalty: float  # Note some online api use "repetition_penalty"


class OnlineChatModuleBase(LLMBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_LIST = []
    NO_PROXY = True

    def __init__(self, model_series: str, api_key: str, base_url: str, model_name: str,
                 stream: Union[bool, Dict[str, str]], return_trace: bool = False,
                 skip_auth: bool = False, static_params: Optional[StaticParams] = None, **kwargs):
        super().__init__(stream=stream, return_trace=return_trace)
        self._model_series = model_series
        if skip_auth and not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self.trainable_models = self.TRAINABLE_MODEL_LIST
        self._set_headers()
        self._set_chat_url()
        self._is_trained = False
        self._model_optional_params = {}
        self._vlm_force_format_input_with_files = False
        self._static_params = static_params or {}

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return "LLM"

    @property
    def static_params(self) -> StaticParams:
        return self._static_params

    @static_params.setter
    def static_params(self, value: StaticParams):
        if not isinstance(value, dict):
            raise TypeError("static_params must be a dict (TypedDict)")
        self._static_params = value

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        super().prompt('' if prompt is None else prompt, history=history)
        self._prompt._set_model_configs(system=self._get_system_prompt())
        return self

    def share(self, prompt: Optional[Union[str, dict, PrompterBase]] = None, format: Optional[FormatterBase] = None,
              stream: Optional[Union[bool, Dict[str, str]]] = None, history: Optional[List[List[str]]] = None,
              copy_static_params: bool = False):
        new = super().share(prompt, format, stream, history)
        if copy_static_params: new._static_params = copy.deepcopy(self._static_params)
        return new

    def _get_system_prompt(self):
        raise NotImplementedError("_get_system_prompt is not implemented.")

    def _set_headers(self):
        self._headers = {
            'Content-Type': 'application/json',
            **({'Authorization': 'Bearer ' + self._api_key} if self._api_key else {})
        }

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

    def _get_models_list(self):
        url = urljoin(self._base_url, 'models')
        headers = {'Authorization': 'Bearer ' + self._api_key} if self._api_key else None
        with requests.get(url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            res_json = r.json()
            return res_json

    def _convert_msg_format(self, msg: Dict[str, Any]):
        return msg

    def _str_to_json(self, msg: str, stream_output: bool):
        if isinstance(msg, bytes):
            pattern = re.compile(r"^data:\s*")
            msg = re.sub(pattern, "", msg.decode('utf-8'))
        try:
            message = self._convert_msg_format(json.loads(msg))
            if not stream_output: return message
            color = stream_output.get('color') if isinstance(stream_output, dict) else None
            for item in message.get("choices", []):
                delta = item.get('message', item.get('delta', {}))
                if (reasoning_content := delta.get("reasoning_content", '')):
                    self._stream_output(reasoning_content, color, cls='think')
                elif (content := delta.get("content", '')) and not delta.get('tool_calls'):
                    self._stream_output(content, color)
            lazyllm.LOG.debug(f"message: {message}")
            return message
        except Exception:
            return ""

    def _extract_specified_key_fields(self, response: Dict[str, Any]):
        if not ("choices" in response and isinstance(response["choices"], list)):
            raise ValueError(f"The response {response} does not contain a 'choices' field.")
        outputs = response['choices'][0].get("message") or response['choices'][0].get("delta", {})
        if 'reasoning_content' in outputs and outputs["reasoning_content"] and 'content' in outputs:
            outputs['content'] = r'<think>' + outputs.pop('reasoning_content') + r'</think>' + outputs['content']

        result, tool_calls = outputs.get('content', ''), outputs.get('tool_calls')
        if tool_calls:
            try:
                if isinstance(tool_calls, list): [item.pop('index', None) for item in tool_calls]
                tool_calls = tool_calls if isinstance(tool_calls, str) else json.dumps(tool_calls, ensure_ascii=False)
                if tool_calls: result += '<|tool_calls|>' + tool_calls
            except (KeyError, IndexError, TypeError):
                pass
        return result

    def _merge_stream_result(self, src: List[Union[str, int, list, dict]], force_join: bool = False):
        src = [ele for ele in src if ele is not None]
        if not src: return None
        elif len(src) == 1: return src[0]
        assert len(set(map(type, src))) == 1, f"The elements in the list: {src} are of inconsistent types"

        if isinstance(src[0], str):
            src = [ele for ele in src if ele]
            if not src: return ''
            if force_join or not all(src[0] == ele for ele in src): return ''.join(src)
        elif isinstance(src[0], list):
            assert len(set(map(len, src))) == 1, f"The lists of elements: {src} have different lengths."
            ret = list(map(self._merge_stream_result, zip(*src)))
            return ret[0] if isinstance(ret[0], list) else ret
        elif isinstance(src[0], dict):  # list of dicts
            if 'index' in src[-1]:
                grouped = [list(g) for _, g in groupby(sorted(src, key=itemget('index')), key=itemget("index"))]
                if len(grouped) > 1: return [self._merge_stream_result(src) for src in grouped]
            return {k: self._merge_stream_result([d.get(k) for d in src], k == 'content') for k in set().union(*src)}
        return src[-1]

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = False, lazyllm_files=None, **kw):
        """LLM inference interface"""
        stream_output = stream_output or self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        params = {'input': __input, 'history': llm_chat_history, 'return_dict': True}
        if tools: params["tools"] = tools
        data = self._prompt.generate_prompt(**params)
        data.update(self._static_params, **dict(model=self._model_name, stream=bool(stream_output)))

        if len(kw) > 0: data.update(kw)
        if len(self._model_optional_params) > 0: data.update(self._model_optional_params)

        if files or (self._vlm_force_format_input_with_files and data["model"] in self.VLM_MODEL_LIST):
            data["messages"][-1]["content"] = self._format_input_with_files(data["messages"][-1]["content"], files)

        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        with requests.post(self._url, json=data, headers=self._headers, stream=stream_output, proxies=proxies) as r:
            if r.status_code != 200:  # request error
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)])) \
                    if stream_output else requests.RequestException(r.text)

            with self.stream_output(stream_output):
                msg_json = list(filter(lambda x: x, ([self._str_to_json(line, stream_output) for line in r.iter_lines()
                                if len(line)] if stream_output else [self._str_to_json(r.text, stream_output)]),))

            usage = {"prompt_tokens": -1, "completion_tokens": -1}
            if len(msg_json) > 0 and "usage" in msg_json[-1] and isinstance(msg_json[-1]["usage"], dict):
                for k in usage:
                    usage[k] = msg_json[-1]["usage"].get(k, usage[k])
            self._record_usage(usage)
            extractor = self._extract_specified_key_fields(self._merge_stream_result(msg_json))
            return self._formatter(extractor) if extractor else ""

    def _record_usage(self, usage: dict):
        globals["usage"][self._module_id] = usage
        par_muduleid = self._used_by_moduleid
        if par_muduleid is None:
            return
        if par_muduleid not in globals["usage"]:
            globals["usage"][par_muduleid] = usage
            return
        existing_usage = globals["usage"][par_muduleid]
        if existing_usage["prompt_tokens"] == -1 or usage["prompt_tokens"] == -1:
            globals["usage"][par_muduleid] = {"prompt_tokens": -1, "completion_tokens": -1}
        else:
            for k in globals["usage"][par_muduleid]:
                globals["usage"][par_muduleid][k] += usage[k]

    def _upload_train_file(self, train_file) -> str:
        raise NotImplementedError(f"{self._model_series} not implemented _upload_train_file method in subclass")

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_series} not implemented _create_finetuning_job method in subclass")

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_series} not implemented _query_finetuning_job method in subclass")

    def _query_finetuned_jobs(self) -> dict:
        raise NotImplementedError(f"{self._model_series} not implemented _query_finetuned_jobs method in subclass")

    def _get_finetuned_model_names(self) -> Tuple[List[str], List[str]]:
        raise NotImplementedError(f"{self._model_series} not implemented _get_finetuned_model_names method in subclass")

    def set_train_tasks(self, train_file, **kw):
        self._train_file = train_file
        self._train_parameters = kw

    def set_specific_finetuned_model(self, model_id):
        valid_jobs, _ = self._get_finetuned_model_names()
        valid_model_id = [model for _, model in valid_jobs]
        if model_id in valid_model_id:
            self._model_name = model_id
            self._is_trained = True
        else:
            raise ValueError(f"Cannot find modle({model_id}), in fintuned model list: {valid_model_id}")

    def _get_temp_save_dir_path(self):
        save_dir = os.path.join(lazyllm.config['temp_dir'], 'online_model_sft_log')
        if not os.path.exists(save_dir):
            os.system(f'mkdir -p {save_dir}')
        else:
            _delete_old_files(save_dir)
        return save_dir

    def _validate_api_key(self):
        try:
            self._query_finetuned_jobs()
            return True
        except Exception:
            return False

    def _get_train_tasks(self):
        if not self._model_name or not self._train_file:
            raise ValueError("train_model and train_file is required")
        if self._model_name not in self.trainable_models:
            lazyllm.LOG.log_once(f"The current model {self._model_name} is not in the trainable \
                                  model list {self.trainable_models}. The deadline for this list is June 1, 2024. \
                                  This model may not be trainable. If your model is a new model, \
                                  you can ignore this warning.")

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
        raise NotImplementedError(f"{self._model_series} not implemented _create_deployment method in subclass")

    def _query_deployment(self, deployment_id) -> str:
        raise NotImplementedError(f"{self._model_series} not implemented _query_deployment method in subclass")

    def _get_deploy_tasks(self):
        if not self._is_trained: return None

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

    def _format_vl_chat_query(self, query: str):
        return [{"type": "text", "text": query}]

    def _format_vl_chat_image_url(self, image_url: str, mime: str) -> List[Dict[str, str]]:
        return [{"type": "image_url", "image_url": {"url": image_url}}]

    # for online vlm
    def _format_input_with_files(self, query: str, query_files: list[str]) -> List[Dict[str, str]]:
        if not query_files:
            return self._format_vl_chat_query(query)
        output = [{"type": "text", "text": query}]
        assert isinstance(query_files, list), "query_files must be a list."
        for file in query_files:
            mime = None
            if not file.startswith("http"):
                file, mime = _image_to_base64(file)
            output.extend(self._format_vl_chat_image_url(file, mime))
        return output

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineChat', name=self._module_name, url=self._base_url,
                                 stream=bool(self._stream), return_trace=self._return_trace)
