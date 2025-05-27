import copy
from itertools import groupby
import json
import os
import requests
import re
import random
from typing import Tuple, List, Dict, Union, Any, Optional
from urllib.parse import urljoin
import time

import lazyllm
from lazyllm import globals, FileSystemQueue
from lazyllm.components.prompter import PrompterBase, ChatPrompter
from lazyllm.components.formatter import (FormatterBase, EmptyFormatter,
                                          encode_query_with_filepaths, decode_query_with_filepaths)
from lazyllm.components.formatter.formatterbase import LAZYLLM_QUERY_PREFIX
from lazyllm.components.utils.file_operate import delete_old_files, image_to_base64
from ..module import ModuleBase, Pipeline


class OnlineChatModuleBase(ModuleBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_LIST = []

    def __init__(self,
                 model_series: str,
                 api_key: str,
                 base_url: str,
                 model_name: str,
                 stream: Union[bool, Dict[str, str]],
                 return_trace: bool = False,
                 vlm_models: List[str] = None,
                 **kwargs):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._stream = stream
        self.trainable_models = self.TRAINABLE_MODEL_LIST
        self.vlm_models = self.VLM_MODEL_LIST
        self._set_headers()
        self._set_chat_url()
        self.prompt()
        self._is_trained = False
        self.formatter()
        self._field_extractor()
        self._model_optional_params = {}
        self._vlm_force_format_input_with_files = False

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return "LLM"

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, v: Union[bool, Dict[str, str]]):
        self._stream = v

    def prompt(self, prompt=None, history: List[List[str]] = None):
        if prompt is None:
            self._prompt = ChatPrompter(history=history)
        elif isinstance(prompt, PrompterBase):
            assert not history, 'history is not supported in user defined prompter'
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt, history=history)
        else:
            raise TypeError(f"{prompt} type is not supported.")
        self._prompt._set_model_configs(system=self._get_system_prompt())
        return self

    def share(self, prompt: PrompterBase = None, format: FormatterBase = None, stream: Optional[bool] = None,
              history: List[List[str]] = None):
        new = copy.copy(self)
        new._hooks = set()
        new._set_mid()
        if prompt is not None: new.prompt(prompt, history=history)
        if format is not None: new.formatter(format)
        if stream is not None: new.stream = stream
        return new

    def _get_system_prompt(self):
        raise NotImplementedError("_get_system_prompt is not implemented.")

    def _set_headers(self):
        self._headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self._api_key
        }

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

    def _get_models_list(self):
        url = urljoin(self._base_url, 'models')
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
        if isinstance(format, FormatterBase) or callable(format):
            self._formatter = format
        elif format is None:
            self._formatter = EmptyFormatter()
        else:
            raise TypeError("format must be a FormatterBase")

        return self

    def _field_extractor(self, key: Union[str, List[str]] = None):
        if key is None:
            self._extractor_fields = ["{content}" + globals['tool_delimiter'] + "{tool_calls|index}"]
        elif isinstance(key, str):
            self._extractor_fields = [key]
        elif isinstance(key, list):
            self._extractor_fields = key
        else:
            raise TypeError(f"Unsupported type: {type(key)}")

        return self

    def _convert_msg_format(self, msg: Dict[str, Any]):
        return msg

    def _str_to_json(self, msg: str, stream_output: bool):
        if isinstance(msg, bytes):
            pattern = re.compile(r"^data:\s*")
            msg = re.sub(pattern, "", msg.decode('utf-8'))
        try:
            chunk = json.loads(msg)
            message = self._convert_msg_format(chunk)
            if stream_output:
                color = stream_output.get('color') if isinstance(stream_output, dict) else None
                for item in message.get("choices", []):
                    delta = item.get("delta", {})
                    reasoning_content = delta.get("reasoning_content", '')
                    if reasoning_content:
                        content = reasoning_content
                        FileSystemQueue().get_instance("think").enqueue(lazyllm.colored_text(content, color))
                    else:
                        content = delta.get("content", '')
                        if content and "tool_calls" not in delta:
                            FileSystemQueue().enqueue(lazyllm.colored_text(content, color))
            lazyllm.LOG.debug(f"message: {message}")
            return message
        except Exception:
            return ""

    def _get_benchmark_data(self, data: Dict[str, Any]):
        if "choices" in data and isinstance(data["choices"], list):
            item = data['choices'][0]
            outputs = item.get("message", item.get("delta", {}))
            if 'reasoning_content' in outputs and 'content' in outputs:
                outputs['content'] = r'<think>' + outputs.pop('reasoning_content') + r'</think>' + outputs['content']
            return outputs
        else:
            raise ValueError(f"The response {data} does not contain a 'choices' field.")

    def _extract_and_format(self, data, template):  # noqa: C901
        # finding placeholders in template and removing rules
        placeholders = re.findall(r"{(.*?)(?:\|(.*?))?}", template)
        delimiters = re.findall(r"<\|.*?\|>", template)
        # extract and format the fields corresponding to the placeholders
        extracted_data = {}
        pkeys = []
        for placeholder, remove_fields in placeholders:
            placeholder_key = placeholder + "|" + remove_fields if remove_fields else placeholder
            pkeys.append(placeholder_key)
            if 'tool_calls' in placeholder:
                # handling remove_fields
                remove_fields = remove_fields.split(',') if remove_fields else []

                # extract the tool_calls field
                keys = placeholder.split('.')
                value = data
                try:
                    for key in (int(key) if key.isdigit() else key for key in keys):
                        value = value[key]

                    if isinstance(value, list):
                        for item in value:
                            [item.pop(field) for field in remove_fields if field in item]
                    # value = json.dumps(value).replace('\n', '').replace(' ', '')
                    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    extracted_data[placeholder_key] = value
                except (KeyError, IndexError, TypeError):
                    extracted_data[placeholder_key] = ""
            else:
                # extracting additional fields
                keys = placeholder.split('.')
                value = data
                try:
                    for key in (int(key) if key.isdigit() else key for key in keys):
                        value = value[key]
                    # convert the extracted value into a JSON string
                    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    extracted_data[placeholder_key] = value
                except (KeyError, IndexError, TypeError):
                    extracted_data[placeholder_key] = ""

        # populate the template with the extracted data
        assert len(extracted_data) == len(delimiters) + 1, \
               "The delimiters and the number of extracted fields are inconsistent."
        result = extracted_data.get(pkeys[0])
        result += ''.join(delimiters[idx] + extracted_data[key]
                          for idx, key in enumerate(pkeys[1:]) if extracted_data.get(key))
        lazyllm.LOG.debug(f"result: {result}")
        return result

    def _extract_specified_key_fields(self, response: Dict[str, Any]):
        if len(self._extractor_fields) > 0:
            res = {}
            for key in self._extractor_fields:
                res[key] = (self._parse_output_by_key(key, response) if "{" not in key else self._extract_and_format(
                    self._get_benchmark_data(response), key) if key else "")
            return list(res.values())[0] if len(res) == 1 else json.dumps(res, ensure_ascii=False)
        else:
            return json.dumps(self._parse_output_by_key(".", response), ensure_ascii=False)

    def _merge_stream_result(self, src: List[str | int | list | dict]):
        types = set(type(ele) for ele in src if ele is not None)
        assert len(src) > 0 and len(types) <= 1, f"The elements in the list: {src} are of inconsistent types"
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

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None, tools: List[Dict[str, Any]] = None, stream_output: bool = False, lazyllm_files=None, **kw):  # noqa C901
        """LLM inference interface"""
        stream_output = stream_output or self._stream
        if lazyllm_files:
            __input = encode_query_with_filepaths(__input, lazyllm_files)
        params = {"input": __input, "history": llm_chat_history}
        if tools:
            params["tools"] = tools
        params["return_dict"] = True
        data = self._prompt.generate_prompt(**params)

        data["model"] = self._model_name
        data["stream"] = bool(stream_output)
        if len(kw) > 0:
            data.update(kw)

        if len(self._model_optional_params) > 0:
            data.update(self._model_optional_params)

        if isinstance(__input, str) and (__input.startswith(LAZYLLM_QUERY_PREFIX)
           or (self._vlm_force_format_input_with_files and data["model"] in self.vlm_models)):
            for idx, message in enumerate(data["messages"]):
                content = message["content"]
                if content.startswith(LAZYLLM_QUERY_PREFIX):
                    content = decode_query_with_filepaths(content)
                query_files = self._format_input_with_files(content)
                data["messages"][idx]["content"] = query_files

        with requests.post(self._url, json=data, headers=self._headers, stream=stream_output) as r:
            if r.status_code != 200:  # request error
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)])) \
                    if stream_output else requests.RequestException(r.text)

            if isinstance(stream_output, dict):
                prefix, prefix_color = stream_output.get('prefix', ''), stream_output.get('prefix_color', '')
                if prefix: FileSystemQueue().enqueue(lazyllm.colored_text(prefix, prefix_color))
            msg_json = list(filter(lambda x: x, ([self._str_to_json(line, stream_output) for line in r.iter_lines()
                            if len(line)] if stream_output else [self._str_to_json(r.text, stream_output)]),))
            if isinstance(stream_output, dict):
                suffix, suffix_color = stream_output.get('suffix', ''), stream_output.get('suffix_color', '')
                if suffix: FileSystemQueue().enqueue(lazyllm.colored_text(suffix, suffix_color))

            usage = {"prompt_tokens": -1, "completion_tokens": -1}
            if len(msg_json) > 0 and "usage" in msg_json[-1] and isinstance(msg_json[-1]["usage"], dict):
                for k in usage:
                    usage[k] = msg_json[-1]["usage"].get(k, usage[k])
            self._record_usage(usage)
            extractor = self._extract_specified_key_fields(
                self._merge_stream_result(msg_json)
            )
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

    def _set_template(self, template_message=None, keys_name_handle=None, template_headers=None):
        self.template_message = template_message
        self.keys_name_handle = keys_name_handle
        self.template_headers = template_headers

    def _upload_train_file(self, train_file) -> str:
        raise NotImplementedError(f"{self._model_series} not implemented _upload_train_file method in subclass")

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_series} not implemented _create_finetuning_job method in subclass")

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        raise NotImplementedError(f"{self._model_series} not implemented _query_finetuning_job method in subclass")

    def _query_finetuned_jobs(self) -> dict:
        raise NotImplementedError(f"{self._model_series} not implemented _query_finetuned_jobs method in subclass")

    def _get_finetuned_model_names(self) -> (List[str], List[str]):
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
            delete_old_files(save_dir)
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
    def _format_input_with_files(self, query_files: str) -> List[Dict[str, str]]:
        if isinstance(query_files, str):
            return self._format_vl_chat_query(query_files)
        assert isinstance(query_files, dict), "query_files must be a dict."
        output = [{"type": "text", "text": query_files["query"]}]
        files = query_files.get("files", [])
        assert isinstance(files, list), "files must be a list."
        for file in files:
            mime = None
            if not file.startswith("http"):
                file, mime = image_to_base64(file)
            output.extend(self._format_vl_chat_image_url(file, mime))
        return output

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineChat', name=self._module_name, url=self._base_url,
                                 stream=bool(self._stream), return_trace=self._return_trace)
