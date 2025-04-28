import json
import os
import requests
from typing import Tuple, List
from urllib.parse import urljoin
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase
from .fileHandler import FileHandlerBase

class QwenModule(OnlineChatModuleBase, FileHandlerBase):
    """
    #TODO: The Qianwen model has been finetuned and deployed successfully,
           but it is not compatible with the OpenAI interface and can only
           be accessed through the Dashscope SDK.
    """
    TRAINABLE_MODEL_LIST = ["qwen-turbo", "qwen-7b-chat", "qwen-72b-chat"]
    MODEL_NAME = "qwen-plus"

    def __init__(self,
                 base_url: str = "https://dashscope.aliyuncs.com/",
                 model: str = None,
                 api_key: str = None,
                 stream: bool = True,
                 return_trace: bool = False,
                 **kwargs):
        OnlineChatModuleBase.__init__(self,
                                      model_series="QWEN",
                                      api_key=api_key or lazyllm.config['qwen_api_key'],
                                      base_url=base_url,
                                      model_name=model or lazyllm.config['qwen_model_name'] or QwenModule.MODEL_NAME,
                                      stream=stream,
                                      trainable_models=QwenModule.TRAINABLE_MODEL_LIST,
                                      return_trace=return_trace,
                                      **kwargs)
        FileHandlerBase.__init__(self)
        self._deploy_paramters = dict()
        if stream:
            self._model_optional_params['incremental_output'] = True
        self.default_train_data = {
            "model": "qwen-turbo",
            "training_file_ids": None,
            "validation_file_ids": None,
            "training_type": "efficient_sft",  # sft or efficient_sft
            "hyper_parameters": {
                "n_epochs": 1,
                "batch_size": 16,
                "learning_rate": "1.6e-5",
                "split": 0.9,
                "warmup_ratio": 0.0,
                "eval_steps": 1,
                "lr_scheduler_type": "linear",
                "max_length": 2048,
                "lora_rank": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
        }
        self.fine_tuning_job_id = None

    def _get_system_prompt(self):
        return ("You are a large-scale language model from Alibaba Cloud, "
                "your name is Tongyi Qianwen, and you are a useful assistant.")

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'compatible-mode/v1/chat/completions')

    def _convert_file_format(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = {"messages": []}
            messages = ex.get("messages", [])
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role in ["system", "user", "assistant"]:
                    lineEx["messages"].append({"role": role, "content": content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return "\n".join(json_strs)

    def _upload_train_file(self, train_file):
        headers = {
            "Authorization": "Bearer " + self._api_key
        }

        url = urljoin(self._base_url, "api/v1/files")

        self.get_finetune_data(train_file)

        file_object = {
            # The correct format should be to pass in a tuple in the format of:
            # (<fileName>, <fileObject>, <Content-Type>),
            # where fileObject refers to the specific value.
            "files": (os.path.basename(train_file), self._dataHandler, "application/json"),
            "descriptions": (None, "training file", None)
        }

        with requests.post(url, headers=headers, files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            if "data" not in r.json().keys():
                raise ValueError("No data found in response")
            if "uploaded_files" not in r.json()["data"].keys():
                raise ValueError("No uploaded_files found in response")
            # delete temporary training file
            self._dataHandler.close()
            return r.json()['data']['uploaded_files'][0]["file_id"]

    def _update_kw(self, data, normal_config):
        current_train_data = self.default_train_data.copy()
        current_train_data.update(data)

        current_train_data["hyper_parameters"]["n_epochs"] = normal_config["num_epochs"]
        current_train_data["hyper_parameters"]["learning_rate"] = str(normal_config["learning_rate"])
        current_train_data["hyper_parameters"]["lr_scheduler_type"] = normal_config["lr_scheduler_type"]
        current_train_data["hyper_parameters"]["batch_size"] = normal_config["batch_size"]
        current_train_data["hyper_parameters"]["max_length"] = normal_config["cutoff_len"]
        current_train_data["hyper_parameters"]["lora_rank"] = normal_config["lora_r"]
        current_train_data["hyper_parameters"]["lora_alpha"] = normal_config["lora_alpha"]

        return current_train_data

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = urljoin(self._base_url, "api/v1/fine-tunes")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": train_model,
            "training_file_ids": [train_file_id]
        }
        if "training_parameters" in kw.keys():
            data.update(kw["training_parameters"])
        elif 'finetuning_type' in kw:
            data = self._update_kw(data, kw)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()["output"]["job_id"]
            self.fine_tuning_job_id = fine_tuning_job_id
            status = r.json()["output"]["status"]
            return (fine_tuning_job_id, status)

    def _cancel_finetuning_job(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            return 'Invalid'
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = urljoin(self._base_url, f"api/v1/fine-tunes/{job_id}/cancel")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        with requests.post(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        status = r.json()['output']['status']
        if status == 'success':
            return 'Cancelled'
        else:
            return f'JOB {job_id} status: {status}'

    def _query_finetuned_jobs(self):
        fine_tune_url = urljoin(self._base_url, "api/v1/fine-tunes")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _get_finetuned_model_names(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        model_data = self._query_finetuned_jobs()
        res = list()
        if 'jobs' not in model_data['output']:
            return res
        for model in model_data['output']['jobs']:
            status = self._status_mapping(model['status'])
            if status == 'Done':
                model_id = model['finetuned_output']
            else:
                model_id = model['model'] + '-' + model['job_id']
            res.append([model['job_id'], model_id, status])
        return res

    def _status_mapping(self, status):
        if status == 'SUCCEEDED':
            return 'Done'
        elif status == 'FAILED':
            return 'Failed'
        elif status in ('CANCELING', 'CANCELED'):
            return 'Cancelled'
        elif status == 'RUNNING':
            return 'Running'
        else:  # PENDING, QUEUING
            return 'Pending'

    def _query_job_status(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError("No job ID specified. Please ensure that a valid 'fine_tuning_job_id' is "
                               "provided as an argument or started a training job.")
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        _, status = self._query_finetuning_job(job_id)
        return self._status_mapping(status)

    def _get_log(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError("No job ID specified. Please ensure that a valid 'fine_tuning_job_id' is "
                               "provided as an argument or started a training job.")
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = urljoin(self._base_url, f"api/v1/fine-tunes/{job_id}/logs")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return job_id, r.json()

    def _get_curr_job_model_id(self):
        if not self.fine_tuning_job_id:
            return None, None
        model_id, _ = self._query_finetuning_job(self.fine_tuning_job_id)
        return self.fine_tuning_job_id, model_id

    def _query_finetuning_job_info(self, fine_tuning_job_id):
        fine_tune_url = urljoin(self._base_url, f"api/v1/fine-tunes/{fine_tuning_job_id}")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()['output']

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        status = info['status']
        # QWen only status == 'SUCCEEDED' can have `finetuned_output`
        if 'finetuned_output' in info:
            fine_tuned_model = info["finetuned_output"]
        else:
            fine_tuned_model = info["model"] + '-' + info["job_id"]
        return (fine_tuned_model, status)

    def _query_finetuning_cost(self, fine_tuning_job_id):
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        if 'usage' in info and info['usage']:
            return info['usage']
        else:
            return None

    def set_deploy_parameters(self, **kw):
        self._deploy_paramters = kw

    def _create_deployment(self) -> Tuple[str, str]:
        url = urljoin(self._base_url, "api/v1/deployments")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model_name": self._model_name,
            "capacity": self._deploy_paramters.get("capcity", 2)
        }
        if self._deploy_paramters and len(self._deploy_paramters) > 0:
            data.update(self._deploy_paramters)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            deployment_id = r.json()["output"]["deployed_model"]
            status = r.json()["output"]["status"]
            return (deployment_id, status)

    def _query_deployment(self, deployment_id) -> str:
        fine_tune_url = urljoin(self._base_url, f"api/v1/deployments/{deployment_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["output"]['status']
            return status

    def _format_vl_chat_image_url(self, image_url, mime):
        assert mime is not None, "Qwen Module requires mime info."
        image_url = f"data:{mime};base64,{image_url}"
        return [{"type": "image_url", "image_url": {"url": image_url}}]
