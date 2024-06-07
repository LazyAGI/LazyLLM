import json
import os
import requests
from typing import Tuple
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

    def __init__(self,
                 base_url: str = "https://dashscope.aliyuncs.com",
                 model: str = "qwen-plus",
                 system_prompt: str = "You are a helpful assistant.",
                 stream: bool = True,
                 return_trace: bool = False,
                 **kwargs):
        OnlineChatModuleBase.__init__(self,
                                      model_type=__class__.__name__,
                                      api_key=lazyllm.config['qwen_api_key'],
                                      base_url=base_url,
                                      model_name=model,
                                      system_prompt=system_prompt,
                                      stream=stream,
                                      trainable_models=QwenModule.TRAINABLE_MODEL_LIST,
                                      return_trace=return_trace,
                                      **kwargs)
        FileHandlerBase.__init__(self)
        self._deploy_paramters = None

    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'compatible-mode/v1/chat/completions')

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

        url = os.path.join(self._base_url, "api/v1/files")

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

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = os.path.join(self._base_url, "api/v1/fine-tunes")
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

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()["output"]["job_id"]
            status = r.json()["output"]["status"]
            return (fine_tuning_job_id, status)

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        fine_tune_url = os.path.join(self._base_url, f"api/v1/fine-tunes/{fine_tuning_job_id}")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["output"]['status']
            fine_tuned_model = None
            if status.lower() == "succeeded":
                fine_tuned_model = r.json()["output"]["finetuned_output"]
            return (fine_tuned_model, status)

    def set_deploy_parameters(self, **kw):
        self._deploy_paramters = kw

    def _create_deployment(self) -> Tuple[str, str]:
        url = os.path.join(self._base_url, "api/v1/deployments")
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
        fine_tune_url = os.path.join(self._base_url, f"api/v1/deployments/{deployment_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["output"]['status']
            return status
