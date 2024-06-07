import json
import os
import requests
from typing import Tuple
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase
from .fileHandler import FileHandlerBase

class OpenAIModule(OnlineChatModuleBase, FileHandlerBase):
    TRAINABLE_MODEL_LIST = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
                            "gpt-3.5-turbo-0613", "babbage-002",
                            "davinci-002", "gpt-4-0613"]

    def __init__(self,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-3.5-turbo",
                 system_prompt: str = "You are a helpful assistant.",
                 stream: bool = True,
                 return_trace: bool = False,
                 **kwargs):
        OnlineChatModuleBase.__init__(self,
                                      model_type=__class__.__name__,
                                      api_key=lazyllm.config['openai_api_key'],
                                      base_url=base_url,
                                      model_name=model,
                                      system_prompt=system_prompt,
                                      stream=stream,
                                      trainable_models=OpenAIModule.TRAINABLE_MODEL_LIST,
                                      return_trace=return_trace,
                                      **kwargs)
        FileHandlerBase.__init__(self)

    def _convert_file_format(self, filepath: str) -> str:
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

        url = os.path.join(self._base_url, "files")

        self.get_finetune_data(train_file)

        file_object = {
            "purpose": (None, "fine-tune", None),
            "file": (os.path.basename(train_file), self._dataHandler, "application/json")
        }

        with requests.post(url, headers=headers, files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            # delete temporary training file
            self._dataHandler.close()
            return r.json()["id"]

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = os.path.join(self._base_url, "fine_tuning/jobs")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": train_model,
            "training_file": train_file_id
        }
        if len(kw) > 0:
            data.update(kw)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()["id"]
            status = r.json()["status"]
            return (fine_tuning_job_id, status)

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        fine_tune_url = os.path.join(self._base_url, f"fine_tuning/jobs/{fine_tuning_job_id}")
        headers = {
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()['status']
            fine_tuned_model = None
            if status.lower() == "succeeded":
                fine_tuned_model = r.json()["fine_tuned_model"]
            return (fine_tuned_model, status)

    def _create_deployment(self) -> Tuple[str, str]:
        return (self._model_name, "RUNNING")

    def _query_deployment(self, deployment_id) -> str:
        return "RUNNING"
