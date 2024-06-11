import json
import os
import requests
from typing import Tuple
import uuid
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase
from .fileHandler import FileHandlerBase

class SenseNovaModule(OnlineChatModuleBase, FileHandlerBase):
    TRAINABLE_MODEL_LIST = ["nova-ptc-s-v2"]

    def __init__(self,
                 base_url="https://api.sensenova.cn/v1/llm",
                 model="SenseChat-5",
                 system_prompt="You are an AI assistant whose name is InternLM (书生·浦语).",
                 stream=True,
                 return_trace=False,
                 **kwargs):
        OnlineChatModuleBase.__init__(self,
                                      model_type=__class__.__name__,
                                      api_key=SenseNovaModule.encode_jwt_token(lazyllm.config['sensenova_ak'],
                                                                               lazyllm.config['sensenova_sk']),
                                      base_url=base_url,
                                      model_name=model,
                                      stream=stream,
                                      system_prompt=system_prompt,
                                      trainable_models=SenseNovaModule.TRAINABLE_MODEL_LIST,
                                      return_trace=return_trace,
                                      **kwargs)
        FileHandlerBase.__init__(self)
        self._deploy_paramters = None

    @staticmethod
    def encode_jwt_token(ak: str, sk: str) -> str:
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        import time
        payload = {
            "iss": ak,
            # Fill in the expected effective time, which represents the current time +24 hours
            "exp": int(time.time()) + 86400,
            # Fill in the desired effective time starting point, which represents the current time
            "nbf": int(time.time())
        }
        import jwt
        token = jwt.encode(payload, sk, headers=headers)
        return token

    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'chat-completions')

    def _parse_response_stream(self, response: str) -> str:
        chunk = response.decode('utf-8')[5:]
        try:
            chunk = json.loads(chunk)["data"]
            content = chunk['choices'][0]['delta']
            chunk['choices'][0]['delta'] = {"content": content}
            return json.dumps(chunk, ensure_ascii=False)
        except Exception:
            return chunk

    def _parse_response_non_stream(self, response: str) -> str:
        cur_msg = json.loads(response)['data']["choices"][0]
        content = cur_msg.get("message", "")
        msg = {"role": cur_msg["role"], "content": content}
        cur_msg.pop("role")
        cur_msg['message'] = msg
        return cur_msg

    def _convert_file_format(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = []
            messages = ex.get("messages", [])
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role in ["system", "knowledge", "user", "assistant"]:
                    lineEx.append({"role": role, "content": content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return "\n".join(json_strs)

    def _upload_train_file(self, train_file):
        headers = {
            "Authorization": "Bearer " + self._api_key
        }
        url = self._train_parameters.get("upload_url", "https://file.sensenova.cn/v1/files")
        self.get_finetune_data(train_file)
        file_object = {
            # The correct format should be to pass in a tuple in the format of:
            # (<fileName>, <fileObject>, <Content-Type>),
            # where fileObject refers to the specific value.

            "description": (None, "train_file", None),
            "scheme": (None, "FINE_TUNE_2", None),
            "file": (os.path.basename(train_file), self._dataHandler, "application/json")
        }

        train_file_id = None
        with requests.post(url, headers=headers, files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException(r.text)

            train_file_id = r.json()["id"]
            # delete temporary training file
            self._dataHandler.close()
            lazyllm.LOG.info(f"train file id: {train_file_id}")

        def _create_finetuning_dataset(description, files):
            url = os.path.join(self._base_url, "fine-tune/datasets")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            data = {
                "description": description,
                "files": files
            }
            with requests.post(url, headers=headers, json=data) as r:
                if r.status_code != 200:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

                dataset_id = r.json()["dataset"]["id"]
                status = r.json()["dataset"]["status"]
                import time
                url = url + f"/{dataset_id}"
                while status.lower() != "ready":
                    try:
                        time.sleep(10)
                        with requests.get(url, headers=headers) as r:
                            if r.status_code != 200:
                                raise requests.RequestException(r.text)

                            dataset_id = r.json()["dataset"]["id"]
                            status = r.json()["dataset"]["status"]
                    except Exception as e:
                        lazyllm.LOG.error(f"error: {e}")
                        raise ValueError(f"created datasets {dataset_id} failed")
                return dataset_id

        return _create_finetuning_dataset("fine-tuning dataset", [train_file_id])

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = os.path.join(self._base_url, "fine-tunes")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": train_model,
            "training_file": train_file_id,
            "suffix": kw.get("suffix", "ft-" + str(uuid.uuid4().hex))
        }
        if "training_parameters" in kw.keys():
            data.update(kw["training_parameters"])

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException(r.text)

            fine_tuning_job_id = r.json()["job"]["id"]
            status = r.json()["job"]["status"]
            return (fine_tuning_job_id, status)

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        fine_tune_url = os.path.join(self._base_url, f"fine-tunes/{fine_tuning_job_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["job"]['status']
            fine_tuned_model = None
            if status.lower() == "succeeded":
                fine_tuned_model = r.json()["job"]["fine_tuned_model"]
            return (fine_tuned_model, status)

    def set_deploy_parameters(self, **kw):
        self._deploy_paramters = kw

    def _create_deployment(self) -> Tuple[str, str]:
        url = os.path.join(self._base_url, "fine-tune/servings")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": self._model_name,
            "config": {
                "run_time": 0
            }
        }
        if self._deploy_paramters and len(self._deploy_paramters) > 0:
            data.update(self._deploy_paramters)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()["job"]["id"]
            status = r.json()["job"]["status"]
            return (fine_tuning_job_id, status)

    def _query_deployment(self, deployment_id) -> str:
        fine_tune_url = os.path.join(self._base_url, f"fine-tune/servings/{deployment_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["job"]['status']
            return status
