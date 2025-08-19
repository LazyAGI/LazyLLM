import json
import os
import re
import requests
from typing import Tuple, List, Any, Dict
from urllib.parse import urljoin
import lazyllm
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
from ..fileHandler import FileHandlerBase
from http import HTTPStatus
from lazyllm.thirdparty import dashscope
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.formatter import encode_query_with_filepaths


class QwenModule(OnlineChatModuleBase, FileHandlerBase):
    """
    #TODO: The Qianwen model has been finetuned and deployed successfully,
           but it is not compatible with the OpenAI interface and can only
           be accessed through the Dashscope SDK.
    """
    TRAINABLE_MODEL_LIST = ["qwen-turbo", "qwen-7b-chat", "qwen-72b-chat"]
    MODEL_NAME = "qwen-plus"

    def __init__(self, base_url: str = "https://dashscope.aliyuncs.com/", model: str = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        OnlineChatModuleBase.__init__(self, model_series="QWEN", api_key=api_key or lazyllm.config['qwen_api_key'],
                                      model_name=model or lazyllm.config['qwen_model_name'] or QwenModule.MODEL_NAME,
                                      base_url=base_url, stream=stream, return_trace=return_trace, **kwargs)
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


class QwenEmbedding(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = ("https://dashscope.aliyuncs.com/api/v1/services/"
                                   "embeddings/text-embedding/text-embedding"),
                 embed_model_name: str = "text-embedding-v1",
                 api_key: str = None):
        super().__init__("QWEN", embed_url, api_key or lazyllm.config['qwen_api_key'], embed_model_name)

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": {
                "texts": [text]
            },
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['output']['embeddings'][0]['embedding']


class QwenReranking(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = ("https://dashscope.aliyuncs.com/api/v1/services/"
                                   "rerank/text-rerank/text-rerank"),
                 embed_model_name: str = "gte-rerank",
                 api_key: str = None, **kwargs):
        super().__init__("QWEN", embed_url, api_key or lazyllm.config['qwen_api_key'], embed_model_name)

    @property
    def type(self):
        return "ONLINE_RERANK"

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "top_n": top_n,
            },
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        results = response['output']['results']
        return [(result["index"], result["relevance_score"]) for result in results]


class QwenMultiModal(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://dashscope.aliyuncs.com/api/v1',
                 base_websocket_url: str = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series="QWEN",
                                      model_name=model_name, return_trace=return_trace, **kwargs)
        dashscope.api_key = lazyllm.config['qwen_api_key']
        dashscope.base_http_api_url = base_url
        dashscope.base_websocket_api_url = base_websocket_url
        self._api_key = api_key


class QwenSTTModule(QwenMultiModal):
    MODEL_NAME = "paraformer-v2"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenMultiModal.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_stt_model_name'] or QwenSTTModule.MODEL_NAME,
                                return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):  # noqa B006
        assert any(file.startswith('http') for file in files), "QwenSTTModule only supports http file urls"
        call_params = {'model': self._model_name, 'file_urls': files, **kwargs}
        if self._api_key: call_params['api_key'] = self._api_key
        task_response = dashscope.audio.asr.Transcription.async_call(**call_params)
        transcribe_response = dashscope.audio.asr.Transcription.wait(task=task_response.output.task_id,
                                                                     api_key=self._api_key)
        if transcribe_response.status_code == HTTPStatus.OK:
            result_text = ""
            for task in transcribe_response.output.results:
                assert task['subtask_status'] == "SUCCEEDED", "subtask_status is not SUCCEEDED"
                response = json.loads(requests.get(task['transcription_url']).text)
                for transcript in response['transcripts']:
                    result_text += re.sub(r"<[^>]+>", "", transcript['text'])
            return result_text
        else:
            lazyllm.LOG.error(f"failed to transcribe: {transcribe_response.output}")
            raise Exception(f"failed to transcribe: {transcribe_response.output.message}")


class QwenTextToImageModule(QwenMultiModal):
    MODEL_NAME = "wanx2.1-t2i-turbo"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenMultiModal.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_text2image_model_name']
                                or QwenTextToImageModule.MODEL_NAME, return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, negative_prompt: str = None, n: int = 1, prompt_extend: bool = True,
                 size: str = '1024*1024', seed: int = None, **kwargs):
        call_params = {
            'model': self._model_name,
            'prompt': input,
            'negative_prompt': negative_prompt,
            'n': n,
            'prompt_extend': prompt_extend,
            'size': size,
            **kwargs
        }
        if self._api_key: call_params['api_key'] = self._api_key
        if seed: call_params['seed'] = seed
        task_response = dashscope.ImageSynthesis.async_call(**call_params)
        response = dashscope.ImageSynthesis.wait(task=task_response.output.task_id, api_key=self._api_key)
        if response.status_code == HTTPStatus.OK:
            return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                    for result in response.output.results]))
        else:
            lazyllm.LOG.error(f"failed to generate image: {response.output}")
            raise Exception(f"failed to generate image: {response.output.message}")


def synthesize_qwentts(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
                       api_key: str = None, **kwargs):
    call_params = {
        'model': model_name,
        'text': input,
        'voice': voice,
        **kwargs
    }
    if api_key: call_params['api_key'] = api_key
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(**call_params)
    if response.status_code == HTTPStatus.OK:
        return requests.get(response.output['audio']['url']).content
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response}")
        raise Exception(f"failed to synthesize: {response.message}")

def synthesize(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
               api_key: str = None, **kwargs):
    assert api_key is None, f"{model_name} does not support multi user, don't set api_key"
    model_name = model_name + '-' + voice
    response = dashscope.audio.tts.SpeechSynthesizer.call(model=model_name, text=input, volume=volume,
                                                          pitch=pitch, rate=speech_rate, **kwargs)
    if response.get_response().status_code == HTTPStatus.OK:
        return response.get_audio_data()
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response.get_response()}")
        raise Exception(f"failed to synthesize: {response.get_response().message}")

def synthesize_v2(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
                  api_key: str = None, **kwargs):
    assert api_key is None, f"{model_name} does not support multi user, don't set api_key"
    synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model_name, voice=voice, volume=volume,
                                                           pitch_rate=pitch, speech_rate=speech_rate, **kwargs)
    audio = synthesizer.call(input)
    if synthesizer.last_response['header']['event'] == 'task-finished':
        return audio
    else:
        lazyllm.LOG.error(f"failed to synthesize: {synthesizer.last_response}")
        raise Exception(f"failed to synthesize: {synthesizer.last_response['header']['error_message']}")


class QwenTTSModule(QwenMultiModal):
    MODEL_NAME = "qwen-tts"
    SYNTHESIZERS = {
        "cosyvoice-v2": (synthesize_v2, 'longxiaochun_v2'),
        "cosyvoice-v1": (synthesize_v2, 'longxiaochun'),
        "sambert": (synthesize, 'zhinan-v1'),
        "qwen-tts": (synthesize_qwentts, 'Cherry'),
        "qwen-tts-latest": (synthesize_qwentts, 'Cherry')
    }

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenMultiModal.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_tts_model_name'] or QwenTTSModule.MODEL_NAME,
                                return_trace=return_trace, **kwargs)
        if self._model_name not in self.SYNTHESIZERS:
            raise ValueError(f"unsupported model: {self._model_name}. "
                             f"supported models: {QwenTTSModule.SYNTHESIZERS.keys()}")
        self._synthesizer_func, self._voice = QwenTTSModule.SYNTHESIZERS[self._model_name]

    def _forward(self, input: str = None, voice: str = None, speech_rate: float = 1.0, volume: int = 50,
                 pitch: float = 1.0, **kwargs):
        call_params = {
            "input": input,
            "model_name": self._model_name,
            "voice": voice or self._voice,
            "speech_rate": speech_rate,
            "volume": volume,
            "pitch": pitch,
            **kwargs
        }
        if self._api_key: call_params['api_key'] = self._api_key
        return encode_query_with_filepaths(None, bytes_to_file(self._synthesizer_func(**call_params)))
