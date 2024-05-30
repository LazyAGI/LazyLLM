import json
import os
import requests
from typing import Tuple
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase
from .fileHandler import FileHandlerBase

class GLMModule(OnlineChatModuleBase, FileHandlerBase):
    """Reasoning and fine-tuning openai interfaces using URLs"""
    TRAINABLE_MODEL_LIST = ["chatglm3-6b", "chatglm_12b", "chatglm_32b", "chatglm_66b", "chatglm_130b"]

    def __init__(self,
                 base_url: str = "https://open.bigmodel.cn/api/paas/v4",
                 model: str = "glm-4",
                 system_prompt: str = "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。",
                 stream: str = True,
                 return_trace: bool = False):
        OnlineChatModuleBase.__init__(self,
                                      model_type=__class__.__name__,
                                      api_key=lazyllm.config['glm_api_key'],
                                      base_url=base_url,
                                      model_name=model,
                                      stream=stream,
                                      system_prompt=system_prompt,
                                      trainable_models=GLMModule.TRAINABLE_MODEL_LIST,
                                      return_trace=return_trace)
        FileHandlerBase.__init__(self)

    def _get_models_list(self):
        return ["glm-4", "glm-4v", "glm-3-turbo", "chatglm-turbo", "cogview-3", "embedding-2", "text-embedding"]

    def _convert_file_format(self, filepath: str) -> str:
        """convert file format"""
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
        """
        Upload train file to server. Individual files can be up to 512 MB

        {"messages": [{"role": "system", "content": "你是一位乐于助人，知识渊博的全能AI助手。"},
                      {"role": "user", "content": "请介绍一下AlphaGo的基本原理。"},
                      {"role": "assistant", "content": "AlphaGo利用深度神经网络与树搜索算法相结合,
                                                        通过对弈数据进行训练,逐步提升游戏水平,
                                                        能够对围棋局面进行判断并选择最佳落子位置。"}]}
        {"messages": [{"role": "system", "content": "你是一位乐于助人，知识渊博的全能AI助手。"},
                      {"role": "user", "content": "请列举一些健康饮食的重要原则。"},
                      {"role": "assistant", "content": "健康饮食的重要原则包括:膳食应营养均衡,合理
                                                        摄入各种营养素;多吃新鲜蔬果,适量饮水;少吃糖、
                                                        油、盐,忌吃过量快餐;养成规律生活作息习惯,适量运动。"}]}
        {"messages": [{"role": "system", "content": "你是一位乐于助人，知识渊博的全能AI助手。"},
                      {"role": "user", "content": "请给出一些改善睡眠质量的建议。"},
                      {"role": "assistant", "content": "改善睡眠质量的建议包括:保证睡眠时间,避免睡眠不足;
                                                        睡前放松身心,远离刺激;养成规律作息时间,不要频繁熬夜;
                                                        适量运动,但不要运动过度;睡前可以喝一杯热牛奶等温和饮料。"}]}
        """
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
        """
        create fine-tuning job
        """
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
        """
        query fine-tuning job
        """
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

    def _create_deployment(self) -> Tuple[str]:
        """
        Create deployment.
        """
        return (self._model_name, "RUNNING")

    def _query_deployment(self, deployment_id) -> str:
        """
        Query deployment.
        """
        return "RUNNING"
