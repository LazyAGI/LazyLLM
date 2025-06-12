import copy
import uuid
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import List, Dict, Optional, Set, Union

import lazyllm
from lazyllm import once_wrapper
from .engine import Engine, Node, ServerGraph, SharedHttpTool
from lazyllm.tools.train_service.serve import TrainServer
from lazyllm.tools.train_service.client import LocalTrainClient, OnlineTrainClient
from lazyllm.tools.infer_service import InferClient, InferServer


@contextmanager
def set_resources(resource):
    lazyllm.globals['engine_resource'] = {r['id']: r for r in resource}
    try:
        yield
    finally:
        lazyllm.globals.pop('engine_resource', None)


class LightEngine(Engine):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not LightEngine._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @once_wrapper
    def __init__(self):
        super().__init__()
        self.node_graph: Set[str, List[str]] = dict()
        self.online_train_client = OnlineTrainClient()

    @once_wrapper
    def launch_localllm_train_service(self):
        train_server = TrainServer()
        self._local_serve = lazyllm.ServerModule(train_server, launcher=lazyllm.launcher.EmptyLauncher(sync=False))
        self._local_serve.start()()
        parsed_url = urlparse(self._local_serve._url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self.local_train_client = LocalTrainClient(base_url)

    @once_wrapper
    def launch_localllm_infer_service(self):
        self._infer_server = lazyllm.ServerModule(InferServer(), launcher=lazyllm.launcher.EmptyLauncher(sync=False))
        self._infer_server.start()()
        parsed_url = urlparse(self._infer_server._url)
        self.infer_client = InferClient(f"{parsed_url.scheme}://{parsed_url.netloc}")

    # Local
    def local_model_train(self, train_config, token):
        """
        Start a new training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to launch a training job with the specified configuration.

        Parameters:
        - train_config (dict): A dictionary containing the training configuration details.
        - token (str): The user group token required for authentication.

        Returns:
        - tuple: A tuple containing the job ID and the current status of the training job if the request is successful.
        - tuple: A tuple containing `None` and an error message if the request fails.

        The training configuration dictionary should include the following keys:
        - finetune_model_name: The name of the model to be fine-tuned.
        - base_model: The base model to use for traning.
        - data_path: The path to the training data.
        - training_type: The type of training (e.g., 'sft').
        - finetuning_type: The type of finetuning (e.g., 'lora').
        - val_size: The ratio of validation data set to training data set.
        - num_gpus: The number of gpus, default: 1.
        - num_epochs: The number of training epochs.
        - learning_rate: The learning rate for training.
        - lr_scheduler_type: The type of learning rate scheduler.
        - batch_size: The batch size for training.
        - cutoff_len: The maximum sequence length for training.
        - lora_r: The LoRA rank.
        - lora_alpha: The LoRA alpha parameter.
        - lora_rate: The parameter ratio for LoRA fine-tuning.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.train(train_config, token)

    def local_model_cancel_training(self, token, job_id):
        """
        Cancel a training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to cancel a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job to be cancelled.

        Returns:
        - bool: True if the job was successfully cancelled, otherwise an error message is returned.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.cancel_training(token, job_id)

    def local_model_get_training_status(self, token, job_id):
        """
        Retrieve the current status of a training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to fetch the current status of a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the status.

        Returns:
        - str: The current status of the training job if the request is successful.
        - 'Invalid' (str): If the request fails or an error occurs.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.get_training_status(token, job_id)

    def local_model_get_training_log(self, token, job_id):
        """
        Retrieve the log for the current training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to fetch the log associated with a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the log.

        Returns:
        - str: The log path if the request is successful.
        - None: If the request fails or an error occurs.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.get_training_log(token, job_id)

    def local_model_get_all_trained_models(self, token):
        """
        List all models with their job-id, model-id and statuse for the LazyLLM training service.

        Parameters:
        - token (str): The user group token required for authentication.

        Returns:
        - list of lists: Each sublist contains [job_id, model_name, status] for each trained model.
        - None: If the request fails or an error occurs.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.get_all_trained_models(token)

    def local_model_get_training_cost(self, token, job_id):
        """
        Retrieve the GPU usage time for a training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to fetch the GPU usage time (in seconds)
        for a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the GPU usage time.

        Returns:
        - int: The GPU usage time in seconds if the request is successful.
        - str: An error message if the request fails.
        """
        if not self.launch_localllm_train_service.flag:
            raise RuntimeError("Please call the member function 'launch_localllm_train_service' "
                               "of the LightEngine instance to start the training service.")
        return self.local_train_client.get_training_cost(token, job_id)

    # Online
    def online_model_train(self, train_config, token, source):
        """
        Initiates an online training task with the specified parameters and configurations.

        Args:
        - train_config (dict): Configuration parameters for the training task.
        - token (str): API-Key provided by the supplier, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - tuple: A tuple containing the Job-ID and its status if the training starts successfully.
            If an error occurs, the Job-ID will be None, and the error message will be included.

        The training configuration dictionary should include the following keys:
        - finetune_model_name: The name of the model to be fine-tuned.
        - base_model: The base model to use for traning.
        - data_path: The path to the training data.
        - training_type: The type of training (e.g., 'sft').
        - finetuning_type: The type of finetuning (e.g., 'lora').
        - val_size: The ratio of validation data set to training data set.
        - num_epochs: The number of training epochs.
        - learning_rate: The learning rate for training.
        - lr_scheduler_type: The type of learning rate scheduler.
        - batch_size: The batch size for training.
        - cutoff_len: The maximum sequence length for training.
        - lora_r: The LoRA rank.
        - lora_alpha: The LoRA alpha parameter.
        - lora_rate: The parameter ratio for LoRA fine-tuning.
        """
        return self.online_train_client.train(train_config, token, source)

    def online_model_cancel_training(self, token, job_id, source):
        """
        Cancels an ongoing online training task by its Job-ID.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the training job to be cancelled.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - bool or str: Returns True if the training task was successfully cancelled. If the cancellation fails,
            it returns a string with the reason for the failure, including any final information about the task.
        """
        return self.online_train_client.cancel_training(token, job_id, source)

    def online_model_get_training_status(self, token, job_id, source):
        """
        Retrieves the current status of a training task by its Job-ID.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the training job to query.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - str: A string representing the current status of the training task. This could be one of:
            'Pending', 'Running', 'Done', 'Cancelled', 'Failed', or 'Invalid' if the query could not be processed.
        """
        return self.online_train_client.get_training_status(token, job_id, source)

    def online_model_get_training_log(self, token, job_id, source, target_path=None):
        """
        Retrieves the training log for a specific training task by its Job-ID and saves it to a file.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the log.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.
        - target_path (str, optional): The path where the log file should be saved. If not provided,
            the log will be saved to a temporary directory.

        Returns:
        - str or None: The path to the saved log file if the log retrieval and saving was successful.
            If an error occurs, None is returned.
        """
        return self.online_train_client.get_training_log(token, job_id, source=source, target_path=target_path)

    def online_model_get_all_trained_models(self, token, source):
        """
        Lists all model jobs with their corresponding job-id, model-id, and statuse for online training services.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - list of lists: Each sublist contains [job_id, model_name, status] for each trained model.
        - None: If the request fails or an error occurs.
        """
        return self.online_train_client.get_all_trained_models(token, source)

    def online_model_get_training_cost(self, token, job_id, source):
        """
        Retrieves the number of tokens consumed by an online traning task.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the traning job for which to retrieve the token consumption.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - int or str: The number of tokens consumed by the traning task if the query is successful.
            If an error occurs, a string containing the error message is returned.
        """
        return self.online_train_client.get_training_cost(token, job_id, source)

    def online_model_validate_api_key(self, token, source, secret_key=None):
        """
        Validates the API key for a given supplier.

        Args:
        - token (str): API-Key provided by the user, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.
        - secret_key (str): The secret key provided by the user for authentication,
            required only when the source is 'sensenova'. Default is None.

        Returns:
        - bool: True if the API key is valid, False otherwise.
        """
        return self.online_train_client.validate_api_key(token, source, secret_key)

    def deploy_model(self, token, model_name, num_gpus=1):
        return self.infer_client.deploy(model_name, token, num_gpus)

    def get_infra_handle(self, token, mid):
        return self.infer_client.get_infra_handle(token, mid)

    def build_node(self, node):
        if not isinstance(node, Node):
            if isinstance(node, str):
                if node not in self._nodes and (resource := lazyllm.globals.get('engine_resource', {}).get(node)):
                    node = resource
                else:
                    return self._nodes.get(node)
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        if node.id not in self._nodes:
            self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def release_node(self, *node_ids: Union[str, List[str]]):
        if len(node_ids) == 1 and isinstance(node_ids[0], (tuple, list)): node_ids = node_ids[0]
        for nodeid in node_ids:
            self.stop(nodeid)
            # TODO(wangzhihong): Analyze dependencies and only allow deleting nodes without dependencies
            [self._nodes.pop(id, None) for id in self.subnodes(nodeid, recursive=True)
             if id not in ('__start__', '__end__')]
            if nodeid not in ('__start__', '__end__'): self._nodes.pop(nodeid, None)

    def update_node(self, node):
        if not isinstance(node, Node):
            node = Node(id=node['id'], kind=node['kind'], name=node['name'], args=node['args'])
        self._nodes[node.id] = super(__class__, self).build_node(node)
        return self._nodes[node.id]

    def start(self, nodes, edges=[], resources=[], gid=None, name=None, _history_ids=None):
        if isinstance(nodes, str):
            assert not edges and not resources and not gid and not name
            self.build_node(nodes).func.start()
        elif isinstance(nodes, dict):
            Engine().build_node(nodes)
        else:
            gid, name = gid or str(uuid.uuid4().hex), name or str(uuid.uuid4().hex)
            node = Node(id=gid, kind='Graph', name=name, args=dict(
                nodes=copy.copy(nodes), edges=copy.copy(edges),
                resources=copy.copy(resources), _history_ids=_history_ids))
            with set_resources(resources):
                self.build_node(node).func.start()
            return gid

    def status(self, node_id: str, task_name: Optional[str] = None):
        node = self.build_node(node_id)
        if not node:
            return 'unknown'
        elif task_name:
            assert node.kind in ('LocalLLM')
            return node.func.status(task_name=task_name)
        elif subs := node.subitems:
            return {n: self.status(n) for n in subs}
        elif node.kind in ('LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA', 'web', 'server'):
            return node.func.status()
        else:
            return 'running'

    def stop(self, node_id: Optional[str] = None, task_name: Optional[str] = None):
        if not node_id:
            for node in self._nodes:
                self.release_node(node)
        elif node := self.build_node(node_id):
            if task_name:
                assert node.kind in ('LocalLLM')
                node.func.stop(task_name=task_name)
            elif node.kind in ('Graph', 'LocalLLM', 'LocalEmbedding', 'SD', 'TTS', 'STT', 'VQA'):
                node.func.stop()

    def update(self, gid_or_nodes: Union[str, Dict, List[Dict]], nodes: List[Dict],
               edges: List[Dict] = [], resources: List[Dict] = []) -> str:
        if isinstance(gid_or_nodes, str):
            assert (gid := gid_or_nodes) in self._nodes
            name = self._nodes[gid].name
            self.release_node(gid)
            self.start(nodes, edges, resources, gid_or_nodes, name=name)
        else:
            for node in gid_or_nodes: self.update_node(node)

    def run(self, id: str, *args, _lazyllm_files: Optional[Union[str, List[str]]] = None,
            _file_resources: Optional[Dict[str, Union[str, List[str]]]] = None,
            _lazyllm_history: Optional[List[List[str]]] = None, **kw):
        if files := _lazyllm_files:
            assert len(args) <= 1 and len(kw) == 0, 'At most one query is enabled when file exists'
            args = [lazyllm.formatter.file(formatter='encode')(dict(query=args[0] if args else '', files=files))]
        if _file_resources:
            lazyllm.globals['lazyllm_files'] = _file_resources
        nodes = [self.build_node(node).func for node in self.subnodes(id, recursive=True)]
        [node.valid_key() for node in nodes if isinstance(node, SharedHttpTool)]
        f = self.build_node(id).func
        lazyllm.FileSystemQueue().dequeue()
        if history := _lazyllm_history:
            assert isinstance(f, ServerGraph), 'Only graph can support history'
            if not isinstance(history, list) and all([isinstance(h, list) for h in history]):
                raise RuntimeError('History shoule be [[str, str], ..., [str, str]] (list of list of str)')
            lazyllm.globals['chat_history'] = {Engine().build_node(i).func._module_id: history for i in f._history_ids}
        result = self.build_node(id).func(*args, **kw)
        lazyllm.globals['lazyllm_files'] = {}
        lazyllm.globals['chat_history'] = {}
        return result
