import os
import json
import requests
from urllib.parse import urljoin

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.utils import update_config, TrainConfig, uniform_sft_dataset


class LocalTrainClient:

    def __init__(self, url):
        self.url = url

    def uniform_status(self, status):
        if status == 'Invalid':
            res = 'Invalid'
        elif Status[status] == Status.Done:
            res = 'Done'
        elif Status[status] == Status.Cancelled:
            res = 'Cancelled'
        elif Status[status] == Status.Failed:
            res = 'Failed'
        elif Status[status] == Status.Running:
            res = 'Running'
        else:  # TBSubmitted, InQueue, Pending
            res = 'Pending'
        return res

    def train(self, train_config, token):
        """
        Start a new training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to launch a training job with the specified configuration.

        Parameters:
        - train_config (dict): A dictionary containing the training configuration details.
        - token (str): The user group token required for authentication.

        Returns:
        - tuple: A tuple containing the job ID and the current status of the training job if the request is successful.
        - tuple: A tuple containing `None` and an error message if the request fails.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.

        The training configuration dictionary should include the following keys:
        - finetune_model_name: The name of the model to be fine-tuned.
        - base_model: The base model to use for traning.
        - data_path: The path to the training data.
        - num_gpus: The number of gpus, default: 1.
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
        """
        url = urljoin(self.url, 'v1/fine_tuning/jobs')
        headers = {
            "Content-Type": "application/json",
            "token": token,
        }
        train_config = update_config(train_config, TrainConfig)
        data = {
            'finetune_model_name': train_config['finetune_model_name'],
            'base_model': train_config['base_model'],
            'data_path': train_config['data_path'],
            'num_gpus': train_config['num_gpus'],
            'hyperparameters': {
                'stage': train_config['training_type'].strip().lower(),
                'finetuning_type': train_config['finetuning_type'].strip().lower(),
                'val_size': train_config['val_size'],
                'num_train_epochs': train_config['num_epochs'],
                'learning_rate': train_config['learning_rate'],
                'lr_scheduler_type': train_config['lr_scheduler_type'],
                'per_device_train_batch_size': train_config['batch_size'],
                'cutoff_len': train_config['cutoff_len'],
                'lora_r': train_config['lora_r'],
                'lora_alpha': train_config['lora_alpha'],
            }
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res = response.json()
            return (res['job_id'], self.uniform_status(res['status']))
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return (None, str(e))

    def cancel_training(self, token, job_id):
        """
        Cancel a training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to cancel a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job to be cancelled.

        Returns:
        - bool: True if the job was successfully cancelled, otherwise an error message is returned.

        Raises:
        - Exception: If an error occurs during the request, it will be logged and an error message will be returned.
        """
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}/cancel')
        headers = {
            "token": token,
        }
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            status = response.json()['status']
            if status == 'Cancelled':
                return True
            else:
                return f"Failed to cancel task. Final status is {status}"
        except Exception as e:
            status = str(e)
            lazyllm.LOG.error(str(e))
            return f"Failed to cancel task. Because: {str(e)}"

    def get_training_cost(self, token, job_id):
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

        Raises:
        - Exception: If an error occurs during the request, it will be logged and an error message will be returned.

        """
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()['cost']
        except Exception as e:
            error = f"Failed to get cost. Because: {str(e)}"
            lazyllm.LOG.error(error)
            return error

    def get_training_status(self, token, job_id):
        """
        Retrieve the current status of a training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to fetch the current status of a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the status.

        Returns:
        - str: The current status of the training job if the request is successful.
        - 'Invalid' (str): If the request fails or an error occurs.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = self.uniform_status(response.json()['status'])
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(str(e))
        return status

    def get_training_log(self, token, job_id):
        """
        Retrieve the log for the current training job on the LazyLLM training service.

        This method sends a request to the LazyLLM API to fetch the log associated with a specific training job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the training job for which to retrieve the log.

        Returns:
        - str: The log content if the request is successful.
        - None: If the request fails or an error occurs.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}/events')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()['log']
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {str(e)}")
            return None

    def get_all_trained_models(self, token):
        """
        List all models with their job-id, model-id and statuse for the LazyLLM training service.

        Parameters:
        - token (str): The user group token required for authentication.

        Returns:
        - list of lists: Each sublist contains [job_id, model_name, status] for each trained model.
        - None: If the request fails or an error occurs.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        url = urljoin(self.url, 'v1/fine_tuning/jobs')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_data = response.json()
            res = list()
            for job_id, job in model_data.items():
                res.append([job_id, job['fine_tuned_model'], job['status']])
            return res
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {e}")
            return None

class OnlineTrainClient:

    def __init__(self):
        pass

    def train(self, train_config, token, source):
        """
        Initiates an online training task with the specified parameters and configurations.

        Args:
        - train_config (dict): Configuration parameters for the training task.
        - token (str): API-Key provided by the supplier, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - tuple: A tuple containing the Job-ID and its status if the training starts successfully.
            If an error occurs, the Job-ID will be None, and the error message will be included.

        Raises:
        - Exception: For any other errors that occur during the process, which will be logged and returned.
        """
        try:
            train_config = update_config(train_config, TrainConfig)
            assert train_config['training_type'].lower() == 'sft', 'Only supported sft!'

            data_path = os.path.join(lazyllm.config['data_path'], train_config['data_path'])
            data_path = uniform_sft_dataset(data_path, target='openai')
            m = lazyllm.OnlineChatModule(model=train_config['base_model'], api_key=token, source=source)

            file_id = m._upload_train_file(train_file=data_path)
            fine_tuning_job_id, status = m._create_finetuning_job(m._model_name, file_id, **train_config)

            return (fine_tuning_job_id, status)
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return (None, str(e))

    def get_all_trained_models(self, token, source):
        """
        Lists all model jobs with their corresponding job-id, model-id, and statuse for online training services.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - list of lists: Each sublist contains [job_id, model_name, status] for each trained model.
        - None: If the request fails or an error occurs.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            return m._get_finetuned_model_names()
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return None

    def get_training_status(self, token, job_id, source):
        """
        Retrieves the current status of a training task by its Job-ID.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the training job to query.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - str: A string representing the current status of the training task. This could be one of:
            'Pending', 'Running', 'Done', 'Cancelled', 'Failed', or 'Invalid' if the query could not be processed.

        Raises:
        - Exception: For any other errors that occur during the status query process,
            which will be logged and returned as 'Invalid'.
        """
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            status = m._query_job_status(job_id)
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(e)
        return status

    def cancel_training(self, token, job_id, source):
        """
        Cancels an ongoing online training task by its Job-ID.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the training job to be cancelled.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - bool or str: Returns True if the training task was successfully cancelled. If the cancellation fails,
            it returns a string with the reason for the failure, including any final information about the task.

        Raises:
        - Exception: For any other errors that occur during the cancellation process,
            which will be logged and returned as a string.
        """
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            res = m._cancel_finetuning_job(job_id)
            if res == 'Cancelled':
                return True
            else:
                return f"Failed to cancel task. Final info is {res}"
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return f"Failed to cancel task. Because: {str(e)}"

    def get_training_log(self, token, job_id, source, target_path=None):
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

        Raises:
        - Exception: For any other errors that occur during the log retrieval and saving process, which will be logged.
        """
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            file_name, log = m._get_log(job_id)
            save_path = target_path if target_path else os.path.join(m._get_temp_save_dir_path(), f'{file_name}.log')
            with open(save_path, 'w', encoding='utf-8') as log_file:
                json.dump(log, log_file, indent=4, ensure_ascii=False)
            return save_path
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {e}")
            return None

    def get_training_cost(self, token, job_id, source):
        """
        Retrieves the number of tokens consumed by an online traning task.

        Args:
        - token (str): API-Key provided by the supplier, used for authentication.
        - job_id (str): The unique identifier of the traning job for which to retrieve the token consumption.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - int or str: The number of tokens consumed by the traning task if the query is successful.
            If an error occurs, a string containing the error message is returned.

        Raises:
        - Exception: For any other errors that occur during the token consumption query process, which will be logged.
        """
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            res = m._query_finetuning_cost(job_id)
            return res
        except Exception as e:
            error = f"Failed to get cost. Because: {str(e)}"
            lazyllm.LOG.error(error)
            return error

    def validate_api_key(self, token, source):
        """
        Validates the API key for a given supplier.

        Args:
        - token (str): API-Key provided by the user, used for authentication.
        - source (str): Specifies the supplier. Supported suppliers are 'openai', 'glm' and 'qwen'.

        Returns:
        - bool: True if the API key is valid, False otherwise.
        """
        m = lazyllm.OnlineChatModule(source=source, api_key=token)
        return m._validate_api_key()
