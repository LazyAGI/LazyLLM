import time
from urllib.parse import urljoin
import requests
import lazyllm
from ..services import ClientBase

class InferClient(ClientBase):
    def __init__(self, url):
        super().__init__(urljoin(url, 'v1/deploy/'))

    def deploy(self, base_model: str, token: str, num_gpus: int = 1):
        """
        Start a new deploy job on the LazyLLM infer service.

        This method sends a request to the LazyLLM API to launch a deploy job with the specified configuration.

        Parameters:
        - base_model(str): A dictionary containing the deploy configuration details.
        - num_gpus(int): Number of gpus to use.
        - token (str): The user group token required for authentication.

        Returns:
        - tuple: A tuple containing the job ID and the current status of the training job if the request is successful.
        - tuple: A tuple containing `None` and an error message if the request fails.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        url = urljoin(self.url, 'jobs')
        headers = {
            'Content-Type': 'application/json',
            'token': token,
        }
        deploy_config = dict(deploy_model=base_model, num_gpus=num_gpus)

        try:
            response = requests.post(url, headers=headers, json=deploy_config)
            response.raise_for_status()
            res = response.json()
            return (res['job_id'], self.uniform_status(res['status']))
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return (None, str(e))

    def cancel(self, token, job_id):
        """
        Cancel a deploy job on the LazyLLM infer service.

        This method sends a request to the LazyLLM API to cancel a specific deploy job.

        Parameters:
        - token (str): The user group token required for authentication.
        - job_id (str): The unique identifier of the deploy job to be cancelled.

        Returns:
        - bool: True if the job was successfully cancelled, otherwise an error message is returned.

        Raises:
        - Exception: If an error occurs during the request, it will be logged and an error message will be returned.
        """
        url = urljoin(self.url, f'jobs/{job_id}/cancel')
        try:
            response = requests.post(url, headers={'token': token})
            response.raise_for_status()
            status = response.json()['status']
            if status == 'Cancelled':
                return True
            else:
                return f'Failed to cancel task. Final status is {status}'
        except Exception as e:
            status = str(e)
            lazyllm.LOG.error(str(e))
            return f'Failed to cancel task. Because: {str(e)}'

    def list_all_tasks(self, token):
        """
        List all models with their job-id, model-name and statuse for the LazyLLM infer service.

        Parameters:
        - token (str): The user group token required for authentication.

        Returns:
        - list of lists: Each sublist contains [job_id, model_name, status] for each deploy task.
        - None: If the request fails or an error occurs.

        Raises:
        - Exception: If an error occurs during the request, it will be logged.
        """
        url = urljoin(self.url, 'jobs')
        headers = {'token': token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_data = response.json()
            res = list()
            for job_id, job in model_data.items():
                res.append([job_id, job['base_model'], job['status']])
            return res
        except Exception as e:
            lazyllm.LOG.error(f'Failed to get log. Because: {e}')
            return None

    def get_infra_handle(self, token, job_id):
        """
        This method retrieves and processes infrastructure details for a specific job based on the provided job ID.
        It validates the job status and prepares a deployable module handle.

        Parameters:
        - token(str): A string representing the authentication token required to access the job details.
        - job_id(str): The unique identifier for the job whose details need to be retrieved.

        Returns:
        - An instance of `lazyllm.TrainableModule` used to infer.

        Raises:
        - requests.exceptions.HTTPError: If the HTTP request to fetch job details fails.
        - RuntimeError: If the job's status is not 'Running'.
        """
        response = requests.get(urljoin(self.url, f'jobs/{job_id}'), headers={'token': token})
        response.raise_for_status()
        response = response.json()
        base_model, url, deploy_method = response['base_model'], response['url'], response['deploy_method']
        if self.uniform_status(response['status']) != 'Ready':
            raise RuntimeError(f'Job {job_id} is not running now')
        if not (deployer := getattr(lazyllm.deploy, deploy_method, None)):
            deployer = type(lazyllm.deploy.auto(base_model))
        return lazyllm.TrainableModule(base_model).deploy_method(deployer, url=url)

    def wait_ready(self, token, job_id, timeout=1800):
        """
        This method to wait for a specific job based on the provided job ID.

        Parameters:
        - token(str): A string representing the authentication token required to access the job details.
        - job_id(str): The unique identifier for the job whose details need to be retrieved.

        Returns:
        - infer service status.
        """
        def get_status():
            response = requests.get(urljoin(self.url, f'jobs/{job_id}'), headers={'token': token})
            response.raise_for_status()
            response = response.json()
            return self.uniform_status(response['status'])

        n = 0
        while (status := get_status()) != 'Ready':
            if status in ('Invalid', 'Cancelled', 'Failed'):
                raise RuntimeError(f'Deploy service failed. status is {status}')
            if n > timeout: raise TimeoutError('Inference service has not started after 1800 seconds.')
            time.sleep(10)
            n += 10
