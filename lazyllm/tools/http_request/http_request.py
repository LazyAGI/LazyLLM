import io
import re
import time
import zipfile
from typing import Optional, Callable

import requests
from lazyllm.thirdparty import httpx
import json
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
from lazyllm import LOG

class HttpRequest(ModuleBase):
    def __init__(self, method, url, api_key, headers, params, body, timeout=10, proxies=None):
        super().__init__()
        if not url:
            return

        self._method = method
        self._url = url
        self._api_key = api_key
        self._headers = headers
        self._params = params
        self._body = body
        self._timeout = timeout
        self._proxies = proxies

    def _process_api_key(self, headers, params):
        if self._api_key and self._api_key != '':
            params = params or {}
            params['api_key'] = self._api_key
        return headers, params

    def forward(self, *args, **kwargs):
        def _map_input(target_str):
            if not isinstance(target_str, str):
                return target_str

            # TODO: replacements could be more complex to create.
            replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
            if not replacements:
                return target_str

            pattern = r'\{\{([^}]+)\}\}'

            full_match = re.fullmatch(pattern, target_str)
            if full_match:
                key = full_match.group(1)
                if key in replacements:
                    return replacements[key]

            def replacer(m):
                key = m.group(1)
                if key not in replacements:
                    return m.group(0)  # Keep original if no replacement found
                replacement = replacements[key]
                if isinstance(replacement, (dict, list, bool)) or replacement is None:
                    return json.dumps(replacement, ensure_ascii=False)
                return str(replacement)

            return re.sub(pattern, replacer, target_str)

        url = _map_input(self._url)
        params = {key: _map_input(value) for key, value in self._params.items()} if self._params else None
        headers = {key: _map_input(value) for key, value in self._headers.items()} if self._headers else None
        headers, params = self._process_api_key(headers, params)
        if isinstance(headers, dict) and headers.get('Content-Type') == 'application/json':
            try:
                body = json.dumps(self._body) if isinstance(self._body, dict) else self._body
                body = json.loads(_map_input(body))

                http_response = httpx.request(method=self._method, url=url, headers=headers,
                                              params=params, json=body, timeout=self._timeout,
                                              proxies=self._proxies)
            except json.JSONDecodeError:
                raise ValueError(f'Invalid JSON format: {self._body}')
        else:
            body = (json.dumps({k: _map_input(v) for k, v in self._body.items()})
                    if isinstance(self._body, dict) else _map_input(self._body))

            http_response = httpx.request(method=self._method, url=url, headers=headers,
                                          params=params, data=body, timeout=self._timeout,
                                          proxies=self._proxies)

        response = HttpExecutorResponse(http_response)

        _, file_binary = response.extract_file()

        outputs = {
            'status_code': response.status_code,
            'content': response.content if len(file_binary) == 0 else None,
            'headers': response.headers,
            'file': file_binary
        }
        return outputs


def post_sync(url: str, payload: dict = None, files: dict = None, headers: dict = None,
              json_payload: dict = None, timeout: Optional[int] = None,
              raise_for_status: bool = True) -> requests.Response:
    """Execute a synchronous POST request with unified error handling."""
    try:
        if json_payload is not None:
            resp = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
        elif files is not None:
            resp = requests.post(url, data=payload, files=files, timeout=timeout)
        else:
            resp = requests.post(url, data=payload, timeout=timeout)
        if raise_for_status:
            resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        LOG.error(f'[HttpRequest] POST request to {url} failed: {e}')
        raise


def get_sync(url: str, headers: dict = None, timeout: Optional[int] = None,
             raise_for_status: bool = True) -> requests.Response:
    """Execute a synchronous GET request with unified error handling."""
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if raise_for_status:
            resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        LOG.error(f'[HttpRequest] GET request to {url} failed: {e}')
        raise


def post_async(submit_url: str, status_url: str, result_url: str = None,
               payload: dict = None, files: dict = None, headers: dict = None,
               timeout: Optional[int] = None,
               success_states: tuple = ('completed', 'done', 'success'),
               failure_states: tuple = ('failed', 'error', 'failure'),
               max_retries: int = 120, interval: int = 3,
               result_extractor: Optional[Callable[[requests.Response], any]] = None) -> any:
    """Submit an async task, poll status, and fetch the final result.

    Args:
        submit_url: URL to submit the task.
        status_url: Status polling URL containing ``{task_id}`` placeholder.
        result_url: Optional result URL containing ``{task_id}`` placeholder.
        result_extractor: Optional callable to extract result from the status
            response when ``result_url`` is not provided.
    """
    resp = post_sync(submit_url, payload=payload, files=files, headers=headers,
                     timeout=timeout, raise_for_status=False)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get('task_id')
    if not task_id and 'data' in data:
        task_id = data['data'].get('task_id')
    if not task_id:
        raise ValueError(f'[HttpRequest] No task_id in submit response: {data}')

    for _ in range(max_retries):
        status_resp = get_sync(status_url.format(task_id=task_id), headers=headers,
                               timeout=timeout, raise_for_status=False)
        if status_resp.status_code == 404:
            raise RuntimeError(f'[HttpRequest] Status endpoint 404: {status_url.format(task_id=task_id)}')
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get('status', status_data.get('state', ''))
        if 'data' in status_data and isinstance(status_data['data'], dict):
            status = status_data['data'].get('state', status)
        if status in success_states:
            if result_url:
                return get_sync(result_url.format(task_id=task_id), headers=headers, timeout=timeout)
            if result_extractor:
                return result_extractor(status_resp)
            return status_resp
        if status in failure_states:
            raise RuntimeError(f'[HttpRequest] Task failed: {status_data}')
        time.sleep(interval)

    raise TimeoutError(f'[HttpRequest] Task polling timed out after {max_retries * interval}s')
