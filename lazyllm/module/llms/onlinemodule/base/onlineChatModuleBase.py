from itertools import groupby
import json
import os
import requests
import re
import random
import time
import uuid
from typing import Tuple, List, Dict, Union, Any, Optional
from urllib.parse import urljoin
from operator import itemgetter as itemget

import lazyllm
from lazyllm import globals, pipeline, config
from lazyllm.components.utils.file_operate import _delete_old_files, _image_to_base64
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ....servermodule import LLMBase, StaticParams
from .utils import LazyLLMOnlineBase, resolve_online_params


config.add('online_chat_transport_max_retries', int, 5, 'ONLINE_CHAT_TRANSPORT_MAX_RETRIES',
           description='The number of extra online chat attempts allowed for transient transport failures.')
config.add('online_chat_transport_retry_base_delay_s', float, 1.0, 'ONLINE_CHAT_TRANSPORT_RETRY_BASE_DELAY_S',
           description='The base delay in seconds for online chat transport retries.')
config.add('online_chat_transport_retry_max_delay_s', float, 16.0, 'ONLINE_CHAT_TRANSPORT_RETRY_MAX_DELAY_S',
           description='The maximum delay in seconds for online chat transport retries.')
config.add('online_chat_transport_retry_jitter_ratio', float, 0.2, 'ONLINE_CHAT_TRANSPORT_RETRY_JITTER_RATIO',
           description='The jitter ratio applied to online chat transport retry delays.')


class _ProviderResponseError(requests.RequestException):
    def __init__(self, message: str, *, http_status: Optional[int], error_body: Optional[str]):  # noqa B042
        super().__init__(message)
        self.http_status = http_status
        self.error_body = error_body


class _ProviderHTTPError(_ProviderResponseError): pass
class _ProviderProtocolError(_ProviderResponseError): pass


class LazyLLMOnlineChatModuleBase(LazyLLMOnlineBase, LLMBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_PREFIX = []
    NO_PROXY = True
    __lazyllm_registry_key__ = LLMType.CHAT
    _message_format = 'openai'
    _openai_compatible_response = True

    def __init__(self, api_key: Union[str, List[str]], base_url: str, model_name: str,
                 stream: Union[bool, Dict[str, str]], return_trace: bool = False, skip_auth: bool = False,
                 static_params: Optional[StaticParams] = None, type: Optional[str] = None,
                 timeout: Optional[Union[int, Tuple[int, int]]] = 180, **kwargs):
        if any([model_name.startswith(prefix) for prefix in self.VLM_MODEL_PREFIX]):
            if type is None: type = LLMType.VLM
            else: assert type == LLMType.VLM, f'model_name {model_name} is a VLM model, but type is {type}'
        super().__init__(api_key=api_key, skip_auth=skip_auth, return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, type=type, static_params=static_params)
        self.__base_url = base_url
        self._model_name = model_name
        self.trainable_models = self.TRAINABLE_MODEL_LIST
        self._is_trained = False
        self._model_optional_params = {}
        self._vlm_force_format_input_with_files = False
        self._timeout = timeout

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        super().prompt('' if prompt is None else prompt, history=history)
        if not config['disable_system_prompt']:
            self._prompt._set_model_configs(system=self._get_system_prompt())
        return self

    def _get_system_prompt(self):
        raise NotImplementedError('_get_system_prompt is not implemented.')

    @property
    def _base_url(self):
        return random.choice(self.__base_url) if isinstance(self.__base_url, list) else self.__base_url

    @property
    def _chat_url(self):
        return self._get_chat_url(self._base_url)

    def _get_chat_url(self, url):
        if url.rstrip('/').endswith('chat/completions'):
            return url
        return urljoin(url, 'chat/completions')

    def _get_models_list(self):
        url = urljoin(self._base_url, 'models')
        with requests.get(url, headers=self._header) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            res_json = r.json()
            return res_json

    def _convert_msg_format(self, msg: Dict[str, Any]):
        return msg

    def _prepare_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def _str_to_json(self, msg: str, stream_output: bool):
        content = self._extract_sse_payload(msg)
        if content is None or content == '[DONE]': return ''
        try:
            raw_message = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise _ProviderProtocolError('Provider returned an invalid JSON frame.', http_status=200,
                                         error_body=content) from exc
        if not isinstance(raw_message, dict):
            raise _ProviderProtocolError('Provider returned a non-object JSON frame.', http_status=200,
                                         error_body=content)
        if raw_message.get('error') is not None:
            raise _ProviderProtocolError('Provider returned an error frame.', http_status=200, error_body=content)
        try:
            message = self._convert_msg_format(raw_message)
        except Exception as exc:
            raise _ProviderProtocolError('Provider response conversion failed.', http_status=200,
                                         error_body=content) from exc
        if not isinstance(message, dict):
            if message in ('', None): return ''
            raise _ProviderProtocolError('Provider response conversion returned an invalid frame.', http_status=200,
                                         error_body=content)
        if stream_output: self._emit_message_content(message, stream_output)
        lazyllm.LOG.debug(f'message: {message}')
        return message

    @staticmethod
    def _extract_sse_payload(msg: Union[str, bytes]) -> Optional[str]:
        if isinstance(msg, bytes):
            try:
                msg = msg.decode('utf-8')
            except UnicodeDecodeError as exc:
                body = msg.decode('utf-8', errors='replace')
                raise _ProviderProtocolError('Provider returned a non-UTF-8 frame.', http_status=200,
                                             error_body=body) from exc
        content = msg.strip()
        if not content or content.startswith(':') or re.match(r'^(event|id|retry):', content): return None
        return re.sub(r'^data:\s?', '', content)

    def _emit_message_content(self, message: Dict[str, Any], stream_output: Union[bool, Dict]):
        color = stream_output.get('color') if isinstance(stream_output, dict) else None
        for item in message.get('choices', []):
            if not isinstance(item, dict): continue
            delta = item.get('message', item.get('delta', {}))
            if not isinstance(delta, dict): continue
            if delta.get('reasoning_content') and delta.get('content'):
                lazyllm.LOG.warning('stream delta contains both reasoning_content and content')
            if (reasoning_content := delta.get('reasoning_content', '')):
                self._stream_output(reasoning_content, color, cls='think')
            if (content := delta.get('content', '')) and not delta.get('tool_calls'):
                self._stream_output(content, color)

    @staticmethod
    def _frame_state(message: Dict[str, Any], error_body: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        choices = message.get('choices')
        if choices is None: return None, False
        if not isinstance(choices, list):
            raise _ProviderProtocolError('Provider response has an invalid choices field.', http_status=200,
                                         error_body=error_body)
        if not choices: return None, False
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ProviderProtocolError('Provider response has an invalid first choice.', http_status=200,
                                         error_body=error_body)
        delta = choice.get('message') or choice.get('delta') or {}
        if not isinstance(delta, dict):
            raise _ProviderProtocolError('Provider response has an invalid choice payload.', http_status=200,
                                         error_body=error_body)
        semantic_output = bool(delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'))
        finish_reason = choice.get('finish_reason')
        return finish_reason if finish_reason not in (None, '') else None, semantic_output

    def _emit_structured_event(self, stream_output: Union[bool, Dict], tag: str, **fields):
        if not stream_output: return
        payload = {'tag': tag, **fields}
        stream_sink = stream_output.get('_stream_sink') if isinstance(stream_output, dict) \
            else getattr(self, '_stream_sink', None)
        if stream_sink is not None:
            stream_sink(payload)
        else:
            lazyllm.FileSystemQueue().enqueue(json.dumps(payload, ensure_ascii=False))

    def _emit_provider_status(self, state: Dict[str, Any], stream_output: Union[bool, Dict],
                              error_body: Optional[str] = None):
        if state['provider_status_emitted']: return
        fields = {
            'model_call_id': state['model_call_id'],
            'http_status': state['http_status'],
            'finish_reason': state['finish_reason'],
        }
        if error_body is not None: fields['error_body'] = error_body
        self._emit_structured_event(stream_output, 'provider_status', **fields)
        state['provider_status_emitted'] = True

    @staticmethod
    def _exception_chain(exc: Exception):
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            yield exc
            exc = exc.__cause__ or exc.__context__

    @classmethod
    def _is_retryable_transport_error(cls, exc: Exception) -> bool:
        chain = tuple(cls._exception_chain(exc))
        if any(isinstance(item, (requests.exceptions.SSLError, requests.exceptions.ProxyError)) for item in chain):
            return False
        retryable_types = (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        )
        retryable_names = {'RemoteDisconnected', 'IncompleteRead', 'ProtocolError'}
        return any(isinstance(item, retryable_types) or type(item).__name__ in retryable_names for item in chain)

    @classmethod
    def _is_transport_error(cls, exc: Exception) -> bool:
        if isinstance(exc, _ProviderResponseError): return False
        chain = tuple(cls._exception_chain(exc))
        transport_types = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.SSLError,
            requests.exceptions.ProxyError,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
        )
        transport_names = {'RemoteDisconnected', 'IncompleteRead', 'ProtocolError'}
        return any(isinstance(item, transport_types) or type(item).__name__ in transport_names for item in chain)

    def _extract_specified_key_fields(self, response: Dict[str, Any]):
        if not ('choices' in response and isinstance(response['choices'], list)):
            raise ValueError(f'The response {response} does not contain a `choices` field.')
        outputs = response['choices'][0].get('message') or response['choices'][0].get('delta', {})
        return outputs

    def _merge_stream_result(self, src: List[Union[str, int, list, dict]], force_join: bool = False):
        src = [ele for ele in src if ele is not None]
        if not src: return None
        elif len(src) == 1: return src[0]
        assert len(set(map(type, src))) == 1, f'The elements in the list: {src} are of inconsistent types'

        if isinstance(src[0], str):
            src = [ele for ele in src if ele]
            if not src: return ''
            if force_join or not all(src[0] == ele for ele in src): return ''.join(src)
        elif isinstance(src[0], list):
            src = [ele for ele in src if ele]
            if not src: return []
            # Providers may stream several tool calls one at a time and may put
            # the tail of one call next to the head of another.  In that case
            # each delta contains a different number of indexed tool-call
            # fragments, so positional zip merging is not valid.  Flatten the
            # fragments and let the dict branch merge them by their stable
            # provider index instead.
            if all(
                isinstance(item, dict) and 'index' in item
                for chunk in src for item in chunk
            ):
                merged = self._merge_stream_result([item for chunk in src for item in chunk])
                return merged if isinstance(merged, list) else [merged]
            assert len(set(map(len, src))) == 1, f'The lists of elements: {src} have different lengths.'
            ret = list(map(self._merge_stream_result, zip(*src)))
            return ret[0] if (len(ret) > 0 and isinstance(ret[0], list)) else ret
        elif isinstance(src[0], dict):  # list of dicts
            if 'index' in src[-1]:
                grouped = [list(g) for _, g in groupby(sorted(src, key=itemget('index')), key=itemget('index'))]
                if len(grouped) > 1: return [self._merge_stream_result(src) for src in grouped]
            return {k: self._merge_stream_result([d.get(k) for d in src], k == 'content') for k in set().union(*src)}
        return src[-1]

    def _forward_impl(self, data: Dict[str, Any], *, runtime_url: str,  # noqa C901
                      stream_output: Union[bool, Dict],
                      proxies: Optional[Dict], request_timeout: Optional[Union[float, Tuple[float, float]]],
                      state: Dict[str, Any],
                      ) -> List[Dict[str, Any]]:
        msg_json = []
        try:
            with requests.post(runtime_url, json=data, headers=self._header, stream=stream_output,
                               proxies=proxies, timeout=request_timeout) as r:
                state['http_status'] = r.status_code
                if r.status_code != 200:
                    error_body = r.text
                    raise _ProviderHTTPError(f'Provider returned HTTP status {r.status_code}.',
                                             http_status=r.status_code, error_body=error_body)

                with self.stream_output(stream_output):
                    frames = r.iter_lines() if stream_output else [r.text]
                    for raw_frame in frames:
                        if raw_frame in (b'', ''): continue
                        if self._openai_compatible_response:
                            payload = self._extract_sse_payload(raw_frame)
                            if payload is None: continue
                            if payload == '[DONE]':
                                if state['finish_reason'] is None:
                                    raise _ProviderProtocolError(
                                        'Provider stream ended without a finish_reason.',
                                        http_status=200, error_body=payload,
                                    )
                                break
                            message = self._str_to_json(payload, stream_output)
                        else:
                            message = self._str_to_json(raw_frame, stream_output)
                        if not message: continue
                        msg_json.append(message)
                        finish_reason, semantic_output = self._frame_state(
                            message, payload if self._openai_compatible_response else None,
                        )
                        if semantic_output: state['semantic_output_emitted'] = True
                        if self._openai_compatible_response and finish_reason is not None:
                            state['finish_reason'] = finish_reason
                if self._openai_compatible_response and state['finish_reason'] is None:
                    raise _ProviderProtocolError('Provider response ended without a finish_reason.',
                                                 http_status=200, error_body='')
        except _ProviderResponseError as exc:
            if self._openai_compatible_response:
                state['http_status'] = exc.http_status
                self._emit_provider_status(state, stream_output, exc.error_body)
            raise
        except Exception as exc:
            if self._openai_compatible_response and state['finish_reason'] is not None \
                    and self._is_retryable_transport_error(exc):
                self._emit_provider_status(state, stream_output)
                return msg_json
            raise
        if self._openai_compatible_response: self._emit_provider_status(state, stream_output)
        return msg_json

    def _request_timeout(self, data: Dict[str, Any],
                         default_timeout: Optional[Union[int, float, Tuple[int, int], Tuple[float, float]]] = None,
                         ) -> Optional[Union[float, Tuple[float, float]]]:
        raw_timeout = data.get('timeout')
        if raw_timeout is None:
            raw_timeout = default_timeout
        if raw_timeout is None:
            return None
        try:
            if isinstance(raw_timeout, (tuple, list)):
                if len(raw_timeout) != 2:
                    raise ValueError('timeout tuple/list must contain exactly two values')
                return float(raw_timeout[0]), float(raw_timeout[1])
            return float(raw_timeout)
        except (TypeError, ValueError) as exc:
            lazyllm.LOG.warning(f'Invalid request timeout {raw_timeout}: {exc}; using default None.')
            return None

    def _forward_with_retry(self, data: Dict[str, Any], *, runtime_url: str, stream_output: Union[bool, Dict],
                            proxies: Optional[Dict], max_retries: Optional[int],
                            request_timeout: Optional[Union[float, Tuple[float, float]]]) -> List[Dict[str, Any]]:
        total_attempts = max_retries if max_retries is not None \
            else config['online_chat_transport_max_retries'] + 1
        if total_attempts < 1: raise ValueError('max_retries must be at least 1.')
        retry_budget = total_attempts - 1
        state = {
            'model_call_id': f'call_{uuid.uuid4().hex}',
            'http_status': None,
            'finish_reason': None,
            'semantic_output_emitted': False,
            'provider_status_emitted': False,
        }
        for attempt in range(total_attempts):
            try:
                return self._forward_impl(data, runtime_url=runtime_url, stream_output=stream_output,
                                          proxies=proxies, request_timeout=request_timeout, state=state)
            except Exception as exc:
                retryable = self._is_retryable_transport_error(exc)
                if retryable and not state['semantic_output_emitted'] and attempt < total_attempts - 1:
                    retry_index = attempt + 1
                    base_delay = max(0.0, config['online_chat_transport_retry_base_delay_s'])
                    max_delay = max(base_delay, config['online_chat_transport_retry_max_delay_s'])
                    jitter_ratio = max(0.0, config['online_chat_transport_retry_jitter_ratio'])
                    nominal_delay = min(base_delay * (2 ** (retry_index - 1)), max_delay)
                    delay = random.uniform(max(0.0, nominal_delay * (1 - jitter_ratio)),
                                           nominal_delay * (1 + jitter_ratio))
                    self._emit_structured_event(stream_output, 'model_retry',
                                                model_call_id=state['model_call_id'], retry_index=retry_index,
                                                max_retries=retry_budget, delay_ms=round(delay * 1000))
                    lazyllm.LOG.warning(f'Online chat transport retry {retry_index}/{retry_budget} '
                                        f'for model call {state["model_call_id"]}.')
                    time.sleep(delay)
                    continue
                if self._is_transport_error(exc):
                    self._emit_structured_event(
                        stream_output, 'model_transport_error', model_call_id=state['model_call_id'],
                        http_status=None, finish_reason=None, error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                raise
        raise RuntimeError('Online chat retry loop exited unexpectedly.')

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = None, stream: bool = None,
                lazyllm_files=None, url: str = None, model: str = None, max_retries: Optional[int] = None, **kw):
        request_timeout = self._request_timeout(kw, default_timeout=self._timeout)
        stream_output = stream_output if stream_output is not None else stream
        stream_output = stream_output if stream_output is not None else self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        model, _, url, kw = resolve_online_params(model, None, url, kw,
                                                  model_aliases='model_name', url_aliases='base_url')
        runtime_url = self._get_chat_url(url) if url else self._chat_url
        runtime_model = model or self._model_name

        params = {'input': __input, 'history': llm_chat_history, 'format': self._message_format}
        if tools: params['tools'] = tools
        data = self._prompt.generate_prompt(**params)
        data.update(self._static_params, **dict(model=runtime_model, stream=bool(stream_output)))

        if len(kw) > 0: data.update(kw)
        if len(self._model_optional_params) > 0: data.update(self._model_optional_params)

        if self.type == 'VLM' and (files or self._vlm_force_format_input_with_files):
            data['messages'][-1]['content'] = self._format_input_with_files(data['messages'][-1]['content'], files)
            if llm_chat_history and len(data['messages']) > 1:
                for msg in data['messages'][:-1]:
                    if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                        msg['content'] = self._format_vl_chat_query(msg['content'])

        data = self._prepare_request_data(data)
        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        msg_json = self._forward_with_retry(data, runtime_url=runtime_url, stream_output=stream_output,
                                            proxies=proxies, max_retries=max_retries,
                                            request_timeout=request_timeout)

        usage = {'prompt_tokens': -1, 'completion_tokens': -1}
        if len(msg_json) > 0 and 'usage' in msg_json[-1] and isinstance(msg_json[-1]['usage'], dict):
            for k in usage:
                usage[k] = msg_json[-1]['usage'].get(k, usage[k])
        self._record_usage(usage)
        extractor = self._extract_specified_key_fields(self._merge_stream_result(msg_json))
        return self._formatter(extractor) if extractor else ''

    def _record_usage(self, usage: dict):
        globals['usage'][self._module_id] = usage
        par_muduleid = self._used_by_moduleid
        if par_muduleid is None:
            return
        if par_muduleid not in globals['usage']:
            globals['usage'][par_muduleid] = usage
            return
        existing_usage = globals['usage'][par_muduleid]
        if existing_usage['prompt_tokens'] == -1 or usage['prompt_tokens'] == -1:
            globals['usage'][par_muduleid] = {'prompt_tokens': -1, 'completion_tokens': -1}
        else:
            for k in globals['usage'][par_muduleid]:
                globals['usage'][par_muduleid][k] += usage[k]

    def _upload_train_file(self, train_file) -> str:
        raise NotImplementedError(f'{self.series} not implemented _upload_train_file method in subclass')

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        raise NotImplementedError(f'{self.series} not implemented _create_finetuning_job method in subclass')

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        raise NotImplementedError(f'{self.series} not implemented _query_finetuning_job method in subclass')

    def _query_finetuned_jobs(self) -> dict:
        raise NotImplementedError(f'{self.series} not implemented _query_finetuned_jobs method in subclass')

    def _get_finetuned_model_names(self) -> Tuple[List[str], List[str]]:
        raise NotImplementedError(f'{self.series} not implemented _get_finetuned_model_names method in subclass')

    def set_train_tasks(self, train_file, **kw):
        self._train_file = train_file
        self._train_parameters = kw

    def set_specific_finetuned_model(self, model_id):
        valid_jobs, _ = self._get_finetuned_model_names()
        valid_model_id = [model for _, model in valid_jobs]
        if model_id in valid_model_id:
            self._model_name = model_id
            self._is_trained = True
        else:
            raise ValueError(f'Cannot find modle({model_id}), in fintuned model list: {valid_model_id}')

    def _get_temp_save_dir_path(self):
        save_dir = os.path.join(lazyllm.config['temp_dir'], 'online_model_sft_log')
        if not os.path.exists(save_dir):
            os.system(f'mkdir -p {save_dir}')
        else:
            _delete_old_files(save_dir)
        return save_dir

    def _validate_api_key(self):
        try:
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def _get_train_tasks(self):
        if not self._model_name or not self._train_file:
            raise ValueError('train_model and train_file is required')
        if self._model_name not in self.trainable_models:
            lazyllm.LOG.log_once(f'The current model {self._model_name} is not in the trainable \
                                  model list {self.trainable_models}. The deadline for this list is June 1, 2024. \
                                  This model may not be trainable. If your model is a new model, \
                                  you can ignore this warning.')

        def _create_for_finetuning_job():
            file_id = self._upload_train_file(train_file=self._train_file)
            lazyllm.LOG.info(f'{os.path.basename(self._train_file)} upload success! file id is {file_id}')
            (fine_tuning_job_id, status) = self._create_finetuning_job(self._model_name,
                                                                       file_id,
                                                                       **self._train_parameters)
            lazyllm.LOG.info(f'fine tuning job {fine_tuning_job_id} created, status: {status}')

            if status.lower() == 'failed':
                raise ValueError(f'Fine tuning job {fine_tuning_job_id} failed')
            while status.lower() != 'succeeded':
                try:
                    # wait 10 seconds before querying again
                    time.sleep(random.randint(60, 120))
                    (fine_tuned_model, status) = self._query_finetuning_job(fine_tuning_job_id)
                    lazyllm.LOG.info(f'fine tuning job {fine_tuning_job_id} status: {status}')
                    if status.lower() == 'failed':
                        raise ValueError(f'Finetuning job {fine_tuning_job_id} failed')
                except ValueError:
                    raise ValueError(f'Finetuning job {fine_tuning_job_id} failed')

            lazyllm.LOG.info(f'fine tuned model: {fine_tuned_model} finished')
            self._model_name = fine_tuned_model
            self._is_trained = True

        return pipeline(_create_for_finetuning_job)

    def _create_deployment(self) -> Tuple[str, str]:
        raise NotImplementedError(f'{self.series} not implemented _create_deployment method in subclass')

    def _query_deployment(self, deployment_id) -> str:
        raise NotImplementedError(f'{self.series} not implemented _query_deployment method in subclass')

    def _get_deploy_tasks(self):
        if not self._is_trained: return None

        def _start_for_deployment():
            (deployment_id, status) = self._create_deployment()
            lazyllm.LOG.info(f'deployment {deployment_id} created, status: {status}')

            if status.lower() == 'failed':
                raise ValueError(f'Deployment task {deployment_id} failed')
            status = self._query_deployment(deployment_id)
            while status.lower() != 'running':
                # wait 10 seconds before querying again
                time.sleep(10)
                status = self._query_deployment(deployment_id)
                lazyllm.LOG.info(f'deployment {deployment_id} status: {status}')
                if status.lower() == 'failed':
                    raise ValueError(f'Deployment task {deployment_id} failed')
            lazyllm.LOG.info(f'deployment {deployment_id} finished')
        return pipeline(_start_for_deployment)

    def _format_vl_chat_query(self, query: str):
        return [{'type': 'text', 'text': query}]

    def _format_vl_chat_image_url(self, image_url: str, mime: str) -> List[Dict[str, str]]:
        return [{'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{image_url}'}}]

    # for online vlm
    def _format_input_with_files(self, query: str, query_files: list[str]) -> List[Dict[str, str]]:
        if not query_files:
            return self._format_vl_chat_query(query)
        output = [{'type': 'text', 'text': query}]
        assert isinstance(query_files, list), 'query_files must be a list.'
        for file in query_files:
            mime = None
            if not file.startswith('http'):
                file, mime = _image_to_base64(file)
            output.extend(self._format_vl_chat_image_url(file, mime))
        return output

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineChat', name=self.name, url=self._base_url,
                                 stream=bool(self._stream), return_trace=self._return_trace)

OnlineChatModuleBase = LazyLLMOnlineChatModuleBase
