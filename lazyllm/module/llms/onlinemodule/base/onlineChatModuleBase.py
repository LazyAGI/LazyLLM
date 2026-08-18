from itertools import groupby
from email.utils import parsedate_to_datetime
import json
import math
import os
import requests
import re
import random
import socket
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
from .model_call_runner import ModelAttemptState, ModelCallRunner
from .model_outcome import (
    ModelFailure,
    ModelFailureCode,
    ModelFailureOrigin,
    ModelFinish,
    ModelResponseError,
)
from .provider_error_mapping import get_provider_error_mapping
from .utils import LazyLLMOnlineBase, resolve_online_params


class LazyLLMOnlineChatModuleBase(LazyLLMOnlineBase, LLMBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_PREFIX = []
    NO_PROXY = True
    __lazyllm_registry_key__ = LLMType.CHAT
    _message_format = 'openai'
    _PROVIDER_SOURCE = 'openai_compatible'
    _FINISH_REASON_MAP = {
        'stop': ModelFinish.STOP,
        'tool_calls': ModelFinish.TOOL_CALLS,
        'function_call': ModelFinish.TOOL_CALLS,
        'length': ModelFinish.LENGTH,
        'content_filter': ModelFinish.CONTENT_FILTER,
    }

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
            raise self._response_error('Provider returned an invalid JSON frame.', ModelFailureOrigin.PROTOCOL) from exc
        if not isinstance(raw_message, dict):
            raise self._response_error('Provider returned a non-object JSON frame.', ModelFailureOrigin.PROTOCOL)
        self._raise_for_provider_error(raw_message)
        try:
            message = self._convert_msg_format(raw_message)
        except Exception as exc:
            raise self._response_error('Provider response conversion failed.', ModelFailureOrigin.PROTOCOL) from exc
        if not isinstance(message, dict):
            if message in ('', None): return ''
            raise self._response_error('Provider response conversion returned an invalid frame.',
                                       ModelFailureOrigin.PROTOCOL)
        if stream_output: self._emit_message_content(message, stream_output)
        lazyllm.LOG.debug(f'message: {message}')
        return message

    @staticmethod
    def _extract_sse_payload(msg: Union[str, bytes]) -> Optional[str]:
        if isinstance(msg, bytes):
            try:
                msg = msg.decode('utf-8')
            except UnicodeDecodeError as exc:
                failure = ModelFailure(
                    origin=ModelFailureOrigin.PROTOCOL,
                    code=ModelFailureCode.PROTOCOL_ERROR,
                    diagnostic_id=uuid.uuid4().hex,
                )
                raise ModelResponseError('Provider returned a non-UTF-8 frame.', failure) from exc
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
    def _provider_error_fields(message: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        error = message.get('error')
        if not isinstance(error, dict): return None, None
        code = error.get('code')
        error_type = error.get('type')
        return (
            str(code) if code is not None else None,
            str(error_type) if error_type is not None else None,
        )

    def _raise_for_provider_error(self, message: Dict[str, Any]) -> None:
        if message.get('error') is None: return
        code, error_type = self._provider_error_fields(message)
        raise self._response_error(
            'Provider returned an error frame.',
            ModelFailureOrigin.PROVIDER,
            provider_error_code=code,
            provider_error_type=error_type,
        )

    def _classify_failure_code(self, *, origin: ModelFailureOrigin,
                               provider_error_code: Optional[str] = None,
                               provider_error_type: Optional[str] = None,
                               provider_http_status: Optional[int] = None) -> ModelFailureCode:
        if origin == ModelFailureOrigin.PROTOCOL:
            return ModelFailureCode.PROTOCOL_ERROR
        if origin == ModelFailureOrigin.TRANSPORT:
            return ModelFailureCode.TRANSPORT_ERROR
        mapping = get_provider_error_mapping(self._PROVIDER_SOURCE)
        if provider_error_code:
            mapped = mapping.code_map.get(provider_error_code.lower())
            if mapped is not None: return mapped
        if provider_error_type:
            mapped = mapping.type_map.get(provider_error_type.lower())
            if mapped is not None: return mapped
        if provider_http_status is not None:
            mapped = mapping.http_map.get(provider_http_status)
            if mapped is not None: return mapped
        return ModelFailureCode.PROVIDER_REJECTED

    def _response_error(self, message: str, origin: ModelFailureOrigin,
                        provider_error_code: Optional[str] = None,
                        provider_error_type: Optional[str] = None,
                        provider_http_status: Optional[int] = None,
                        retry_after_ms: Optional[int] = None) -> ModelResponseError:
        return ModelResponseError(message, ModelFailure(
            origin=origin,
            code=self._classify_failure_code(
                origin=origin,
                provider_error_code=provider_error_code,
                provider_error_type=provider_error_type,
                provider_http_status=provider_http_status,
            ),
            provider_error_code=provider_error_code,
            provider_error_type=provider_error_type,
            provider_http_status=provider_http_status,
            retry_after_ms=retry_after_ms,
            diagnostic_id=uuid.uuid4().hex,
        ))

    def _log_failure(self, failure: ModelFailure) -> None:
        lazyllm.LOG.warning(
            'provider_failure '
            f'diagnostic_id={failure.diagnostic_id} source={self._PROVIDER_SOURCE} '
            f'origin={failure.origin.value} code={failure.code.value} '
            f'http_status={failure.provider_http_status} provider_code={failure.provider_error_code} '
            f'provider_type={failure.provider_error_type} response_started={failure.response_started} '
            f'semantic_output={failure.has_semantic_output}'
        )

    @classmethod
    def _map_finish_reason(cls, raw_finish_reason: Any) -> ModelFinish:
        return cls._FINISH_REASON_MAP.get(raw_finish_reason, ModelFinish.UNKNOWN)

    @staticmethod
    def _retry_after_ms(headers: Any) -> Optional[int]:
        if headers is None: return None
        raw_value = headers.get('Retry-After')
        if raw_value is None: return None
        value = str(raw_value).strip()
        try:
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0: return None
            return int(seconds * 1000)
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None: return None
            seconds = retry_at.timestamp() - time.time()
            if not math.isfinite(seconds) or seconds < 0: return None
            return int(seconds * 1000)
        except (TypeError, ValueError, OverflowError):
            return None

    def _update_attempt_state(self, message: Dict[str, Any], state: ModelAttemptState):
        choices = message.get('choices')
        if choices is None:
            return
        if not isinstance(choices, list):
            raise self._response_error('Provider response has an invalid choices field.', ModelFailureOrigin.PROTOCOL)
        for choice in choices:
            if not isinstance(choice, dict):
                raise self._response_error('Provider response contains an invalid choice.',
                                           ModelFailureOrigin.PROTOCOL)
            output = choice.get('message') or choice.get('delta') or {}
            if not isinstance(output, dict):
                raise self._response_error('Provider response contains an invalid choice payload.',
                                           ModelFailureOrigin.PROTOCOL)
            if (output.get('content') or output.get('reasoning_content')
                    or output.get('tool_calls') or output.get('function_call')):
                state.semantic_output = True
            raw_finish_reason = choice.get('finish_reason')
            if raw_finish_reason not in (None, ''):
                state.raw_finish_reason = str(raw_finish_reason)
                state.finish = self._map_finish_reason(raw_finish_reason)

    def _emit_runtime_event(self, stream_output: Union[bool, Dict], event_type: str, data: Dict[str, Any]):
        if not stream_output: return
        payload = {
            'tag': 'runtime_event',
            'runtime_event': {
                'schema_version': 1,
                'event_id': uuid.uuid4().hex,
                'type': event_type,
                'data': data,
            },
        }
        stream_sink = stream_output.get('_stream_sink') if isinstance(stream_output, dict) \
            else getattr(self, '_stream_sink', None)
        if stream_sink is not None:
            stream_sink(payload)
        else:
            lazyllm.FileSystemQueue().enqueue(json.dumps(payload, ensure_ascii=False))

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
        dns_errors = tuple(item for item in chain if isinstance(item, socket.gaierror))
        if dns_errors:
            return any(item.errno == socket.EAI_AGAIN for item in dns_errors)
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

    def _extract_partial_content(self, msg_json: list) -> str:
        if not msg_json:
            return ''
        try:
            extractor = self._extract_specified_key_fields(self._merge_stream_result(msg_json))
            return extractor.get('content', '') or ''
        except Exception:
            return ''

    def _raise_for_http_error(self, response: Any) -> None:
        if response.status_code == 200: return
        try:
            error_message = response.json()
        except (TypeError, ValueError):
            error_message = {}
        code, error_type = self._provider_error_fields(error_message) \
            if isinstance(error_message, dict) else (None, None)
        raise self._response_error(
            f'Provider returned HTTP status {response.status_code}.',
            ModelFailureOrigin.HTTP,
            provider_error_code=code,
            provider_error_type=error_type,
            provider_http_status=response.status_code,
            retry_after_ms=self._retry_after_ms(getattr(response, 'headers', None)),
        )

    def _consume_openai_frame(self, raw_frame: Union[str, bytes], msg_json: List[Dict[str, Any]],
                              stream_output: Union[bool, Dict], state: Optional[ModelAttemptState]) -> bool:
        if raw_frame in (b'', ''): return False
        content = self._extract_sse_payload(raw_frame)
        if content is None: return False
        if content == '[DONE]':
            if state is None or state.finish is None:
                raise self._response_error('Provider stream ended without a finish_reason.',
                                           ModelFailureOrigin.PROTOCOL)
            return True
        message = self._str_to_json(content, stream_output)
        if not message: return False
        msg_json.append(message)
        if state is not None: self._update_attempt_state(message, state)
        return False

    def _collect_openai_frames(self, frames: Any, stream_output: Union[bool, Dict],
                               state: Optional[ModelAttemptState]) -> List[Dict[str, Any]]:
        msg_json = []
        try:
            for raw_frame in frames:
                if self._consume_openai_frame(raw_frame, msg_json, stream_output, state):
                    break
        except Exception:
            # A valid OpenAI-compatible finish_reason is the semantic terminal.
            # Transport failure after that terminal must not discard the outcome
            # or trigger another paid attempt.
            if state is None or state.finish is None:
                raise
        if state is None or state.finish is None:
            raise self._response_error('Provider response ended without a finish_reason.',
                                       ModelFailureOrigin.PROTOCOL)
        return msg_json

    def _forward_impl(self, data: Dict[str, Any], *, runtime_url: str, stream_output: Union[bool, Dict],
                      proxies: Optional[Dict], request_timeout: Optional[Union[float, Tuple[float, float]]],
                      state: Optional[ModelAttemptState] = None,
                      ) -> List[Dict[str, Any]]:
        with requests.post(runtime_url, json=data, headers=self._header, stream=stream_output,
                           proxies=proxies, timeout=request_timeout) as r:
            if state is not None: state.response_started = True
            self._raise_for_http_error(r)

            with self.stream_output(stream_output):
                if self._message_format != 'openai':
                    return list(filter(lambda x: x, (
                        [self._str_to_json(line, stream_output) for line in r.iter_lines() if len(line)]
                        if stream_output else [self._str_to_json(r.text, stream_output)]
                    )))
                frames = r.iter_lines() if stream_output else [r.text]
                return self._collect_openai_frames(frames, stream_output, state)

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
                            proxies: Optional[Dict], max_retries: int,
                            request_timeout: Optional[Union[float, Tuple[float, float]]]) -> List[Dict[str, Any]]:
        if self._message_format != 'openai':
            return self._forward_impl(data, runtime_url=runtime_url, stream_output=stream_output,
                                      proxies=proxies, request_timeout=request_timeout)
        runner = ModelCallRunner(
            emit_event=lambda event_type, event_data: self._emit_runtime_event(
                stream_output, event_type, event_data,
            ),
            is_retryable_transport_error=self._is_retryable_transport_error,
            report_failure=self._log_failure,
        )
        return runner.run(
            lambda state: self._forward_impl(
                data,
                runtime_url=runtime_url,
                stream_output=stream_output,
                proxies=proxies,
                request_timeout=request_timeout,
                state=state,
            ),
            max_attempts=max_retries,
        )

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = None, stream: bool = None,
                lazyllm_files=None, url: str = None, model: str = None, max_retries: int = 3, **kw):
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
