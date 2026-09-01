from functools import partial
from itertools import groupby
import json
import os
import requests
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
from .model_call_runner import (
    ModelAttemptState,
    _ModelCallRunner,
    is_retryable_transport_error,
)
from .model_outcome import ModelCallError, ModelFailure
from .provider_response import (
    OPENAI_COMPATIBLE_PROFILE,
    _OpenAICompatibleResponseParser,
    raise_for_http_error,
    select_primary_choice,
    usage_from_frames,
)
from .utils import LazyLLMOnlineBase, resolve_online_params


class LazyLLMOnlineChatModuleBase(LazyLLMOnlineBase, LLMBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_PREFIX = []
    NO_PROXY = True
    __lazyllm_registry_key__ = LLMType.CHAT
    _message_format = 'openai'
    PROVIDER_NAME = 'openai_compatible'
    RESPONSE_PROFILE = OPENAI_COMPATIBLE_PROFILE
    RESPONSE_PARSER_CLASS = _OpenAICompatibleResponseParser

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

    def _response_parser(self, stream_output: Union[bool, Dict]) -> _OpenAICompatibleResponseParser:
        emit_message = partial(self._emit_message_content, stream_output=stream_output) if stream_output else None
        return self.RESPONSE_PARSER_CLASS(
            profile=self.RESPONSE_PROFILE,
            convert_message=self._convert_msg_format,
            emit_message=emit_message,
        )

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

    def _log_failure(self, failure: ModelFailure) -> None:
        lazyllm.LOG.warning(
            'provider_failure '
            f'diagnostic_id={failure.diagnostic_id} source={self.PROVIDER_NAME} '
            f'origin={failure.origin.value} code={failure.code.value} '
            f'http_status={failure.provider_http_status} provider_code={failure.provider_error_code} '
            f'provider_type={failure.provider_error_type} response_started={failure.response_started} '
        )

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

    def _extract_specified_key_fields(self, response: Dict[str, Any]):
        if not ('choices' in response and isinstance(response['choices'], list)):
            raise ValueError(f'The response {response} does not contain a `choices` field.')
        choice = select_primary_choice(response['choices'])
        if choice is None:
            raise ValueError(f'The response {response} contains no choices.')
        outputs = choice.get('message') or choice.get('delta', {})
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

    def _forward_impl(self, data: Dict[str, Any], *, runtime_url: str, stream_output: Union[bool, Dict],
                      proxies: Optional[Dict], request_timeout: Optional[Union[float, Tuple[float, float]]],
                      state: Optional[ModelAttemptState] = None,
                      ) -> List[Dict[str, Any]]:
        # Always defer body consumption so response_started has the same
        # response-header boundary for provider streaming and non-streaming calls.
        with requests.post(runtime_url, json=data, headers=self._header, stream=True,
                           proxies=proxies, timeout=request_timeout) as r:
            if state is not None: state.response_started = True
            raise_for_http_error(r, self.RESPONSE_PROFILE)

            with self.stream_output(stream_output):
                if self._message_format != 'openai':
                    return list(filter(lambda x: x, (
                        [self._parse_response_frame(line, stream_output) for line in r.iter_lines() if len(line)]
                        if stream_output else [self._parse_response_frame(r.text, stream_output)]
                    )))
                frames = r.iter_lines() if stream_output else [r.text]
                return self._response_parser(stream_output).collect(frames, state)

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
            max_attempts = max(1, int(max_retries))
            for attempt_index in range(1, max_attempts + 1):
                state = ModelAttemptState()
                try:
                    return self._forward_impl(
                        data, runtime_url=runtime_url, stream_output=stream_output,
                        proxies=proxies, request_timeout=request_timeout, state=state,
                    )
                except Exception as exc:
                    if (not is_retryable_transport_error(exc) or state.response_started
                            or attempt_index >= max_attempts):
                        raise
                    time.sleep(_ModelCallRunner._retry_delay(attempt_index))
        runner = _ModelCallRunner(
            emit_event=lambda event_type, event_data: self._emit_runtime_event(
                stream_output, event_type, event_data,
            ),
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
                lazyllm_files=None, url: str = None, model: str = None, max_retries: int = 5, **kw):
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
        try:
            msg_json = self._forward_with_retry(data, runtime_url=runtime_url, stream_output=stream_output,
                                                proxies=proxies, max_retries=max_retries,
                                                request_timeout=request_timeout)
        except ModelCallError as exc:
            usage = self._extract_usage(exc.partial_response)
            exc.usage = usage
            self._record_usage(usage)
            raise

        usage = self._extract_usage(msg_json)
        self._record_usage(usage)
        extractor = self._extract_specified_key_fields(self._merge_stream_result(msg_json))
        return self._formatter(extractor) if extractor else ''

    @staticmethod
    def _extract_usage(msg_json: List[Dict[str, Any]]) -> Dict[str, Any]:
        usage: Dict[str, Any] = {'prompt_tokens': -1, 'completion_tokens': -1}
        raw_usage = usage_from_frames(msg_json)
        if raw_usage is not None:
            usage['prompt_tokens'] = raw_usage.get('prompt_tokens', usage['prompt_tokens'])
            usage['completion_tokens'] = raw_usage.get('completion_tokens', usage['completion_tokens'])
            usage['provider_usage'] = dict(raw_usage)
        return usage

    @staticmethod
    def _provider_usage_frames(usage: dict) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        listed = usage.get('provider_usages')
        if isinstance(listed, list):
            frames.extend(item for item in listed if isinstance(item, dict))
        raw = usage.get('provider_usage')
        if isinstance(raw, dict):
            frames.append(raw)
        return frames

    @classmethod
    def _normalize_usage_record(cls, usage: dict) -> Dict[str, Any]:
        record = {
            key: value for key, value in usage.items()
            if key not in ('provider_usage', 'provider_usages')
        }
        frames = cls._provider_usage_frames(usage)
        if frames:
            record['provider_usages'] = frames
        return record

    @classmethod
    def _merge_usage_records(cls, existing: dict, usage: dict) -> Dict[str, Any]:
        if existing.get('prompt_tokens') == -1 or usage.get('prompt_tokens') == -1:
            return {'prompt_tokens': -1, 'completion_tokens': -1}
        merged = dict(existing)
        for key, value in usage.items():
            if key in ('provider_usage', 'provider_usages'):
                continue
            if not isinstance(value, (int, float)):
                continue
            current = merged.get(key)
            if isinstance(current, (int, float)):
                merged[key] = current + value
            elif key not in merged:
                merged[key] = value
        frames = cls._provider_usage_frames(existing)
        frames.extend(cls._provider_usage_frames(usage))
        merged.pop('provider_usage', None)
        if frames:
            merged['provider_usages'] = frames
        else:
            merged.pop('provider_usages', None)
        return merged

    def _record_usage(self, usage: dict):
        current = globals['usage'].get(self._module_id)
        if current is None:
            globals['usage'][self._module_id] = self._normalize_usage_record(usage)
        else:
            globals['usage'][self._module_id] = self._merge_usage_records(current, usage)
        par_muduleid = self._used_by_moduleid
        if par_muduleid is None:
            return
        parent = globals['usage'].get(par_muduleid)
        if parent is None:
            globals['usage'][par_muduleid] = self._normalize_usage_record(usage)
            return
        globals['usage'][par_muduleid] = self._merge_usage_records(parent, usage)

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
