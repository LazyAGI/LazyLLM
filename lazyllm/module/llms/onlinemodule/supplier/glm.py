import json
import os
import uuid
import time
import requests
from typing import Tuple, List, Dict, Union, Optional
from urllib.parse import urljoin, quote
import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    ModelFailureCode, ModelFinish, OnlineChatModuleBase,
    LazyLLMOnlineEmbedModuleBase, LazyLLMOnlineRerankModuleBase,
    LazyLLMOnlineSTTModuleBase, LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineTTSModuleBase, LazyLLMOnlineText2VideoModuleBase
)
from ..fileHandler import FileHandlerBase
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.formatter import encode_query_with_filepaths


_GLM_ERROR_CODE_MAP = {
    '500': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1000': ModelFailureCode.AUTHENTICATION_FAILED,
    '1001': ModelFailureCode.AUTHENTICATION_FAILED,
    '1002': ModelFailureCode.AUTHENTICATION_FAILED,
    '1003': ModelFailureCode.AUTHENTICATION_FAILED,
    '1004': ModelFailureCode.AUTHENTICATION_FAILED,
    '1005': ModelFailureCode.AUTHENTICATION_FAILED,
    '1100': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1110': ModelFailureCode.PERMISSION_DENIED,
    '1111': ModelFailureCode.AUTHENTICATION_FAILED,
    '1112': ModelFailureCode.PERMISSION_DENIED,
    '1113': ModelFailureCode.BALANCE_EXHAUSTED,
    '1120': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1121': ModelFailureCode.PERMISSION_DENIED,
    '1200': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1210': ModelFailureCode.INVALID_REQUEST,
    '1211': ModelFailureCode.NOT_FOUND,
    '1212': ModelFailureCode.INVALID_REQUEST,
    '1213': ModelFailureCode.INVALID_REQUEST,
    '1214': ModelFailureCode.INVALID_REQUEST,
    '1215': ModelFailureCode.INVALID_REQUEST,
    '1220': ModelFailureCode.PERMISSION_DENIED,
    '1221': ModelFailureCode.NOT_FOUND,
    '1222': ModelFailureCode.NOT_FOUND,
    '1230': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1231': ModelFailureCode.CONFLICT,
    '1234': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
    '1261': ModelFailureCode.TOKEN_LIMIT,
    '1300': ModelFailureCode.PROVIDER_REJECTED,
    '1301': ModelFailureCode.INPUT_FILTERED,
    '1302': ModelFailureCode.RATE_LIMITED,
    '1304': ModelFailureCode.QUOTA_EXHAUSTED,
    '1305': ModelFailureCode.PROVIDER_OVERLOADED,
    '1308': ModelFailureCode.QUOTA_EXHAUSTED,
    '1309': ModelFailureCode.QUOTA_EXHAUSTED,
    '1310': ModelFailureCode.QUOTA_EXHAUSTED,
    '1311': ModelFailureCode.PERMISSION_DENIED,
    '1313': ModelFailureCode.RATE_LIMITED,
    '1314': ModelFailureCode.QUOTA_EXHAUSTED,
    '1315': ModelFailureCode.PERMISSION_DENIED,
    '1316': ModelFailureCode.QUOTA_EXHAUSTED,
    '1317': ModelFailureCode.QUOTA_EXHAUSTED,
    '1318': ModelFailureCode.QUOTA_EXHAUSTED,
    '1319': ModelFailureCode.QUOTA_EXHAUSTED,
    '1320': ModelFailureCode.QUOTA_EXHAUSTED,
    '1321': ModelFailureCode.QUOTA_EXHAUSTED,
}


class GLMChat(OnlineChatModuleBase, FileHandlerBase):
    PROVIDER_NAME = 'glm'
    RESPONSE_PROFILE = OnlineChatModuleBase.RESPONSE_PROFILE.extend(
        code_map=_GLM_ERROR_CODE_MAP,
        http_map={435: ModelFailureCode.INVALID_REQUEST},
        finish_map={'sensitive': ModelFinish.CONTENT_FILTER},
    )
    TRAINABLE_MODEL_LIST = ['chatglm3-6b', 'chatglm_12b', 'chatglm_32b', 'chatglm_66b', 'chatglm_130b']
    VLM_MODEL_PREFIX = ['glm-4.5v', 'glm-4.6v', 'glm-4.1v', 'glm-4v',
                        'glm-5v', 'glm-5.3-flash', 'autoglm-phone']
    MODEL_NAME = 'glm-4'

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: str = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://open.bigmodel.cn/api/paas/v4/'
        model = model or lazyllm.config['glm_model_name'] or GLMChat.MODEL_NAME
        if any(model.lower().startswith(prefix) for prefix in self.VLM_MODEL_PREFIX):
            kwargs['type'] = LLMType.VLM
        super().__init__(api_key=api_key or self._default_api_key(),
                         model_name=model or lazyllm.config['glm_model_name'] or GLMChat.MODEL_NAME,
                         base_url=base_url, stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        self.default_train_data = {
            'model': None,
            'training_file': None,
            'validation_file': None,
            'extra_hyperparameters': {
                'fine_tuning_method': None,  # lora\full, default: lora,
                'fine_tuning_parameters': {
                    'max_sequence_length': None  # [1, 8192](int), default: 8192
                }
            },
            'hyperparameters': {
                'learning_rate_multiplier': 0.01,  # (0,5] , default: 1.0
                'batch_size': None,  # [1, 32], default: 8
                'n_epochs': 1,  # [1, 10], default: 3
            },
            'suffix': None,
            'request_id': None
        }
        self.fine_tuning_job_id = None

    def _get_system_prompt(self):
        return ('You are ChatGLM, an AI assistant developed based on a language model trained by Zhipu AI. '
                'Your task is to provide appropriate responses and support for user\'s questions and requests.')

    def _get_chat_url(self, url):
        return url if url.rstrip('/').endswith('chat/completions') else url.rstrip('/') + '/chat/completions'

    def _prepare_request_data(self, data):
        data = dict(data)
        data['model'] = _glm_model_name(data.get('model'))
        return data

    def _format_vl_chat_image_url(self, image_url, mime):
        if image_url.startswith(('http://', 'https://', 'data:')):
            return [{'type': 'image_url', 'image_url': {'url': image_url}}]
        return super()._format_vl_chat_image_url(image_url, mime)

    def _get_models_list(self):
        return list(GLM_MODELS)

    def _convert_file_format(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = {'messages': []}
            messages = ex.get('messages', [])
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                if role in ['system', 'user', 'assistant']:
                    lineEx['messages'].append({'role': role, 'content': content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return '\n'.join(json_strs)

    def _upload_train_file(self, train_file):
        url = urljoin(self._base_url, 'files')
        self.get_finetune_data(train_file)

        file_object = {
            'purpose': (None, 'fine-tune', None),
            'file': (os.path.basename(train_file), self._dataHandler, 'application/json')
        }

        with requests.post(url, headers=self._get_empty_header(), files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            # delete temporary training file
            self._dataHandler.close()
            return r.json()['id']

    def _update_kw(self, data, normal_config):
        cur_data = self.default_train_data.copy()
        cur_data.update(data)

        cur_data['extra_hyperparameters']['fine_tuning_method'] = normal_config['finetuning_type'].strip().lower()
        cur_data['extra_hyperparameters']['fine_tuning_parameters']['max_sequence_length'] = normal_config['cutoff_len']
        cur_data['hyperparameters']['learning_rate_multiplier'] = normal_config['learning_rate']
        cur_data['hyperparameters']['batch_size'] = normal_config['batch_size']
        cur_data['hyperparameters']['n_epochs'] = normal_config['num_epochs']
        cur_data['suffix'] = str(uuid.uuid4())[:7]
        return cur_data

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = urljoin(self._base_url, 'fine_tuning/jobs')
        data = {'model': train_model, 'training_file': train_file_id}
        if len(kw) > 0:
            if 'finetuning_type' in kw:
                data = self._update_kw(data, kw)
            else:
                data.update(kw)

        with requests.post(url, headers=self._header, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()['id']
            self.fine_tuning_job_id = fine_tuning_job_id
            status = self._status_mapping(r.json()['status'])
            return (fine_tuning_job_id, status)

    def _cancel_finetuning_job(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            return 'Invalid'
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{job_id}/cancel')
        with requests.post(fine_tune_url, headers=self._header) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        status = r.json()['status']
        if status == 'cancelled':
            return 'Cancelled'
        else:
            return f'JOB {job_id} status: {status}'

    def _query_finetuned_jobs(self):
        fine_tune_url = os.path.join(self._base_url, 'fine_tuning/jobs/')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _get_finetuned_model_names(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        model_data = self._query_finetuned_jobs()
        res = list()
        for model in model_data['data']:
            res.append([model['id'], model['fine_tuned_model'], self._status_mapping(model['status'])])
        return res

    def _status_mapping(self, status):
        if status == 'succeeded':
            return 'Done'
        elif status == 'failed':
            return 'Failed'
        elif status == 'cancelled':
            return 'Cancelled'
        elif status == 'running':
            return 'Running'
        else:  # create, validating_files, queued
            return 'Pending'

    def _query_job_status(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        _, status = self._query_finetuning_job(job_id)
        return self._status_mapping(status)

    def _get_log(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{job_id}/events')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return job_id, r.json()

    def _get_curr_job_model_id(self):
        if not self.fine_tuning_job_id:
            return None, None
        model_id, _ = self._query_finetuning_job(self.fine_tuning_job_id)
        return self.fine_tuning_job_id, model_id

    def _query_finetuning_job_info(self, fine_tuning_job_id):
        fine_tune_url = os.path.join(self._base_url, f'fine_tuning/jobs/{fine_tuning_job_id}')
        with requests.get(fine_tune_url, headers=self._get_empty_header()) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        status = info['status']
        fine_tuned_model = info['fine_tuned_model'] if 'fine_tuned_model' in info else None
        return (fine_tuned_model, status)

    def _query_finetuning_cost(self, fine_tuning_job_id):
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        if 'trained_tokens' in info and info['trained_tokens']:
            return info['trained_tokens']
        else:
            return None

    def _create_deployment(self) -> Tuple[str]:
        return (self._model_name, 'RUNNING')

    def _query_deployment(self, deployment_id) -> str:
        return 'RUNNING'


class GLMEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self,
                 embed_url: Optional[str] = None,
                 embed_model_name: Optional[str] = None,
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        embed_url = embed_url or 'https://open.bigmodel.cn/api/paas/v4/embeddings'
        embed_model_name = embed_model_name or 'embedding-2'
        super().__init__(embed_url, api_key or self._default_api_key(), embed_model_name,
                         batch_size=batch_size, **kw)

    def _set_embed_url(self):
        self._embed_url = self._get_embed_url(self._embed_url)

    def _encapsulated_data(self, input, **kwargs):
        kwargs['model'] = _glm_model_name(kwargs.get('model') or self._embed_model_name)
        return super()._encapsulated_data(input, **kwargs)


class GLMRerank(LazyLLMOnlineRerankModuleBase):
    def __init__(self,
                 embed_url: Optional[str] = None,
                 embed_model_name: str = 'rerank',
                 api_key: str = None, **kw):
        embed_url = embed_url or 'https://open.bigmodel.cn/api/paas/v4/rerank'
        embed_model_name = embed_model_name or 'rerank'
        super().__init__(embed_url, api_key or self._default_api_key(), embed_model_name, **kw)

    def _set_embed_url(self):
        self._embed_url = self._get_embed_url(self._embed_url)

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            'model': _glm_model_name(kwargs.pop('model', None) or self._embed_model_name),
            'query': query,
            'documents': documents,
            'top_n': top_n,
            'return_documents': False,
            'return_raw_scores': True
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        return [(result['index'], result['relevance_score']) for result in response['results']]


# Public synchronous model APIs; Realtime, layout parsing and voice cloning use
# separate protocols and are not advertised as ordinary chat/TTS models.
GLM_MODELS = (
    'glm-5.3', 'glm-5.3-flash', 'glm-5.3-flashx', 'glm-5.3-flashx-legal',
    'glm-5.2', 'glm-5.1', 'glm-5', 'glm-5-turbo', 'glm-4.7', 'glm-4.7-flashx', 'glm-4.7-flash',
    'glm-4.6', 'glm-4.5-air', 'glm-4.5-airx', 'glm-4.5-flash', 'glm-4-long',
    'glm-4-flashx-250414', 'glm-4-flash-250414', 'glm-5v-turbo', 'glm-4.6v', 'glm-4.6v-flash',
    'autoglm-phone', 'glm-4.1v-thinking-flashx', 'glm-4.1v-thinking-flash', 'glm-4v-flash',
    'glm-image', 'cogview-4-250304', 'cogview-3-flash', 'glm-asr-2512', 'glm-tts',
    'cogvideox-3', 'cogvideox-flash', 'viduq1-text', 'viduq1-image', 'viduq1-start-end',
    'vidu2-image', 'vidu2-start-end', 'vidu2-reference', 'embedding-2', 'embedding-3', 'rerank', 'codegeex-4',
)


def _glm_model_name(model):
    lowered = str(model or '').lower()
    if lowered == 'cogview-4':
        return 'cogview-4-250304'
    return lowered if lowered in GLM_MODELS else model


def _glm_request(method, base_url, endpoint, headers, *, binary=False, timeout=180, **kwargs):
    url = base_url.rstrip('/')
    if not url.endswith('/' + endpoint):
        url += '/' + endpoint
    with requests.request(method, url, headers=headers, timeout=timeout,
                          allow_redirects=False, **kwargs) as response:
        response.raise_for_status()
        return response.content if binary else response.json()


def _glm_download(url, media_type):
    # Generated-file hosts must not receive the provider's authorization header.
    with requests.get(url, timeout=180) as response:
        response.raise_for_status()
        return response.content


class GLMSTT(LazyLLMOnlineSTTModuleBase):
    MODEL_NAME = 'glm-asr-2512'

    def __init__(self, model_name: str = None, api_key: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False, **kwargs):
        super().__init__(model_name=model_name or self.MODEL_NAME, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, base_url=base_url or 'https://open.bigmodel.cn/api/paas/v4',
                         **kwargs)

    def _forward(self, input=None, files=None, url=None, model=None, hotwords=None, **kwargs):
        paths = files or ([input] if isinstance(input, str) and input else [])
        if len(paths) != 1:
            raise ValueError('GLMSTT requires one local audio file')
        data = {'model': _glm_model_name(model or self._model_name), 'stream': 'false'}
        if hotwords is not None:
            data['hotwords'] = json.dumps(hotwords, ensure_ascii=False)
        with open(paths[0], 'rb') as audio:
            result = _glm_request('POST', url or self._base_url, 'audio/transcriptions', self._get_empty_header(),
                                  data=data, files={'file': (os.path.basename(paths[0]), audio)})
        if not isinstance(result.get('text'), str):
            raise RuntimeError('GLM transcription returned no text')
        return result['text']


class GLMText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'cogview-4-250304'

    def __init__(self, model_name: str = None, api_key: str = None, return_trace: bool = False,
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or self.MODEL_NAME, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, base_url=base_url or 'https://open.bigmodel.cn/api/paas/v4',
                         **kwargs)
        if self._type == LLMType.IMAGE_EDITING:
            raise ValueError('The hosted GLM image API adapter supports text-to-image only; '
                             'open-source GLM-Image editing requires a separately supported deployment')

    def _forward(self, input=None, n=1, size='1024x1024', url=None, model=None,
                 image_size=None, batch_size=None, quality=None, watermark_enabled=None, **kwargs):
        count = batch_size if batch_size is not None else n
        if not isinstance(count, int) or count < 1:
            raise ValueError('Image count must be a positive integer')
        selected = _glm_model_name(model or self._model_name)
        payload = {'model': selected, 'prompt': input, 'size': image_size or size}
        if quality is not None:
            payload['quality'] = quality
        if watermark_enabled is not None:
            payload['watermark_enabled'] = watermark_enabled
        images = []
        for _ in range(count):
            result = _glm_request('POST', url or self._base_url, 'images/generations', self._header,
                                  json=payload, timeout=300)
            data = result.get('data')
            if not isinstance(data, list) or not data or not all(isinstance(v, dict) and v.get('url') for v in data):
                raise RuntimeError('GLM image response contains no image URLs')
            images.extend(_glm_download(item['url'], 'image') for item in data)
        return encode_query_with_filepaths(None, bytes_to_file(images))


class GLMTTS(LazyLLMOnlineTTSModuleBase):
    MODEL_NAME = 'glm-tts'

    def __init__(self, model_name=None, api_key=None, base_url=None, return_trace=False, **kwargs):
        super().__init__(model_name=model_name or self.MODEL_NAME, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, base_url=base_url or 'https://open.bigmodel.cn/api/paas/v4',
                         **kwargs)

    def _forward(self, input=None, voice='female', response_format='wav', speed=1.0, volume=1.0,
                 url=None, model=None, **kwargs):
        selected = _glm_model_name(model or self._model_name)
        if selected.lower() != 'glm-tts':
            raise ValueError('Use glm-tts for speech synthesis; voice cloning and Realtime require separate APIs')
        payload = {'model': selected, 'input': input, 'voice': voice, 'response_format': response_format,
                   'speed': speed, 'volume': volume, 'stream': False}
        audio = _glm_request('POST', url or self._base_url, 'audio/speech', self._header, json=payload, binary=True)
        return encode_query_with_filepaths(None, bytes_to_file([audio]))


class GLMText2Video(LazyLLMOnlineText2VideoModuleBase):
    MODEL_NAME = 'cogvideox-3'
    SUPPORTED_IMAGE_ROLES = True

    def __init__(self, model_name=None, api_key=None, base_url=None, return_trace=False, **kwargs):
        super().__init__(model_name=model_name or self.MODEL_NAME, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, base_url=base_url or 'https://open.bigmodel.cn/api/paas/v4',
                         **kwargs)

    def _forward(self, input=None, files=None, image_roles=None, duration=None, size=None, resolution=None,
                 ratio=None, quality=None, with_audio=None, fps=None, style=None, movement_amplitude=None,
                 url=None, model=None, timeout=900, poll_interval=3, **kwargs):
        if timeout <= 0 or poll_interval < 0:
            raise ValueError('timeout must be positive and poll_interval non-negative')
        selected = _glm_model_name(model or self._model_name)
        payload = {'model': selected, 'prompt': input}
        urls, roles = self._video_images(files, image_roles)
        self._validate_video_images(selected, urls, roles)
        if urls:
            payload['image_url'] = urls[0] if len(urls) == 1 else urls
        size = self._video_size(selected, size, resolution, ratio)
        is_vidu = selected.lower().startswith('vidu')
        for key, value in {'duration': duration, 'size': size}.items():
            if value is not None:
                payload[key] = value
        options = {'aspect_ratio': ratio, 'style': style, 'movement_amplitude': movement_amplitude} if is_vidu else {
            'quality': quality, 'with_audio': with_audio, 'fps': fps}
        payload.update({key: value for key, value in options.items() if value is not None})
        base = url or self._base_url
        headers = self._header
        deadline = time.monotonic() + timeout
        task = _glm_request('POST', base, 'videos/generations', headers, json=payload, timeout=min(60, timeout))
        return self._wait_video(task['id'], base, headers, deadline, poll_interval)

    def _video_images(self, files, image_roles):
        urls = []
        for path in files or []:
            if path.startswith(('https://', 'http://', 'data:')):
                urls.append(path)
            else:
                encoded, _ = self._load_images(path)[0]
                urls.append(f'data:image/png;base64,{encoded}')
        roles = list(image_roles or [])
        if roles and len(roles) != len(urls):
            raise ValueError('image_roles must have the same length as files')
        if roles == ['last_frame', 'first_frame']:
            urls.reverse()
            roles.reverse()
        return urls, roles

    @staticmethod
    def _validate_video_images(selected, urls, roles):
        if selected.endswith('-text') and urls:
            raise ValueError('Vidu text models do not accept images; choose an image or start-end model')
        if selected.endswith('-image') and len(urls) != 1:
            raise ValueError('Vidu image models require one image')
        if selected.endswith('-start-end') and len(urls) != 2:
            raise ValueError('Vidu start-end models require two images')
        if selected.endswith('-reference'):
            if not urls or (roles and any(role != 'reference_image' for role in roles)):
                raise ValueError('Vidu reference models require reference images')
        elif roles and roles not in (['first_frame'], ['first_frame', 'last_frame']):
            raise ValueError('This GLM video model accepts first frame or first/last frames')
        if not selected.lower().startswith('vidu') and len(urls) > 2:
            raise ValueError('CogVideoX accepts at most first and last frame')

    @staticmethod
    def _video_size(selected, size, resolution, ratio):
        if size is None and resolution:
            sizes = {'720p': ('1280x720', '720x1280'), '1080p': ('1920x1080', '1080x1920'),
                     '4k': ('3840x2160', '2160x3840'), '480p': ('720x480', '480x720')}
            if resolution.lower() not in sizes:
                raise ValueError('Unsupported GLM video resolution; provide size explicitly')
            # CogVideoX-3 starts at 720p. The tool default is 480p for other providers.
            level = '720p' if selected.lower() == 'cogvideox-3' and resolution.lower() == '480p' else resolution.lower()
            size = '1024x1024' if ratio == '1:1' else sizes[level][ratio == '9:16']
        return size

    def _wait_video(self, task_id, base, headers, deadline, poll_interval):
        while (remaining := deadline - time.monotonic()) > 0:
            result = _glm_request('GET', base, 'async-result/' + quote(task_id, safe=''), headers,
                                  timeout=min(60, remaining))
            status = result.get('task_status')
            if status == 'SUCCESS':
                videos = result.get('video_result') or []
                if not videos or not all(isinstance(item, dict) and item.get('url') for item in videos):
                    raise RuntimeError('GLM video task succeeded without video URLs')
                return encode_query_with_filepaths(None, bytes_to_file([
                    _glm_download(item['url'], 'video') for item in videos]))
            if status != 'PROCESSING':
                raise RuntimeError('GLM video task failed or returned an unknown status')
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        raise TimeoutError('GLM video generation timed out')
