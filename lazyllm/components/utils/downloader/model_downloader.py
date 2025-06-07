import os
import time
import shutil
import functools
import threading
from abc import ABC, abstractmethod

import lazyllm
from .model_mapping import model_name_mapping, model_provider, model_groups
from lazyllm.common.common import EnvVarContextManager

lazyllm.config.add('model_source', str, 'modelscope', 'MODEL_SOURCE')
lazyllm.config.add('model_cache_dir', str, os.path.join(os.path.expanduser('~'), '.lazyllm', 'model'),
                   'MODEL_CACHE_DIR')
lazyllm.config.add('model_path', str, '', 'MODEL_PATH')
lazyllm.config.add('model_source_token', str, '', 'MODEL_SOURCE_TOKEN')
lazyllm.config.add('data_path', str, '', 'DATA_PATH')


class ModelManager():
    def __init__(self, model_source=lazyllm.config['model_source'],
                 token=lazyllm.config['model_source_token'],
                 cache_dir=lazyllm.config['model_cache_dir'],
                 model_path=lazyllm.config['model_path']):
        self.model_source = model_source
        self.token = token or None
        self.cache_dir = cache_dir
        self.model_paths = model_path.split(":") if len(model_path) > 0 else []
        if self.model_source == 'huggingface':
            self.hub_downloader = HuggingfaceDownloader(token=self.token)
        else:
            self.hub_downloader = ModelscopeDownloader(token=self.token)
            if self.model_source != 'modelscope':
                lazyllm.LOG.warning("Only support Huggingface and Modelscope currently. "
                                    f"Unsupported model source: {self.model_source}. Forcing use of Modelscope.")

    @classmethod
    def get_model_type(cls, model) -> str:
        assert isinstance(model, str) and len(model) > 0, "model name should be a non-empty string"
        for name, info in model_name_mapping.items():
            if 'type' not in info: continue

            model_name_set = {name.casefold()}
            for source in info['source']:
                model_name_set.add(info['source'][source].split('/')[-1].casefold())

            if model.split(os.sep)[-1].casefold() in model_name_set:
                return info['type']
        return 'llm'

    @classmethod
    def get_model_name(cls, model) -> str:
        search_string = os.path.basename(model)
        for model_name, sources in model_name_mapping.items():
            if model_name.lower() == search_string.lower() or any(
                os.path.basename(source_file).lower() == search_string.lower()
                for source_file in sources["source"].values()
            ):
                return model_name
        return ""

    @classmethod
    def get_model_prompt_keys(cls, model) -> dict:
        model_name = cls.get_model_name(model)
        if model_name and "prompt_keys" in model_name_mapping[model_name.lower()]:
            return model_name_mapping[model_name.lower()]["prompt_keys"]
        else:
            return dict()

    @classmethod
    def validate_model_path(cls, model_path):
        extensions = {'.pt', '.bin', '.safetensors'}
        for _, _, files in os.walk(model_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    return True
        return False

    def _try_add_mapping(self, model):
        model_base = os.path.basename(model)
        model = model_base.lower()
        if model in model_name_mapping.keys():
            return
        matched_model_prefix = next((key for key in model_provider if model.startswith(key)), None)
        if matched_model_prefix and self.model_source in model_provider[matched_model_prefix]:
            matching_keys = [key for key in model_groups.keys() if key in model]
            if matching_keys:
                matched_groups = max(matching_keys, key=len)
                model_name_mapping[model] = {
                    "prompt_keys": model_groups[matched_groups]["prompt_keys"],
                    "source": {k: v + '/' + model_base for k, v in model_provider[matched_model_prefix].items()}
                }

    def download(self, model='', call_back=None):
        assert isinstance(model, str), "model name should be a string."
        if model.lower() in model_name_mapping.keys() and 'download_by_other' in model_name_mapping[model.lower()]:
            if model_name_mapping[model.lower()]['download_by_other'] is True:
                return model
        self._try_add_mapping(model)
        # Dummy or local model.
        if len(model) == 0 or model[0] in (os.sep, '.', '~') or os.path.isabs(model): return model

        model_at_path = self._model_exists_at_path(model)
        if model_at_path: return model_at_path

        if self.model_source == '' or self.model_source not in ('huggingface', 'modelscope'):
            print("[WARNING] model automatic downloads only support Huggingface and Modelscope currently.")
            return model

        if model.lower() in model_name_mapping.keys() and \
                self.model_source in model_name_mapping[model.lower()]['source'].keys():
            full_model_dir = os.path.join(self.cache_dir, model)

            mapped_model_name = model_name_mapping[model.lower()]['source'][self.model_source]
            model_save_dir = self._do_download(mapped_model_name, call_back)
            if model_save_dir:
                # The code safely creates a symbolic link by removing any existing target.
                if os.path.exists(full_model_dir):
                    os.remove(full_model_dir)
                if os.path.islink(full_model_dir):
                    os.unlink(full_model_dir)
                os.symlink(model_save_dir, full_model_dir, target_is_directory=True)
                return full_model_dir
            return model_save_dir  # return False
        else:
            model_name_for_download = model

            if '/' not in model_name_for_download:
                # Try to figure out a possible model provider
                matched_model_prefix = next((key for key in model_provider if model.lower().startswith(key)), None)
                if matched_model_prefix and self.model_source in model_provider[matched_model_prefix]:
                    model_name_for_download = model_provider[matched_model_prefix][self.model_source] + '/' + model

            model_save_dir = self._do_download(model_name_for_download, call_back)
            return model_save_dir

    def validate_token(self):
        return self.hub_downloader.verify_hub_token()

    def validate_model_id(self, model_id):
        return self.hub_downloader.verify_model_id(model_id)

    def _model_exists_at_path(self, model_name):
        if len(self.model_paths) == 0:
            return None
        model_dirs = []

        # For short model name, get all possible names from the mapping.
        if model_name.lower() in model_name_mapping.keys():
            for source in ('huggingface', 'modelscope'):
                if source in model_name_mapping[model_name.lower()]['source'].keys():
                    model_dirs.append(model_name_mapping[model_name.lower()]['source'][source].replace('/', os.sep))
        model_dirs.append(model_name.replace('/', os.sep))

        for model_path in self.model_paths:
            if len(model_path) == 0: continue
            if model_path[0] != os.sep:
                print(f"[WARNING] skipping path {model_path} as only absolute paths is accepted.")
                continue
            for model_dir in model_dirs:
                full_model_dir = os.path.join(model_path, model_dir)
                if self._is_model_valid(full_model_dir):
                    return full_model_dir
        return None

    def _is_model_valid(self, model_dir):
        if not os.path.isdir(model_dir):
            return False
        return any((True for _ in os.scandir(model_dir)))

    def _do_download(self, model='', call_back=None):
        model_dir = model.replace('/', os.sep)
        full_model_dir = os.path.join(self.cache_dir, self.model_source, model_dir)

        try:
            return self.hub_downloader.download(model, full_model_dir, call_back)
        # Use `BaseException` to capture `KeyboardInterrupt` and normal `Exceptioin`.
        except BaseException as e:
            lazyllm.LOG.warning(f"Download encountered an error: {e}")
            if not self.token:
                lazyllm.LOG.warning('Token is empty, which may prevent private models from being downloaded, '
                                    'as indicated by "the model does not exist." Please set the token with the '
                                    'environment variable LAZYLLM_MODEL_SOURCE_TOKEN to download private models.')
            if os.path.isdir(full_model_dir):
                shutil.rmtree(full_model_dir)
                lazyllm.LOG.warning(f"{full_model_dir} removed due to exceptions.")
        return False

class HubDownloader(ABC):

    def __init__(self, token=None):
        self._token = token if self._verify_hub_token(token) else None
        self._api = self._build_hub_api(self._token)

    @abstractmethod
    def _verify_hub_token(self, token):
        pass

    @abstractmethod
    def _build_hub_api(self, token):
        pass

    @abstractmethod
    def verify_model_id(self, model_id):
        pass

    @abstractmethod
    def _do_download(self, model_id, model_dir):
        pass

    @abstractmethod
    def _get_repo_files(self, model_id):
        pass

    def _polling_progress(self, model_dir, total, polling_event, call_back):
        while not polling_event.is_set():
            n = self._get_current_files_size(model_dir)
            n = min(n, total)
            if callable(call_back):
                try:
                    call_back(n, total)
                except Exception as e:
                    print(f"Error in callback: {e}")
            time.sleep(1)

    def _get_current_files_size(self, model_dir):
        total_size = 0
        for dirpath, _, filenames in os.walk(model_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def _get_files_total_size(self, hub_model_info):
        size = 0
        for item in hub_model_info:
            size += item['Size']
        return size

    def download(self, model_id, model_dir, call_back=None):
        total = self._get_files_total_size(self._get_repo_files(model_id))
        if call_back:
            polling_event = threading.Event()
            polling_thread = threading.Thread(target=self._polling_progress,
                                              args=(model_dir, total, polling_event, call_back))
            polling_thread.daemon = True
            polling_thread.start()
        downloaded_path = self._do_download(model_id, model_dir)
        if call_back and polling_thread:
            polling_event.set()
            polling_thread.join()
        return downloaded_path

    def verify_hub_token(self):
        return True if self._token else False

class HuggingfaceDownloader(HubDownloader):

    def _envs_manager(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            env_vars = {'https_proxy': lazyllm.config['https_proxy'] or os.environ.get("https_proxy", None),
                        'http_proxy': lazyllm.config['http_proxy'] or os.environ.get("http_proxy", None)}
            with EnvVarContextManager(env_vars):
                if not os.environ.get("https_proxy", None):
                    lazyllm.LOG.warning('If there is no download response or if downloads repeatedly fail over an '
                                        'extended period, please set the `LAZYLLM_HTTPS_PROXY` environment variable '
                                        'to configure a proxy. Do not directly set the `https_proxy` and `http_proxy` '
                                        'environment variables in your environment, as doing so may disrupt model '
                                        'deployment and result in deployment failures.')
                return func(self, *args, **kwargs)
        return wrapper

    def _build_hub_api(self, token):
        from huggingface_hub import HfApi
        return HfApi(token=token)

    @_envs_manager
    def _verify_hub_token(self, token):
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.whoami(token)
            return True
        except Exception:
            if token: lazyllm.LOG.warning(f'Huggingface token {token} verified failed')
            return False

    @_envs_manager
    def verify_model_id(self, model_id):
        try:
            self._api.model_info(model_id)
            return True
        except Exception as e:
            lazyllm.LOG.warning('Verify failed: ', e)
            return False

    @_envs_manager
    def _do_download(self, model_id, model_dir):
        from huggingface_hub import snapshot_download
        # refer to https://huggingface.co/docs/huggingface_hub/v0.23.1/en/package_reference/file_download
        if not self.verify_model_id(model_id):
            lazyllm.LOG.warning(f"Invalid model id:{model_id}")
            return False
        downloaded_path = snapshot_download(repo_id=model_id, local_dir=model_dir, token=self._token)
        lazyllm.LOG.info(f"model downloaded at {downloaded_path}")
        return downloaded_path

    @_envs_manager
    def _get_repo_files(self, model_id):
        assert self._api
        orgin_info = self._api.list_repo_tree(model_id, expand=True, recursive=True)
        hub_model_info = []
        for item in list(orgin_info):
            if hasattr(item, 'size'):
                hub_model_info.append({
                    'Path': item.path,
                    'Size': item.size,
                    'SHA': item.blob_id,
                })
        return hub_model_info

class ModelscopeDownloader(HubDownloader):

    def _build_hub_api(self, token):
        from modelscope.hub.api import HubApi
        api = HubApi()
        if token:
            api.login(token)
        return api

    def _verify_hub_token(self, token):
        from modelscope.hub.api import HubApi
        api = HubApi()
        try:
            api.login(token)
            return True
        except Exception:
            if token: lazyllm.LOG.warning(f'Modelscope token {token} verified failed')
            return False

    def verify_model_id(self, model_id):
        try:
            self._api.get_model(model_id)
            return True
        except Exception as e:
            lazyllm.LOG.warning('Verify failed: ', e)
            return False

    def _do_download(self, model_id, model_dir):
        from modelscope.hub.snapshot_download import snapshot_download
        # refer to https://www.modelscope.cn/docs/models/download
        if not self.verify_model_id(model_id):
            lazyllm.LOG.warning(f"Invalid model id:{model_id}")
            return False
        downloaded_path = snapshot_download(model_id=model_id, local_dir=model_dir)
        lazyllm.LOG.info(f"Model downloaded at {downloaded_path}")
        return downloaded_path

    def _get_repo_files(self, model_id):
        assert self._api
        orgin_info = self._api.get_model_files(model_id, recursive=True)
        hub_model_info = []
        for item in orgin_info:
            if item['Type'] == 'blob':
                hub_model_info.append({
                    'Path': item['Path'],
                    'Size': item['Size'],
                    'SHA': item['Sha256']
                })
        return hub_model_info
