import importlib
import pkg_resources
from lazyllm.common import LOG
import os

package_name_map = {
    'huggingface_hub': 'huggingface-hub',
    'jwt': 'PyJWT',
    'rank_bm25': 'rank-bm25',
    'collie': 'collie-lm',
    'faiss': 'faiss-cpu',
    'flash_attn': 'flash-attn',
    'sklearn': 'scikit-learn'
}

requirements = {}

def check_packages(names):
    assert isinstance(names, list)
    missing_pack = []
    for name in names:
        try:
            pkg_resources.get_distribution(name)
        except pkg_resources.DistributionNotFound:
            missing_pack.append(name)
    if len(missing_pack) > 0:
        packs = get_pip_install_cmd(missing_pack)
        if packs:
            LOG.warning(f'Some packages not found, please install it by \'pip install {packs}\'')
        else:
            # should not be here.
            LOG.warning('Some packages not found: ' + " ".join(missing_pack))

def get_pip_install_cmd(names):
    if len(requirements) == 0:
        prep_req_dict()
    install_parts = []
    for name in names:
        if name in package_name_map:
            name = package_name_map[name]
        install_parts.append("\"" + name + requirements[name] + "\"")
    if len(install_parts) > 0:
        return "pip install " + " ".join(install_parts)
    return None


def prep_req_dict():
    req_file_path = os.path.abspath(__file__).replace("lazyllm/thirdparty/__init__.py", "requirements.full.txt")
    try:
        with open(req_file_path, 'r') as req_f:
            lines = req_f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            req_parts = line.split('>=')
            if len(req_parts) == 2:
                requirements[req_parts[0]] = '>=' + req_parts[1]
    except FileNotFoundError:
        LOG.error("requirements.full.txt missing. Cannot generate pip install command.")


class PackageWrapper(object):
    def __init__(self, key, package=None) -> None:
        self._Wrapper__key = key
        self._Wrapper__package = package

    def __getattribute__(self, __name):
        if __name in ('_Wrapper__key', '_Wrapper__package'):
            return super(__class__, self).__getattribute__(__name)
        try:
            return getattr(importlib.import_module(
                self._Wrapper__key, package=self._Wrapper__package), __name)
        except (ImportError, ModuleNotFoundError):
            pip_cmd = get_pip_install_cmd([self._Wrapper__key])
            if pip_cmd:
                err_msg = f'Cannot import module {self._Wrapper__key}, please install it by {pip_cmd}'
            else:
                err_msg = f'Cannot import module {self._Wrapper__key}'
            raise ImportError(err_msg)

modules = ['redis', 'huggingface_hub', 'jieba', 'modelscope', 'pandas', 'jwt', 'rank_bm25', 'redisvl', 'datasets',
           'deepspeed', 'fire', 'numpy', 'peft', 'torch', 'transformers', 'collie', 'faiss', 'flash_attn', 'google',
           'lightllm', 'vllm', 'ChatTTS', 'wandb', 'funasr', 'sklearn', 'torchvision', 'scipy', 'pymilvus',
           'sentence_transformers', 'gradio', 'chromadb', 'nltk', 'PIL', 'httpx', 'bm25s', 'kubernetes', 'pymongo',
           'rapidfuzz', 'FlagEmbedding', 'mcp']
for m in modules:
    vars()[m] = PackageWrapper(m)
