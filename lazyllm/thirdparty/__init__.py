import importlib
from lazyllm.common import LOG
import os

package_name_map = {
    'huggingface_hub': 'huggingface-hub',
    'jwt': 'PyJWT',
    'rank_bm25': 'rank-bm25',
    'faiss': 'faiss-cpu',
    'flash_attn': 'flash-attn',
    'sklearn': 'scikit-learn',
    'volcenginesdkarkruntime': 'volcengine-python-sdk[ark]',
    'opensearchpy': 'opensearch-py',
}

requirements = {}

def get_pip_install_cmd(names):
    if len(requirements) == 0:
        prep_req_dict()
    install_parts = []
    for name in names:
        if name in package_name_map:
            name = package_name_map[name]
        install_parts.append('\'' + name + requirements.get(name, '') + '\'')
    if len(install_parts) > 0:
        return 'pip install ' + ' '.join(install_parts)
    return None


def prep_req_dict():
    req_file_path = os.path.abspath(__file__).replace('lazyllm/thirdparty/__init__.py', 'requirements.full.txt')
    try:
        with open(req_file_path, 'r') as req_f:
            lines = req_f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            req_parts = line.split('>=')
            if len(req_parts) == 2:
                requirements[req_parts[0]] = '>=' + req_parts[1]
    except FileNotFoundError:
        LOG.error('requirements.full.txt missing. Cannot generate pip install command.')


class PackageWrapper(object):
    def __init__(self, key, *sub_package, package=None, register_patches=None) -> None:
        self._Wrapper__key = key
        self._Wrapper__package = package
        self._Wrapper__sub_packages = sorted(sub_package, reverse=True)
        self._Wrapper__patches = []
        self._Wrapper__lib = None
        if register_patches: self.register_patches(register_patches)

    def register_patches(self, patch_func):
        if isinstance(patch_func, list):
            self._Wrapper__patches.extend(patch_func)
        else:
            self._Wrapper__patches.append(patch_func)

    def __getattribute__(self, __name):
        if __name in ('_Wrapper__key', '_Wrapper__package', '_Wrapper__patches',
                      '_Wrapper__lib', '_Wrapper__sub_packages', 'register_patches'):
            return super(__class__, self).__getattribute__(__name)
        for sub_package in self._Wrapper__sub_packages:
            if __name == sub_package.split('.')[0]:
                return PackageWrapper(f'{self._Wrapper__key}.{__name}', sub_package[len(__name) + 1:],
                                      register_patches=self._Wrapper__patches)
        if self._Wrapper__lib is None:
            try:
                self._Wrapper__lib = importlib.import_module(self._Wrapper__key, package=self._Wrapper__package)
                for patch_func in self._Wrapper__patches: patch_func()
            except ImportError:
                pip_cmd = get_pip_install_cmd([self._Wrapper__key])
                if pip_cmd:
                    err_msg = f'Cannot import module {self._Wrapper__key}, please install it by {pip_cmd}'
                else:
                    err_msg = f'Cannot import module {self._Wrapper__key}'
                raise ImportError(err_msg)
        return getattr(self._Wrapper__lib, __name)

    def __setattr__(self, __name, __value):
        if __name in ('_Wrapper__key', '_Wrapper__package', '_Wrapper__patches',
                      '_Wrapper__lib', '_Wrapper__sub_packages'):
            return super(__class__, self).__setattr__(__name, __value)
        setattr(importlib.import_module(
            self._Wrapper__key, package=self._Wrapper__package), __name, __value)

# os.path is used for test
modules = ['redis', 'huggingface_hub', 'jieba', 'modelscope', 'pandas', 'jwt', 'rank_bm25', 'redisvl', 'datasets',
           'deepspeed', 'fire', 'numpy', 'peft', 'torch', 'transformers', 'faiss', 'flash_attn', 'google',
           'lightllm', 'vllm', 'ChatTTS', 'wandb', 'funasr', 'sklearn', 'torchvision', 'scipy', 'pymilvus',
           'sentence_transformers', 'gradio', 'chromadb', 'nltk', 'PIL', 'httpx', 'bm25s', 'kubernetes', 'pymongo',
           'rapidfuzz', 'FlagEmbedding', 'mcp', 'diffusers', 'pypdf', 'pptx', 'html2text', 'ebooklib', 'docx2txt',
           'zlib', 'struct', 'olefile', 'spacy', 'tarfile', 'boto3', 'botocore', 'paddleocr', 'volcenginesdkarkruntime',
           'zhipuai', 'dashscope', ['mineru', 'cli.common'], 'opensearchpy', ['os', 'path'], 'pkg_resources', 'fastapi',
           ['fsspec', 'implementations.local'], 'bs4', 'requests', 'uvicorn', 'elasticsearch']
for m in modules:
    if isinstance(m, str):
        vars()[m] = PackageWrapper(m)
    else:
        vars()[m[0]] = PackageWrapper(m[0], *m[1:])

def check_packages(names):
    assert isinstance(names, list)
    missing_pack = []
    for name in names:
        try:
            pkg_resources.get_distribution(name)  # noqa: F821
        except pkg_resources.DistributionNotFound:  # noqa: F821
            missing_pack.append(name)
    if len(missing_pack) > 0:
        packs = get_pip_install_cmd(missing_pack)
        if packs:
            LOG.warning(f'Some packages not found, please install it by \'pip install {packs}\'')
        else:
            # should not be here.
            LOG.warning('Some packages not found: ' + ' '.join(missing_pack))
