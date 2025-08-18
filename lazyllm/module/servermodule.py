import re
import time
import requests
import pickle
import codecs
from typing import Callable, Dict, List, Union, Optional, Tuple
import copy
from dataclasses import dataclass

import lazyllm
from lazyllm import launchers, LOG, package, encode_request, globals, is_valid_url, LazyLLMLaunchersBase
from ..components.formatter import FormatterBase, EmptyFormatter, decode_query_with_filepaths
from ..components.formatter.formatterbase import LAZYLLM_QUERY_PREFIX, _lazyllm_get_file_list
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..flow import FlowBase, Pipeline
from ..client import get_redis, redis_client
from urllib.parse import urljoin
from .utils import light_reduce
from .module import ModuleBase, ActionModule


class LLMBase(ModuleBase):
    def __init__(self, stream: Union[bool, Dict[str, str]] = False, return_trace: bool = False,
                 init_prompt: bool = True):
        super().__init__(return_trace=return_trace)
        self._stream = stream
        if init_prompt: self.prompt()
        __class__.formatter(self)

    def _get_files(self, input, lazyllm_files):
        if isinstance(input, package):
            assert not lazyllm_files, 'Duplicate `files` argument provided by args and kwargs'
            input, lazyllm_files = input
        if isinstance(input, str) and input.startswith(LAZYLLM_QUERY_PREFIX):
            assert not lazyllm_files, 'Argument `files` is already provided by query'
            deinput = decode_query_with_filepaths(input)
            assert isinstance(deinput, dict), "decode_query_with_filepaths must return a dict."
            input, files = deinput['query'], deinput['files']
        else:
            files = _lazyllm_get_file_list(lazyllm_files) if lazyllm_files else []
        return input, files

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        if prompt is None:
            assert not history, 'history is not supported in EmptyPrompter'
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            assert not history, 'history is not supported in user defined prompter'
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt, history=history)
        else:
            raise TypeError(f"{prompt} type is not supported.")
        return self

    def formatter(self, format: Optional[FormatterBase] = None):
        assert format is None or isinstance(format, FormatterBase) or callable(format), 'format must be None or Callable'
        self._formatter = format or EmptyFormatter()
        return self

    def share(self, prompt: Optional[Union[str, dict, PrompterBase]] = None, format: Optional[FormatterBase] = None,
              stream: Optional[Union[bool, Dict[str, str]]] = None, history: Optional[List[List[str]]] = None):
        new = copy.copy(self)
        new._hooks = set()
        new._set_mid()
        if prompt is not None: new.prompt(prompt, history=history)
        if format is not None: new.formatter(format)
        if stream is not None: new.stream = stream
        return new

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, v: Union[bool, Dict[str, str]]):
        self._stream = v

    def __or__(self, other):
        if not isinstance(other, FormatterBase):
            return NotImplemented
        return self.share(format=(other if isinstance(self._formatter, EmptyFormatter) else (self._formatter | other)))


class _UrlHelper(object):
    @dataclass
    class _Wrapper:
        url: Optional[str] = None

    def __init__(self, url):
        self._url_wrapper = url if isinstance(url, _UrlHelper._Wrapper) else _UrlHelper._Wrapper(url=url)

    _url_id = property(lambda self: self._module_id)

    @property
    def _url(self) -> str:
        if not self._url_wrapper.url:
            if redis_client:
                try:
                    while not self._url_wrapper.url:
                        self._url_wrapper.url = get_redis(self._url_id)
                        if self._url_wrapper.url: break
                        time.sleep(lazyllm.config["redis_recheck_delay"])
                except Exception as e:
                    LOG.error(f"Error accessing Redis: {e}")
                    raise
        return self._url_wrapper.url

    def _set_url(self, url):
        if redis_client:
            redis_client.set(self._url_id, url)
        LOG.debug(f'url: {url}')
        self._url_wrapper.url = url


class UrlModule(LLMBase, _UrlHelper):

    def __new__(cls, *args, **kw):
        if cls is not UrlModule:
            return super().__new__(cls)
        return ServerModule(*args, **kw)

    def __init__(self, *, url: Optional[str] = '', stream: Union[bool, Dict[str, str]] = False,
                 return_trace: bool = False, init_prompt: bool = True):
        super().__init__(stream=stream, return_trace=return_trace, init_prompt=init_prompt)
        _UrlHelper.__init__(self, url)

    def _estimate_token_usage(self, text):
        if not isinstance(text, str):
            return 0
        # extract english words, number and comma
        pattern = r"\b[a-zA-Z0-9]+\b|,"
        ascii_words = re.findall(pattern, text)
        ascii_ch_count = sum(len(ele) for ele in ascii_words)
        non_ascii_pattern = r"[^\x00-\x7F]"
        non_ascii_chars = re.findall(non_ascii_pattern, text)
        non_ascii_char_count = len(non_ascii_chars)
        return int(ascii_ch_count / 3.0 + non_ascii_char_count + 1)

    def _decode_line(self, line: bytes):
        try:
            return pickle.loads(codecs.decode(line, "base64"))
        except Exception:
            return line.decode('utf-8')

    def _extract_and_format(self, output: str) -> str:
        return output

    def forward(self, *args, **kw): raise NotImplementedError

    def __call__(self, *args, **kw):
        assert self._url is not None, f'Please start {self.__class__} first'
        if len(args) > 1:
            return super(__class__, self).__call__(package(args), **kw)
        return super(__class__, self).__call__(*args, **kw)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Url', name=self._module_name, url=self._url,
                                 stream=self._stream, return_trace=self._return_trace)


@light_reduce
class _ServerModuleImpl(ModuleBase, _UrlHelper):
    def __init__(self, m=None, pre=None, post=None, launcher=None, port=None, pythonpath=None, url_wrapper=None):
        super().__init__()
        _UrlHelper.__init__(self, url=url_wrapper)
        self._m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        self._launcher = launcher.clone() if launcher else launchers.remote(sync=False)
        self._port = port
        self._pythonpath = pythonpath

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._m is None: return None
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self._m, pre_func=self._pre_func, port=self._port,
                                       pythonpath=self._pythonpath, post_func=self._post_func, launcher=self._launcher),
            self._set_url)

    def stop(self):
        self._launcher.cleanup()
        self._get_deploy_tasks.flag.reset()

    def __del__(self):
        self.stop()


class ServerModule(UrlModule):
    def __init__(self, m: Optional[Union[str, ModuleBase]] = None, pre: Optional[Callable] = None,
                 post: Optional[Callable] = None, stream: Union[bool, Dict] = False,
                 return_trace: bool = False, port: Optional[int] = None, pythonpath: Optional[str] = None,
                 launcher: Optional[LazyLLMLaunchersBase] = None, url: Optional[str] = None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        assert (post is None) or (stream is False), 'Stream cannot be true when post-action exists'
        if isinstance(m, str):
            assert url is None, 'url should be None when m is a url'
            url, m = m, None
        if url:
            assert is_valid_url(url), f'Invalid url: {url}'
            assert m is None, 'm should be None when url is provided'
        super().__init__(url=url, stream=stream, return_trace=return_trace)
        self._impl = _ServerModuleImpl(m, pre, post, launcher, port, pythonpath, self._url_wrapper)
        if url: self._impl._get_deploy_tasks.flag.set()

    _url_id = property(lambda self: self._impl._module_id)

    def wait(self):
        self._impl._launcher.wait()

    def stop(self):
        self._impl.stop()

    @property
    def status(self):
        return self._impl._launcher.status

    def _call(self, fname, *args, **kwargs):
        args, kwargs = lazyllm.dump_obj(args), lazyllm.dump_obj(kwargs)
        url = urljoin(self._url.rsplit("/", 1)[0], '_call')
        r = requests.post(url, json=(fname, args, kwargs), headers={'Content-Type': 'application/json'})
        return pickle.loads(codecs.decode(r.content, "base64"))

    def forward(self, __input: Union[Tuple[Union[str, Dict], str], str, Dict] = package(), **kw):  # noqa B008
        headers = {
            'Content-Type': 'application/json',
            'Global-Parameters': encode_request(globals._pickle_data),
            'Session-ID': encode_request(globals._sid)
        }
        data = encode_request((__input, kw))

        # context bug with httpx, so we use requests
        with requests.post(self._url, json=data, stream=True, headers=headers,
                           proxies={'http': None, 'https': None}) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            messages = ''
            with self.stream_output(self._stream):
                for line in r.iter_lines(delimiter=b"<|lazyllm_delimiter|>"):
                    line = self._decode_line(line)
                    if self._stream:
                        self._stream_output(str(line), getattr(self._stream, 'get', lambda x: None)('color'))
                    messages = (messages + str(line)) if self._stream else line

                temp_output = self._extract_and_format(messages)
                return self._formatter(temp_output)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)
