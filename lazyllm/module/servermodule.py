import re
import time
import requests
import pickle
import codecs
from typing import Callable, Dict, List, Union, Optional, Tuple
import copy
from dataclasses import dataclass

import lazyllm
from lazyllm import launchers, LOG, package, encode_request, globals, is_valid_url, LazyLLMLaunchersBase, redis_client
from ..components.formatter import FormatterBase, EmptyFormatter, decode_query_with_filepaths
from ..components.formatter.formatterbase import LAZYLLM_QUERY_PREFIX, _lazyllm_get_file_list
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..components.utils import LLMType
from ..flow import FlowBase, Pipeline
from urllib.parse import urljoin
from .utils import light_reduce
from .module import ModuleBase, ActionModule


class LLMBase(object):
    """Base class for large language model modules, inheriting from ModuleBase.  
Manages initialization and switching of streaming output, prompts, and formatters; processes file information in inputs; supports instance sharing.

Args:
    stream (bool or dict): Whether to enable streaming output or streaming configuration, default is False.
    return_trace (bool): Whether to return execution trace, default is False.
    init_prompt (bool): Whether to automatically create a default prompt at initialization, default is True.
"""
    def __init__(self, stream: Union[bool, Dict[str, str]] = False,
                 init_prompt: bool = True, type: Optional[Union[str, LLMType]] = None):
        self._stream = stream
        self._type = LLMType(type) if type else LLMType.LLM
        if init_prompt: self.prompt()
        __class__.formatter(self)

    def _get_files(self, input, lazyllm_files):
        if isinstance(input, package):
            assert not lazyllm_files, 'Duplicate `files` argument provided by args and kwargs'
            input, lazyllm_files = input
        if isinstance(input, str) and input.startswith(LAZYLLM_QUERY_PREFIX):
            assert not lazyllm_files, 'Argument `files` is already provided by query'
            deinput = decode_query_with_filepaths(input)
            assert isinstance(deinput, dict), 'decode_query_with_filepaths must return a dict.'
            input, files = deinput['query'], deinput['files']
        else:
            files = _lazyllm_get_file_list(lazyllm_files) if lazyllm_files else []
        return input, files

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        """Set or switch the prompt. Supports None, PrompterBase subclass, or string/dict to create ChatPrompter.

Args:
    prompt (str/dict/PrompterBase/None): The prompt to set.
    history (list): Conversation history, only valid when prompt is str or dict.

**Returns:**

- self: For chaining calls.
"""
        if prompt is None:
            assert not history, 'history is not supported in EmptyPrompter'
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            assert not history, 'history is not supported in user defined prompter'
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt, history=history)
        else:
            raise TypeError(f'{prompt} type is not supported.')
        return self

    def formatter(self, format: Optional[FormatterBase] = None):
        """Set or switch the output formatter. Supports None, FormatterBase subclass or callable.

Args:
    format (FormatterBase/Callable/None): Formatter object or function, default is None.

**Returns:**

- self: For chaining calls.
"""
        assert format is None or isinstance(format, FormatterBase) or callable(format), 'format must be None or Callable'
        self._formatter = format or EmptyFormatter()
        return self

    def share(self, prompt: Optional[Union[str, dict, PrompterBase]] = None, format: Optional[FormatterBase] = None,
              stream: Optional[Union[bool, Dict[str, str]]] = None, history: Optional[List[List[str]]] = None):
        """Creates a shallow copy of the current instance, with optional resetting of prompt, formatter, and stream attributes.  
Useful for scenarios where multiple sessions or agents share a base configuration but customize certain parameters.

Args:
    prompt (str/dict/PrompterBase/None): New prompt, optional.
    format (FormatterBase/None): New formatter, optional.
    stream (bool/dict/None): New streaming settings, optional.
    history (list/None): New conversation history, effective only when setting prompt.

**Returns:**

- LLMBase: The new shared instance.
"""
        new = copy.copy(self)
        new._hooks = set()
        new._set_mid()
        if prompt is not None: new.prompt(prompt, history=history)
        if format is not None: new.formatter(format)
        if stream is not None: new.stream = stream
        return new

    @property
    def type(self):
        return self._type.value

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
                        url = redis_client['url'].get(self._url_id)
                        self._url_wrapper.url = url.decode('utf-8') if url else None
                        if self._url_wrapper.url: break
                        time.sleep(lazyllm.config['redis_recheck_delay'])
                except Exception as e:
                    LOG.error(f'Error accessing Redis: {e}')
                    raise
        return self._url_wrapper.url

    def _set_url(self, url):
        if redis_client:
            redis_client['url'].set(self._url_id, url)
        LOG.debug(f'url: {url}')
        self._url_wrapper.url = url

    def _release_url(self):
        if redis_client:
            redis_client['url'].delete(self._url_id)

class UrlModule(ModuleBase, LLMBase, _UrlHelper):
    """The URL obtained from deploying the ServerModule can be wrapped into a Module. When calling ``__call__`` , it will access the service.

Args:
    url (str): The URL of the service to be wrapped, defaults to empty string.
    stream (bool|Dict[str, str]): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is False.
    init_prompt (bool): Whether to initialize prompt, defaults to True.


Examples:
    >>> import lazyllm
    >>> def demo(input): return input * 2
    ... 
    >>> s = lazyllm.ServerModule(demo, launcher=lazyllm.launchers.empty(sync=False))
    >>> s.start()
    INFO:     Uvicorn running on http://0.0.0.0:35485
    >>> u = lazyllm.UrlModule(url=s._url)
    >>> print(u(1))
    2
    """

    def __new__(cls, *args, **kw):
        if cls is not UrlModule:
            return super().__new__(cls)
        return ServerModule(*args, **kw)

    def __init__(self, *, url: Optional[str] = '', stream: Union[bool, Dict[str, str]] = False,
                 return_trace: bool = False, init_prompt: bool = True):
        super().__init__(return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, init_prompt=init_prompt)
        _UrlHelper.__init__(self, url)

    def _estimate_token_usage(self, text):
        if not isinstance(text, str):
            return 0
        # extract english words, number and comma
        pattern = r'\b[a-zA-Z0-9]+\b|,'
        ascii_words = re.findall(pattern, text)
        ascii_ch_count = sum(len(ele) for ele in ascii_words)
        non_ascii_pattern = r'[^\x00-\x7F]'
        non_ascii_chars = re.findall(non_ascii_pattern, text)
        non_ascii_char_count = len(non_ascii_chars)
        return int(ascii_ch_count / 3.0 + non_ascii_char_count + 1)

    def _decode_line(self, line: bytes):
        try:
            return pickle.loads(codecs.decode(line, 'base64'))
        except Exception:
            return line.decode('utf-8')

    def _extract_and_format(self, output: str) -> str:
        return output

    def forward(self, *args, **kw):
        """Defines the computation steps to be executed each time. All subclasses of ModuleBase need to override this function.


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...    def forward(self, input):
    ...        return input + 1
    ...
    >>> MyModule()(1)
    2
    """
        raise NotImplementedError

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
    """The ServerModule class inherits from UrlModule and provides functionality to deploy any callable object as an API service.  
Built on FastAPI, it supports launching a main service with multiple satellite services, as well as preprocessing, postprocessing, and streaming capabilities.  
A local callable can be deployed as a service, or an existing service can be accessed directly via a URL.

Args:
    m (Optional[Union[str, ModuleBase]]): The module or its name to be wrapped as a service.  
        If a string is provided, it is treated as a URL and `url` must be None.  
        If a ModuleBase is provided, it will be wrapped as a service.
    pre (Optional[Callable]): Preprocessing function executed in the service process. Default is ``None``.
    post (Optional[Callable]): Postprocessing function executed in the service process. Default is ``None``.
    stream (Union[bool, Dict]): Whether to enable streaming output. Can be a boolean or a dictionary with streaming configuration. Default is ``False``.
    return_trace (Optional[bool]): Whether to return debug trace information. Default is ``False``.
    port (Optional[int]): Port to deploy the service. If ``None``, a random port will be assigned.
    pythonpath (Optional[str]): PYTHONPATH environment variable passed to the subprocess. Defaults to ``None``.
    launcher (Optional[LazyLLMLaunchersBase]): The launcher used to deploy the service. Defaults to asynchronous remote deployment.
    url (Optional[str]): URL of an already deployed service. If provided, `m` must be None.


Examples:
    >>> import lazyllm
    >>> def demo(input): return input * 2
    ...
    >>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
    >>> s.start()
    INFO:     Uvicorn running on http://0.0.0.0:35485
    >>> print(s(1))
    2
    
    >>> class MyServe(object):
    ...     def __call__(self, input):
    ...         return 2 * input
    ...
    ...     @lazyllm.FastapiApp.post
    ...     def server1(self, input):
    ...         return f'reply for {input}'
    ...
    ...     @lazyllm.FastapiApp.get
    ...     def server2(self):
    ...        return f'get method'
    ...
    >>> m = lazyllm.ServerModule(MyServe(), launcher=launchers.empty(sync=False))
    >>> m.start()
    INFO:     Uvicorn running on http://0.0.0.0:32028
    >>> print(m(1))
    2
    """
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
        """Wait for the current module service to finish starting or executing.  
Typically used to block the main thread until the service finishes or is interrupted.  
"""
        self._impl._launcher.wait()

    def stop(self):
        """Stop the current module service and its related subprocesses.  
After this call, the module will no longer respond to requests.  
"""
        self._impl.stop()

    @property
    def status(self):
        return self._impl._launcher.status

    def _call(self, fname, *args, **kwargs):
        args, kwargs = lazyllm.dump_obj(args), lazyllm.dump_obj(kwargs)
        url = urljoin(self._url.rsplit('/', 1)[0], '_call')
        r = requests.post(url, json=(fname, args, kwargs), headers={'Content-Type': 'application/json'})
        if r.status_code != 200:
            try:
                error_info = r.json()
            except ValueError:
                error_info = r.text
            raise requests.RequestException(f'{r.status_code}: {error_info}')
        return pickle.loads(codecs.decode(r.content, 'base64'))

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
                for line in r.iter_lines(delimiter=b'<|lazyllm_delimiter|>'):
                    line = self._decode_line(line)
                    if self._stream:
                        self._stream_output(str(line), getattr(self._stream, 'get', lambda x: None)('color'))
                    messages = (messages + str(line)) if self._stream else line

                temp_output = self._extract_and_format(messages)
                return self._formatter(temp_output)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)
