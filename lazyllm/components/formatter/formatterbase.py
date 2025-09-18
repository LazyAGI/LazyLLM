import json
import uuid
from typing import Callable, Optional, List, Union, Any

from lazyllm.flow.flow import Pipeline

from ...common import LazyLLMRegisterMetaClass, package, Finalizer

def _is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        if s == 'None' or len(s) == 0:
            return False
        else:
            raise ValueError('Invalid number: ' + s + '. You can enter an integer, None or an empyt string.')

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    """This class is the base class of the formatter. The formatter is the formatter of the model output result. Users can customize the formatter or use the formatter provided by LazyLLM.


Examples:
    >>> from lazyllm.components.formatter import LazyLLMFormatterBase
    >>> class MyFormatter(LazyLLMFormatterBase):
    ...     def __init__(self, formatter: str = None):
    ...         self._formatter = formatter
    ...         if self._formatter:
    ...             self._parse_formatter()
    ...         else:
    ...             self._slices = None
    ...     def _parse_formatter(self):
    ...         slice_str = self._formatter.strip()[1:-1]
    ...         slices = []
    ...         parts = slice_str.split(":")
    ...         start = int(parts[0]) if parts[0] else None
    ...         end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    ...         step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    ...         slices.append(slice(start, end, step))
    ...         self._slices = slices
    ...     def _load(self, data):
    ...         return [int(x) for x in data.strip('[]').split(',')]
    ...     def _parse_py_data_by_formatter(self, data):
    ...         if self._slices is not None:
    ...             result = []
    ...             for s in self._slices:
    ...                 if isinstance(s, slice):
    ...                     result.extend(data[s])
    ...                 else:
    ...                     result.append(data[int(s)])
    ...             return result
    ...         else:
    ...             return data
    ...
    >>> fmt = MyFormatter("[1:3]")
    >>> res = fmt.format("[1,2,3,4,5]")
    >>> print(res)
    [2, 3]
    """
    def _load(self, msg: str):
        return msg

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError('This data parse function is not implemented.')

    def format(self, msg):
        """Format input message.

Args:
    msg: Input message, can be string or other format

**Returns:**

- Formatted data, specific type determined by subclass implementation
"""
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, *msg):
        return self.format(msg[0] if len(msg) == 1 else package(msg))

    def __or__(self, other):
        if not isinstance(other, LazyLLMFormatterBase):
            return NotImplemented
        return PipelineFormatter(other.__ror__(self))

    def __ror__(self, f: Callable) -> Pipeline:
        if isinstance(f, Pipeline):
            if not f._capture:
                _ = Finalizer(lambda: setattr(f, '_capture', True), lambda: setattr(f, '_capture', False))
            f._add(str(uuid.uuid4().hex) if len(f._item_names) else None, self)
            return f
        return Pipeline(f, self)


class PipelineFormatter(LazyLLMFormatterBase):
    """Pipeline formatter for encapsulating data processing pipelines as formatters.

This class wraps Pipeline instances as formatters and supports combining multiple formatters through pipe operators.

Args:
    formatter (Pipeline): Pipeline instance to encapsulate
"""
    def __init__(self, formatter: Pipeline):
        self._formatter = formatter

    def _parse_py_data_by_formatter(self, py_data):
        return self._formatter(py_data)

    def __or__(self, other):
        if isinstance(other, LazyLLMFormatterBase):
            if isinstance(other, PipelineFormatter): other = other._formatter
            return PipelineFormatter(self._formatter | other)
        return NotImplemented


class JsonLikeFormatter(LazyLLMFormatterBase):
    """This class is used to extract subfields from nested structures (like dicts, lists, tuples) using a JSON-like indexing syntax.

The behavior is driven by a formatter string similar to Python-style slicing and dictionary access:

- `[0]` fetches the first item
- `[0][{key}]` accesses the `key` field in the first item
- `[0,1][{a,b}]` fetches the `a` and `b` fields from the first and second items
- `[::2]` does slicing with a step of 2
- `*[0][{x}]` means return a wrapped/structured result

Args:
    formatter (str, optional): A format string that controls how to slice and extract the structure. If None, the input will be returned directly.


Examples:
    >>> from lazyllm.components.formatter.formatterbase import JsonLikeFormatter
    >>> formatter = JsonLikeFormatter("[{a,b}]")
    """
    class _ListIdxes(tuple): pass
    class _DictKeys(tuple): pass

    def __init__(self, formatter: Optional[str] = None):
        if formatter and formatter.startswith('*['):
            self._return_package = True
            self._formatter = formatter.strip('*')
        else:
            self._return_package = False
            self._formatter = formatter

        if self._formatter:
            assert '*' not in self._formatter, '`*` can only be used before `[` in the beginning'
            self._formatter = self._formatter.strip().replace('{', '[{').replace('}', '}]')
            self._parse_formatter()
        else:
            self._slices = None

    def _parse_formatter(self):
        # Remove the surrounding brackets
        assert self._formatter.startswith('[') and self._formatter.endswith(']')
        slice_str = self._formatter.strip()[1:-1]
        dimensions = slice_str.split('][')
        slices = []

        for dim in dimensions:
            if '{' in dim:
                slices.append(__class__._DictKeys(d.strip() for d in dim[1:-1].split(',') if d.strip()))
            elif ':' in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(':')
                start = int(parts[0]) if _is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and _is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and _is_number(parts[2]) else None
                slices.append(slice(start, end, step))
            elif ',' in dim:
                slices.append(__class__._ListIdxes(d.strip() for d in dim.split(',') if d.strip()))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _parse_py_data_by_formatter(self, data, *, slices=None):  # noqa C901
        def _impl(data, slice):
            if isinstance(data, (tuple, list)) and isinstance(slice, str):
                return data[int(slice)]
            if isinstance(slice, __class__._ListIdxes):
                if isinstance(data, dict): return [data[k] for k in slice]
                elif isinstance(data, (tuple, list)): return type(data)(data[int(k)] for k in slice)
                else: raise RuntimeError('Only tuple/list/dict is supported for [a,b,c]')
            if isinstance(slice, __class__._DictKeys):
                assert isinstance(data, dict)
                if len(slice) == 1 and slice[0] == ':': return data
                return {k: data[k] for k in slice}
            return data[slice]

        if slices is None: slices = self._slices
        if not slices: return data
        curr_slice = slices[0]
        if isinstance(curr_slice, slice):
            if isinstance(data, dict):
                assert curr_slice.start is None and curr_slice.stop is None and curr_slice.step is None, (
                    'Only {:} and [:] is supported in dict slice')
                curr_slice = __class__._ListIdxes(data.keys())
            elif isinstance(data, (tuple, list)):
                return type(data)(self._parse_py_data_by_formatter(d, slices=slices[1:])
                                  for d in _impl(data, curr_slice))
        if isinstance(curr_slice, __class__._DictKeys):
            return {k: self._parse_py_data_by_formatter(v, slices=slices[1:])
                    for k, v in _impl(data, curr_slice).items()}
        elif isinstance(curr_slice, __class__._ListIdxes):
            tp = package if self._return_package else list if isinstance(data, dict) else type(data)
            return tp(self._parse_py_data_by_formatter(r, slices=slices[1:]) for r in _impl(data, curr_slice))
        else: return self._parse_py_data_by_formatter(_impl(data, curr_slice), slices=slices[1:])


class PythonFormatter(JsonLikeFormatter):
    """Reserved formatter class for supporting Python-style data extraction syntax. To be developed.

Currently inherits from JsonLikeFormatter with no additional behavior.
"""
    pass


class EmptyFormatter(LazyLLMFormatterBase):
    """This type is the system default formatter. When the user does not specify anything or does not want to format the model output, this type is selected. The model output will be in the same format.


Examples:
    >>> import lazyllm
    >>> from lazyllm.components import EmptyFormatter
    >>> toc_prompt='''
    ... You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.
    ... 
    ... Please generate the corresponding list of nested dictionaries based on the following user input:
    ... 
    ... Example output:
    ... [
    ...     {
    ...         "title": "# Level 1 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    ...     },
    ...     {
    ...         "title": "## Level 2 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    ...     },
    ...     {
    ...         "title": "### Level 3 Title",
    ...         "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    ...     }
    ... ]
    ... User input is as follows:
    ... '''
    >>> query = "Please help me write an article about the application of artificial intelligence in the medical field."
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").prompt(toc_prompt).start()  # the model output without specifying a formatter
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret!r}")
    'Based on your user input, here is the corresponding list of nested dictionaries:
    [
        {
            "title": "# Application of Artificial Intelligence in the Medical Field",
            "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."
        },
        {
            "title": "## AI in Medical Diagnosis",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."
        },
        {
            "title": "### AI in Medical Imaging",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."
        },
        {
            "title": "### AI in Drug Discovery and Development",
            "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."
        },
        {
            "title": "## AI in Medical Research",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."
        },
        {
            "title": "### AI in Genomics and Precision Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."
        },
        {
            "title": "### AI in Epidemiology and Public Health",
            "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."
        },
        {
            "title": "### AI in Clinical Trials",
            "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."
        },
        {
            "title": "## AI in Medical Practice",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."
        },
        {
            "title": "### AI in Patient Monitoring",
            "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."
        },
        {
            "title": "### AI in Personalized Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."
        },
        {
            "title": "### AI in Telemedicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."
        },
        {
            "title": "## AI in Medical Ethics and Policy",
            "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."
        }
    ]'
    >>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(EmptyFormatter()).prompt(toc_prompt).start()  # the model output of the specified formatter
    >>> ret = m(query, max_new_tokens=2048)
    >>> print(f"ret: {ret!r}")
    'Based on your user input, here is the corresponding list of nested dictionaries:
    [
        {
            "title": "# Application of Artificial Intelligence in the Medical Field",
            "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."
        },
        {
            "title": "## AI in Medical Diagnosis",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."
        },
        {
            "title": "### AI in Medical Imaging",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."
        },
        {
            "title": "### AI in Drug Discovery and Development",
            "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."
        },
        {
            "title": "## AI in Medical Research",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."
        },
        {
            "title": "### AI in Genomics and Precision Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."
        },
        {
            "title": "### AI in Epidemiology and Public Health",
            "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."
        },
        {
            "title": "### AI in Clinical Trials",
            "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."
        },
        {
            "title": "## AI in Medical Practice",
            "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."
        },
        {
            "title": "### AI in Patient Monitoring",
            "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."
        },
        {
            "title": "### AI in Personalized Medicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."
        },
        {
            "title": "### AI in Telemedicine",
            "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."
        },
        {
            "title": "## AI in Medical Ethics and Policy",
            "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."
        }
    ]'
    """
    def _parse_py_data_by_formatter(self, msg: str):
        return msg

LAZYLLM_QUERY_PREFIX = '<lazyllm-query>'

def encode_query_with_filepaths(query: str = None, files: Union[str, List[str]] = None) -> str:
    """Encodes a query string together with associated file paths into a structured string format with context.

If file paths are provided, the query and file list will be wrapped into a JSON object prefixed with ``__lazyllm_docs__``. Otherwise, it returns the original query string.

Args:
    query (str): The user query string. Defaults to an empty string.
    files (str or List[str]): File path(s) associated with the query. Can be a single string or a list of strings.

**Returns:**

- str: A structured encoded query string or the raw query.

Raises:
    AssertionError: If `files` is not a string or list of strings.


Examples:
    >>> from lazyllm.components.formatter import encode_query_with_filepaths
    
    >>> # Encode a query along with associated documentation files
    >>> encode_query_with_filepaths("Generate questions based on the document", files=["a.md"])
    '<lazyllm-query>{"query": "Generate questions based on the document", "files": ["a.md"]}'
    """
    query = query if query else ''
    if files:
        if isinstance(files, str): files = [files]
        assert isinstance(files, list), 'files must be a list.'
        assert all(isinstance(item, str) for item in files), 'All items in files must be strings'
        return LAZYLLM_QUERY_PREFIX + json.dumps({'query': query, 'files': files})
    else:
        return query

def decode_query_with_filepaths(query_files: str) -> Union[dict, str]:
    """Decodes a structured query string into a dictionary containing the original query and file paths.

If the input string starts with the special prefix ``__lazyllm_docs__``, it attempts to parse the JSON content; otherwise, it returns the raw query string as-is.

Args:
    query_files (str): The encoded query string that may include both query and file paths.

**Returns:**

- Union[dict, str]: A dictionary containing 'query' and 'files' if structured, otherwise the original query string.

Raises:
    AssertionError: If the input is not a string.
    ValueError: If the string is prefixed but JSON decoding fails.


Examples:
    >>> from lazyllm.components.formatter import decode_query_with_filepaths
    
    >>> # Decode a structured query with files
    >>> decode_query_with_filepaths('<lazyllm-query>{"query": "Summarize the content", "files": ["doc.md"]}')
    {'query': 'Summarize the content', 'files': ['doc.md']}
    
    >>> # Decode a plain string without files
    >>> decode_query_with_filepaths("This is just a simple question")
    'This is just a simple question'
    """
    assert isinstance(query_files, str), 'query_files must be a str.'
    query_files = query_files.strip()
    if query_files.startswith(LAZYLLM_QUERY_PREFIX):
        try:
            obj = json.loads(query_files[len(LAZYLLM_QUERY_PREFIX):])
            return obj
        except json.JSONDecodeError as e:
            raise ValueError(f'JSON parsing failed: {e}')
    else:
        return query_files

def lazyllm_merge_query(*args: str) -> str:
    """Merges multiple query strings (potentially with associated file paths) into a single structured query string.

Each argument can be a plain query string or a structured query created by ``encode_query_with_filepaths``. The function decodes each input, concatenates all query texts, and merges the associated file paths. The final result is re-encoded into a single query string with unified context.

Args:
    *args (str): Multiple query strings. Each can be either plain text or an encoded structured query with files.

**Returns:**

- str: A single structured query string containing the merged query and file paths.


Examples:
    >>> from lazyllm.components.formatter import encode_query_with_filepaths, lazyllm_merge_query
    
    >>> # Merge two structured queries with English content and associated files
    >>> q1 = encode_query_with_filepaths("Please summarize document one", files=["doc1.md"])
    >>> q2 = encode_query_with_filepaths("Add details from document two", files=["doc2.md"])
    >>> lazyllm_merge_query(q1, q2)
    '<lazyllm-query>{"query": "Please summarize document oneAdd details from document two", "files": ["doc1.md", "doc2.md"]}'
    
    >>> # Merge plain English text queries without documents
    >>> lazyllm_merge_query("What is AI?", "Explain deep learning.")
    'What is AI?Explain deep learning.'
    """
    if len(args) == 1:
        return args[0]
    for item in args:
        assert isinstance(item, str), 'Merge object must be str!'
    querys = ''
    files = []
    for item in args:
        decode = decode_query_with_filepaths(item)
        if isinstance(decode, dict):
            querys += decode['query']
            files.extend(decode['files'])
        else:
            querys += decode
    return encode_query_with_filepaths(querys, files)

def _lazyllm_get_file_list(files: Any) -> list:
    if isinstance(files, str):
        decode = decode_query_with_filepaths(files)
        if isinstance(decode, str):
            return [decode]
        if isinstance(decode, dict):
            return decode['files']
    elif isinstance(files, dict) and set(files.keys()) == {'query', 'files'}:
        return files['files']
    elif isinstance(files, list) and all(isinstance(item, str) for item in files):
        return files
    else:
        raise TypeError(f'Not supported type: {type(files)}.')

class FileFormatter(LazyLLMFormatterBase):
    """A formatter that transforms query strings with document context between structured formats.

Supports three modes:

- "decode": Decodes structured query strings into dictionaries with `query` and `files`.
- "encode": Encodes a dictionary with `query` and `files` into a structured query string.
- "merge": Merges multiple structured query strings into one.

Args:
    formatter (str): The operation mode. Must be one of "decode", "encode", or "merge". Defaults to "decode".


Examples:
    >>> from lazyllm.components.formatter import FileFormatter
    
    >>> # Decode mode
    >>> fmt = FileFormatter('decode')
    """

    def __init__(self, formatter: str = 'decode'):
        self._mode = formatter.strip().lower()
        assert self._mode in ('decode', 'encode', 'merge')

    def _parse_py_data_by_formatter(self, py_data):
        if self._mode == 'merge':
            if isinstance(py_data, str):
                return py_data
            assert isinstance(py_data, package)
            return lazyllm_merge_query(*py_data)

        if isinstance(py_data, package):
            res = []
            for i_data in py_data:
                res.append(self._parse_py_data_by_formatter(i_data))
            return package(res)
        elif isinstance(py_data, (str, dict)):
            return self._decode_one_data(py_data)
        else:
            return py_data

    def _decode_one_data(self, py_data):
        if self._mode == 'decode':
            if isinstance(py_data, str):
                return decode_query_with_filepaths(py_data)
            else:
                return py_data
        else:
            if isinstance(py_data, dict) and 'query' in py_data and 'files' in py_data:
                return encode_query_with_filepaths(**py_data)
            else:
                return py_data
