import json
from typing import Optional, List, Union, Any

from ...common import LazyLLMRegisterMetaClass, package

def is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        if s == "None" or len(s) == 0:
            return False
        else:
            raise ValueError("Invalid number: " + s + ". You can enter an integer, None or an empyt string.")

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    """This class is the base class of the formatter. The formatter is the formatter of the model output result. Users can customize the formatter or use the formatter provided by LazyLLM.
Main methods: _parse_formatter: parse the index content. _load: Parse the str object, and the part containing Python objects is parsed out, such as list, dict and other objects. _parse_py_data_by_formatter: format the python object according to the custom formatter and index. format: format the passed content. If the content is a string type, convert the string into a python object first, and then format it. If the content is a python object, format it directly.


Examples:
    >>> from lazyllm.components.formatter import FormatterBase
    >>> class MyFormatter(FormatterBase):
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
        raise NotImplementedError("This data parse function is not implemented.")

    def format(self, msg):
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, *msg):
        return self.format(msg[0] if len(msg) == 1 else package(msg))


class JsonLikeFormatter(LazyLLMFormatterBase):
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
        dimensions = slice_str.split("][")
        slices = []

        for dim in dimensions:
            if '{' in dim:
                slices.append(__class__._DictKeys(d.strip() for d in dim[1:-1].split(',') if d.strip()))
            elif ":" in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(":")
                start = int(parts[0]) if is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and is_number(parts[2]) else None
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


class PythonFormatter(JsonLikeFormatter): pass


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
    query = query if query else ''
    query_with_docs = {'query': query, 'files': files}
    if files:
        if isinstance(files, str): files = [files]
        assert isinstance(files, list), "files must be a list."
        assert all(isinstance(item, str) for item in files), "All items in files must be strings"
        return LAZYLLM_QUERY_PREFIX + json.dumps(query_with_docs)
    else:
        return query

def decode_query_with_filepaths(query_files: str) -> Union[dict, str]:
    assert isinstance(query_files, str), "query_files must be a str."
    query_files = query_files.strip()
    if query_files.startswith(LAZYLLM_QUERY_PREFIX):
        try:
            obj = json.loads(query_files[len(LAZYLLM_QUERY_PREFIX):])
            return obj
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")
    else:
        return query_files

def lazyllm_merge_query(*args: str) -> str:
    if len(args) == 1:
        return args[0]
    for item in args:
        assert isinstance(item, str), "Merge object must be str!"
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
