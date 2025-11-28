from typing import Union, Dict, Any, Optional, List
from enum import Enum
import json
import lazyllm
from lazyllm.module import ModuleBase, LLMBase
from lazyllm.components.formatter import JsonFormatter


class JsonExtractor(ModuleBase):
    _prompt = '''
You are an intelligent assistant. Your task is to extract \
information from the user's input text and convert it into the specified JSON format.

## JSON Schema Requirements:
{schema}

## Field Descriptions:
{field_descriptions}

## Extraction Requirements:
1. Extract information from the user's input text that conforms to the above JSON structure
2. Ensure the output JSON format is correct, using double quotes, and all keys and string \
values must be wrapped in double quotes
3. If a field is not mentioned in the input text, set its value to null
4. Output only the JSON string, do not output any other content
5. The output JSON must conform to the provided JSON structure requirements
6. The language output should be same with the JSON structure(key) and input text(value), \
do not translate the input text into other language.
7. If the input text contains multiple JSON objects, extract all of them and return a list of JSON objects.
{extra_requirements}

User input text:
'''

    def __init__(self, base_model: LLMBase, schema: Union[str, Dict[str, Any]],
                 field_descriptions: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]] = None,
                 extra_requirements: Union[str, List[str]] = ''):
        super().__init__()
        self._schema = json.dumps(schema, ensure_ascii=False, indent=2) if isinstance(schema, dict) else str(schema)

        if not field_descriptions:
            self._field_descriptions_str = 'No special instructions, please infer meaning from field names.'
        elif isinstance(field_descriptions, dict):
            desc_lines = [f'- {field}: {desc}' for field, desc in field_descriptions.items()]
            self._field_descriptions_str = '\n'.join(desc_lines)
        elif isinstance(field_descriptions, list):
            self._field_descriptions_str = '\n'.join([f'- {desc}' for desc in field_descriptions])
        else:
            self._field_descriptions_str = str(field_descriptions)

        if isinstance(extra_requirements, str): extra_requirements = [extra_requirements]
        extra_requirements_str = '\n'.join([f'{i}. {inst}' for i, inst in enumerate(extra_requirements, 8)])
        self._prompt = self._prompt.format(schema=self._schema, field_descriptions=self._field_descriptions_str,
                                           extra_requirements=extra_requirements_str)
        self._llm = base_model.share(prompt=self._prompt)
        self._json_formatter = JsonFormatter()

    def forward(self, text: str) -> Dict[str, Any]:
        json_str = ''
        for _ in range(3):
            try:
                json_str = self._llm(text)
                result = self._json_formatter(json_str)
            except Exception: continue
            else:
                return result if isinstance(result, (dict, tuple, list)) else {}
        lazyllm.LOG.warning(f'Failed to parse JSON from model output after 3 attempts. Raw output: {json_str}')
        return {}

    def _batch_forward(self, text_list: List[str]) -> List[Dict[str, Any]]:
        results = lazyllm.FlatList()
        for text in text_list:
            results.absorb(self.forward(text))
        return results


class JsonConcentrator(ModuleBase):
    class Mode(str, Enum):
        REDUCE = 'reduce'
        DISTINCT = 'distinct'

    _summary_prompt = '''
You are an intelligent assistant. Your task is to summarize the values in the list and return a concise summary text. \
If the values in the list are the same or similar, merge the descriptions; if there are different values, summarize \
their main characteristics. The language output should be same with the value list, do not translate the value list \
into other language.

## Extra Requirements:
{requirements}

### Json Schema:
{schema}

### Key of values:
{key}

### Value list:
{value_list}

### Summary:
'''

    _distinct_prompt = '''
You are an intelligent assistant. Your task is to determine whether the current JSON object is semantically \
similar to any of the reference JSON objects in the list.

## Task:
Compare the current JSON object with all reference JSON objects. If the current JSON is semantically similar \
or equivalent to any reference JSON, return "false" (meaning it's a duplicate and should be filtered out). \
Otherwise, return "true" (meaning it's distinct and should be kept).

## Semantic Similarity Criteria:
1. Two JSON objects are considered semantically similar if they convey the same or very similar meaning, even \
if the exact wording or structure differs slightly
2. Minor variations in formatting, order, or wording should not be considered as distinct
3. Only return "false" if there is a clear semantic match with at least one reference JSON

## Output Format:
Output only "true" or "false" (without quotes), nothing else.

## Reference JSON objects (already kept):
{references}

## Current JSON object (to be evaluated):
{curr}

## Decision (true/false):
'''

    # TODO: support specify reduce functor such as sum, max, min, avg, count, etc. for each key in reduce mode.
    # TODO: use schema to make language model understand your keys and values easier.
    def __init__(self, base_model: Optional[LLMBase] = None, schema: Union[str, Dict[str, Any]] = None,
                 mode: str = Mode.REDUCE, *, distinct_roi: Optional[Union[str, List[str]]] = None,
                 raise_on_error: bool = False, extra_requirements: Union[str, List[str]] = 'No extra requirements.'):
        super().__init__()
        self._schema_str = json.dumps(schema, ensure_ascii=False, indent=2) if isinstance(schema, dict) else str(schema)
        try:
            self._schema = json.loads(self._schema_str)
        except Exception:
            self._schema = None
        if self._schema and not isinstance(self._schema, dict):
            raise ValueError(f'Schema is not a valid dictionary: {self._schema_str}')

        pmpt = self._summary_prompt.replace('{schema}', self._schema_str).replace('{requirements}', extra_requirements)
        self._llm = base_model.share(prompt=pmpt) if base_model is not None else None

        self._json_formatter = JsonFormatter()
        self._raise_on_error = raise_on_error
        self._mode = mode
        if distinct_roi:
            if not self._mode == self.Mode.DISTINCT: raise ValueError('distinct_roi only supported in distinct mode.')
            self._distinct_roi = [distinct_roi] if isinstance(distinct_roi, str) else distinct_roi
        else:
            self._distinct_roi = None

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        '''Validate if the data conforms to the schema specification, missing key is allowed.'''
        if self._schema is None:
            return True
        return self._validate_schema_impl(self._schema, data)

    def _validate_schema_impl(self, schema: Dict[str, Any], data: Dict[str, Any], prefix: str = '') -> bool:
        schema_keys = set(schema.keys())
        data_keys = set(data.keys())
        extra_keys = data_keys - schema_keys

        if extra_keys:
            error_msg = f'Schema validation failed: found extra keys {extra_keys} not in schema at {prefix}.' \
                        f'Expected keys: {schema_keys}, given keys: {data_keys}'
            if self._raise_on_error: raise ValueError(error_msg)
            lazyllm.LOG.warning(error_msg)
            return False

        for key, value in data.items():
            if isinstance(value, dict):
                if not self._validate_schema(schema[key], value, f'{prefix}.{key}' if prefix else key):
                    return False
        return True

    def _reduce_aggregate(self, schema: Optional[Dict[str, Any]], jsons: List[Dict[str, Any]],
                          prefix: str = '') -> Dict[str, Any]:
        if not jsons: return {}
        result = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                result[key] = self._reduce_aggregate(value, [json[key] for json in jsons if json.get(key)],
                                                     f'{prefix}.{key}' if prefix else key)
            else:
                result[key] = [json[key] for json in jsons if key in json]
                if self._llm is not None:
                    result[key] = self._llm(dict(key=key, value_list=str(result[key])))
        return result

    def _extract_schema(self, jsons: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None,
                        prefix: str = '') -> Dict[str, Any]:
        schema = schema or {}
        for json_obj in jsons:
            for key, value in json_obj.items():
                if isinstance(value, dict):
                    if (sub_schema := schema.get(key, {})) and not isinstance(sub_schema, dict):
                        raise ValueError(f'Key {key} already in schema with type {sub_schema} at {prefix}.')
                    schema[key] = self._extract_schema(value, sub_schema, prefix=f'{prefix}.{key}' if prefix else key)
                else:
                    if key not in schema: schema[key] = type(value)
                    elif schema[key] != type(value):
                        raise ValueError(f'Key {key} already in schema with type {schema[key]} at {prefix}.')
        return schema

    def _distinct_aggregate(self, jsons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(jsons) <= 1: return jsons
        if self._llm is None:
            raise ValueError('base_model must be provided for distinct mode.')

        def _extract_roi_impl(item: Dict[str, Any], roi: str) -> Any:
            try:
                result = item
                for k in roi.split('.'):
                    result = result.get(k, '')
                return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception:
                return ''

        def extract_roi(item: Dict[str, Any], rois: List[str]) -> str:
            if not rois:
                return json.dumps(item, ensure_ascii=False, indent=2)
            return '\n'.join([f'- {key}: {_extract_roi_impl(item, key)}' for key in rois])

        llm = self._llm.share(prompt=self._distinct_prompt)
        result = [jsons[0].copy()]
        references = [extract_roi(jsons[0], self._distinct_roi)]

        for item in jsons[1:]:
            curr = extract_roi(item, self._distinct_roi)
            references_str = '\n'.join([f'### Reference {i+1}:\n{ref}' for i, ref in enumerate(references)])
            response = llm(dict(references=references_str, curr=curr)).strip().lower()
            if response in ['true', '1', 'yes', 'distinct', 'keep'] or 'true' in response:
                result.append(item.copy())
                references.append(curr)
            # TODO: add combine logic for similar items
        return result

    def forward(self, data_list: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        jsons = lazyllm.FlatList()
        for item in data_list:
            if isinstance(item, str):
                try:
                    jsons.absorb(self._json_formatter(item))
                except Exception as e:
                    if self._raise_on_error: raise e from None
                    lazyllm.LOG.warning(f'Error parsing JSON string: {e}, item: {item}')
            else:
                jsons.absorb(item)
        if self._schema is None:
            schema = self._extract_schema(jsons)
        else:
            schema = self._schema
            jsons = [item for item in jsons if self._validate_schema(item)]
        if not jsons: return {}
        if self._mode == self.Mode.REDUCE:
            return self._reduce_aggregate(schema, jsons)
        elif self._mode == self.Mode.DISTINCT:
            return self._distinct_aggregate(jsons)
        else:
            raise ValueError(f'Invalid mode: {self._mode}')
