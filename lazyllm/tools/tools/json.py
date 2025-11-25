from typing import Union, Dict, Any, Optional, List
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

User input text:
'''

    def __init__(self, base_model: LLMBase, schema: Union[str, Dict[str, Any]],
                 field_descriptions: Union[str, Dict[str, str]] = None):
        super().__init__()
        self._schema = json.dumps(schema, ensure_ascii=False, indent=2) if isinstance(schema, dict) else str(schema)

        if not field_descriptions:
            self._field_descriptions_str = 'No special instructions, please infer meaning from field names.'
        elif isinstance(field_descriptions, dict):
            desc_lines = [f'- {field}: {desc}' for field, desc in field_descriptions.items()]
            self._field_descriptions_str = '\n'.join(desc_lines)
        else:
            self._field_descriptions_str = str(field_descriptions)

        self._prompt = self._prompt.format(schema=self._schema, field_descriptions=self._field_descriptions_str)
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

    def batch_forward(self, text_list: List[str]) -> List[Dict[str, Any]]:
        results = lazyllm.FlatList()
        for text in text_list:
            results.absorb(self.forward(text))
        return results


class JsonConcentrator(ModuleBase):
    _summary_prompt_template = '''
You are an intelligent assistant. Your task is to summarize the values in the list and return a concise summary text. \
If the values in the list are the same or similar, merge the descriptions; if there are different values, summarize \
their main characteristics. The language output should be same with the value list, do not translate the value list \
into other language.

### Json Schema:
{schema}

### Key of values:
{key}

### Value list:
{value_list}

### Summary:
'''

    def __init__(self, base_model: Optional[LLMBase] = None, schema: Union[str, Dict[str, Any]] = None,
                 *, raise_on_error: bool = False):
        super().__init__()
        self._schema_str = json.dumps(schema, ensure_ascii=False, indent=2) if isinstance(schema, dict) else str(schema)
        try:
            self._schema = json.loads(self._schema_str)
        except Exception:
            self._schema = None
        if self._schema and not isinstance(self._schema, dict):
            raise ValueError(f'Schema is not a valid dictionary: {self._schema_str}')

        self._llm = (base_model.share(prompt=self._summary_prompt.replace(schema=schema))
                     if base_model is not None else None)

        self._json_formatter = JsonFormatter()
        self._raise_on_error = raise_on_error

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

    def _aggregate_impl(self, schema: Optional[Dict[str, Any]], jsons: List[Dict[str, Any]],
                        prefix: str = '') -> Dict[str, Any]:
        if not jsons: return {}
        result = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                result[key] = self._aggregate_impl(value, [json[key] for json in jsons if json.get(key)],
                                                   f'{prefix}.{key}' if prefix else key)
            else:
                result[key] = [json[key] for json in jsons if key in json]
                if self._llm is not None:
                    result[key] = self._llm(dict(key=key, value_list=result[key]))
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
        return self._aggregate_impl(schema, jsons)
