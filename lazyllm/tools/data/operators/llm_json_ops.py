from typing import Tuple, Union
from pydantic import BaseModel, ValidationError

from .llm_base_ops import LLMDataJson
from lazyllm import DataPrompt

from ..base_data import data_register
LLMJsonBase = data_register.new_group('llm_json_base')

class FieldExtractor(LLMDataJson, LLMJsonBase):
    _default_prompt = DataPrompt('zh')('field_extractor')
    _default_inference_kwargs = {
        'max_new_tokens': 1024,
        'temperature': 0.2,
    }

    def __init__(self, model, prompt=None, input_keys=None, output_key=None, **kwargs):
        super().__init__(model, prompt, **kwargs)
        self.input_keys = input_keys or ['persona', 'text', 'fields']
        assert len(self.input_keys) == 3, 'input_keys must contain exactly three keys.'
        self.output_key = output_key or 'structured_data'

    def preprocess(self, data: dict, **kwargs) -> Tuple[dict, dict]:
        raw_values = [data.get(k) for k in self.input_keys]
        persona, text, fields = ['' if v is None else str(v) for v in raw_values]
        if not text or not fields:
            raise ValueError(
                f'Missing required input keys. Received persona: "{persona}", '
                f'text: "{text}", fields: "{fields}"')
        return {'persona': persona or 'Extractor', 'text': text, 'fields': fields}, kwargs

    def verify_output(self, output: dict, data: dict) -> bool:
        if not isinstance(output, dict):
            return False
        for key in data.get(self.input_keys[2], []):
            if key not in output:
                return False
        return True

    def postprocess(self, output: dict, data: dict) -> dict:
        processed_output = {k: v.strip() if isinstance(v, str) else v for k, v in output.items()}
        data[self.output_key] = processed_output
        return data


class SchemaExtractor(LLMDataJson, LLMJsonBase):
    _default_prompt = DataPrompt('zh')('schema_extractor')
    _default_inference_kwargs = {
        'max_new_tokens': 1024,
        'temperature': 0.2,
    }
    _default_schema = {'subject': 'subject of the event', 'description': 'detailed description of the event'}

    def __init__(self, model, prompt=None, input_key=None, output_key=None, **kwargs):
        super().__init__(model, prompt, **kwargs)
        self.input_key = input_key or 'text'
        self.output_key = output_key or 'structured_data'

    def _get_schema_dict(self, schema: Union[dict, type]) -> dict:
        if isinstance(schema, dict):
            return schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        else:
            raise ValueError(
                f'Invalid schema format. Expected dict or BaseModel, got {type(schema)}. '
                f'Received schema: "{schema}"'
            )

    def preprocess(self, data: dict, **kwargs) -> Tuple[dict, dict]:
        text = data.get(self.input_key)
        schema = data.get('schema', self._default_schema)
        if not text:
            raise ValueError(f'Missing required input key "{self.input_key}". Received text: "{text}"')
        schema_dict = self._get_schema_dict(schema)
        return {'text': text, 'schema': str(schema_dict)}, kwargs

    def verify_output(self, output: dict, data: dict) -> bool:
        if not isinstance(output, dict):
            return False
        schema = data.get('schema', self._default_schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**output)
                return True
            except ValidationError:
                return False
        for key in schema:
            if key not in output:
                return False
        return True

    def postprocess(self, output: dict, data: dict) -> dict:
        processed_output = {k: v.strip() if isinstance(v, str) else v for k, v in output.items()}
        data[self.output_key] = processed_output
        return data
