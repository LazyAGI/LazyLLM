from typing import Dict, List, Any
from pydantic import ConfigDict, Field, model_validator

from lazyllm.prompt_templates.base import BasePromptTemplate


class PromptTemplate(BasePromptTemplate):
    model_config = ConfigDict(frozen=True)
    template: str = Field(..., description='The prompt template string with {variable} placeholders')
    required_vars: List[str] = Field(default_factory=list, description='List of required variable names')
    partial_vars: Dict[str, Any] = Field(
        default_factory=dict, description='Dictionary of partial variable functions or values'
    )

    @model_validator(mode='after')
    def validate_variables(self):
        # 1. All keys in partial_vars must exist in template variables
        # 2. required_vars + partial_vars keys must exactly equal all template variables
        # Extract all variables from template
        all_vars = set(BasePromptTemplate.get_template_variables(self.template))

        # Check if partial_vars.keys() exist in template variables
        partial_vars_keys = set(self.partial_vars.keys())
        invalid_partial_vars = partial_vars_keys - all_vars
        if invalid_partial_vars:
            raise ValueError(f'partial_vars contains variables not found in template: {invalid_partial_vars}')

        required_vars_set = set(self.required_vars)
        # Check if required_vars and partial_vars have overlap
        overlap_vars = required_vars_set & partial_vars_keys
        if overlap_vars:
            raise ValueError(f'required_vars and partial_vars have overlap: {overlap_vars}')

        # Check if required_vars + partial_vars keys exactly equal all template variables
        combined_vars = required_vars_set | partial_vars_keys
        if combined_vars != all_vars:
            missing_vars = all_vars - combined_vars
            extra_vars = combined_vars - all_vars
            error_msg = []
            if missing_vars:
                error_msg.append(f'Missing variables in required_vars or partial_vars: {missing_vars}')
            if extra_vars:
                error_msg.append(f'Extra variables not found in template: {extra_vars}')
            raise ValueError('; '.join(error_msg))

        return self

    def format(self, **kwargs) -> str:
        # Check if all required variables are provided
        missing_vars = set(self.required_vars) - set(kwargs.keys())
        if missing_vars:
            raise KeyError(f'Missing required variables: {missing_vars}')

        # 1. Apply partial variables. Note: Overrides the values in kwargs if var_name exists in partial_vars
        format_kwargs = {**kwargs}
        for var_name in self.partial_vars:
            try:
                # Apply partial variable function
                val = self.partial_vars[var_name]
                if callable(val):
                    format_kwargs[var_name] = val()
                else:
                    format_kwargs[var_name] = val
            except Exception as e:
                raise TypeError(f'Error applying partial function for variable "{var_name}": {e}')
        # 2. Format the template
        try:
            return self.template.format(**format_kwargs)
        except KeyError as e:
            raise KeyError(f'Template variable not found: {e}')
        except Exception as e:
            raise ValueError(f'Error formatting template: {e}')

    def partial(self, **partial_kwargs) -> 'PromptTemplate':
        # Check if all partial variables exist in the template
        template_vars = set(BasePromptTemplate.get_template_variables(self.template))
        invalid_vars = set(partial_kwargs.keys()) - template_vars
        if invalid_vars:
            raise KeyError(f'Variables not found in template: {invalid_vars}')

        # Create new partial_vars dict with additional partial functions
        new_partial_vars = self.partial_vars.copy()
        new_required_vars = list(set(self.required_vars) - set(partial_kwargs.keys()))
        for var_name, var_value in partial_kwargs.items():
            # Create a function that returns the fixed value
            new_partial_vars[var_name] = var_value

        # Create new template with updated partial_vars
        return PromptTemplate(
            template=self.template,
            required_vars=new_required_vars,
            partial_vars=new_partial_vars
        )

    @classmethod
    def from_template(cls, template: str) -> 'PromptTemplate':
        # Extract all variables from template
        all_vars = BasePromptTemplate.get_template_variables(template)
        return cls(
            template=template,
            required_vars=all_vars,
            partial_vars={}
        )
