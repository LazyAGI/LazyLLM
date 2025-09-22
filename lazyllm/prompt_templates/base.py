from abc import ABC
from string import Formatter
from pydantic import BaseModel


class BasePromptTemplate(BaseModel, ABC):
    """String prompt that exposes the format method, returning a prompt."""

    @classmethod
    def get_template_variables(cls, template: str) -> list[str]:
        try:
            input_variables = {
                v for _, v, _, _ in Formatter().parse(template) if v is not None
            }
            return sorted(input_variables)
        except Exception as e:
            raise ValueError(f"Error getting template variables: {e}")
