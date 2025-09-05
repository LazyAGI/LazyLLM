# coding: utf-8

from dataclasses import dataclass
from typing import ClassVar, List


EN_PROMPT_CLASSIFIER_TEMPLATE = """
## role：Intent Classifier
You are an intent classification engine responsible for analyzing user input text based on dialogue information and determining a unique intent category.{user_prompt}

## Constrains:
You only need to reply with the name of the intent. Do not output any additional fields and do not translate it.{user_constrains}

## Attention:
{attention}

## Text Format
The input text is in JSON format, where "human_input" contains the user's raw input and "intent_list" contains a list of all intent names.

## Example
User: {{"human_input": "What's the weather like in Beijing tomorrow?", "intent_list": ["Check Weather", "Search Engine Query", "View Surveillance", "Report Summary", "Chat"]}}
Assistant: Check Weather
{user_examples}

## Conversation History
The chat history between the human and the assistant is stored within the <histories></histories> XML tags below.
<histories>
{history_info}
</histories>

Input text is as follows:
"""  # noqa E501


@dataclass
class IntentClassifierPrompts:
    """Prompt configuration class for Intent Classifier operations"""

    template: str = EN_PROMPT_CLASSIFIER_TEMPLATE

    KEYWORDS_IN_TEMPLATE: ClassVar[List[str]] = [
        "{user_prompt}", "{user_constrains}", "{attention}", "{user_examples}", "{history_info}"
    ]

    def __post_init__(self):
        if not isinstance(self.template, str):
            raise ValueError("english prompt must be a string")

        def check_keywords(template: str, keywords: List[str]):
            for keyword in keywords:
                if keyword not in template:
                    raise ValueError(f"template must contain {keyword}")

        check_keywords(self.template, self.KEYWORDS_IN_TEMPLATE)
