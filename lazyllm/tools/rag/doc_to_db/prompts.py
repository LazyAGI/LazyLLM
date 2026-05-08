# flake8: noqa E501

SCHEMA_EXTRACT_PROMPT = '''# Task
You are a helpful assistant that extracts relevant information from a document or text.
You are given a document and a set of schema with description. Your task is to extract the most relevant information from the document and return it in a JSON format.

# Requirements
- You must return the information in the JSON format.
- Make sure all the fields in the schema are present in the JSON.
- For date fields, you should return the date in the format YYYY-MM-DD.
- For every field, you should return the value that is most relevant to the field, and extract the related information as clues.

# Output Format
The output should be a JSON object with the following format:
- field_name(string): The name of the field.
- value(any): The extracted value.
- clues(list of string): The relevant information from input that you used to extract the value, can be multiple.
If some fields are not present in the document, you should give a null value for them and keep the clues empty.

## Output Example
```json
[
    {
        "field_name": "Topic",
        "value": "Scientific Research",
        "clues": ["This paper introduces a new method for Agentic AI.", "According to the paper, the method is used to analyze scientific research papers."]
    }
]
```

# Input
'''

SCHEMA_EXTRACT_INPUT_FORMAT = '''
## Schema
$schema

## Document
$text
'''

SCHEMA_ANALYZE_PROMPT = '''# Role
You are a concise schema designer. Given raw document content, propose a compact set of fields that captures the essential information in the text.

# Requirements
- The fields should be professional and high level, and the description should not contains any specific words (for example the certain company).
- Prefer fewer, high-signal fields over many granular ones.

# Output Format
The output should be a JSON list of object with the following format:
- `name`: snake_case, short, no spaces.
- `description`: brief, clear, grounded in the text.
- `type`: one of `string`, `integer`, `float`, `boolean`, `list`, `dict`.

## Output Example
```json
[
  {"name": "company", "description": "Company name mentioned in the text", "type": "string"},
  {"name": "profit", "description": "Profit of the company in million USD", "type": "float"},
]
```

# Document
'''

SCHEMA_ANALYZE_INPUT_FORMAT = '''
$text
'''
