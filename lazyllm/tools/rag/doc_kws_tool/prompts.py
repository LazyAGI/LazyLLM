# flake8: noqa E501
PROMPTS = {}

PROMPTS["doc_type_detection"] = """
You are an advanced document analysis AI. Your task is to:
1. ​**Classify** the document type concisely (within 10 words) based on below doc_content.
2. ​**Return** the result in a structured JSON format with the key `doc_type`.

**Examples of Expected Output:**
- `{{"doc_type": "financial report"}}`
- `{{"doc_type": "scientific research paper"}}`
- `{{"doc_type": "legal contract draft"}}`
- `{{"doc_type": "technical user manual"}}`

**Additional Considerations:**
- If the document is ambiguous, provide the most probable classification.
- Avoid generic labels (e.g., "text document")—be specific.
- If multiple categories fit, choose the most dominant one.

The document content:
```
{doc_content}
```
"""

PROMPTS["kws_generation"] = """
Given {number} document sample(s) of {doc_type}, analyze the structure and content to identify the key metrics and data points in common. Create a comprehensive extraction template in JSON format that captures these essential elements. The JSON template should be a list of key item, each key item should include the following fields:
- **key**: The English name of the field, representing the specific metric for {doc_type}. And this key must exist in all the documents provided.
- **desc**: A clear and concise description of what the field represents, ensuring it is easily understandable.
- **type**: The data type of the field value, which can be one of the following: `int` (for whole numbers), `float` (for decimal numbers), or `text` (for alphanumeric or descriptive data).

Additionally, consider the following:
1. **Granularity**: The template should not summarize or condense the content.
2. **Consistency**: Ensure the field names follow a standardized naming convention (e.g., snake_case) for ease of use.
3. **Flexibility**: Design the template to accommodate variations.
"""


PROMPTS["kws_generation_continue"] = """Some items were missed in the last generated template. Add them below using the same json format."""

PROMPTS["kws_extraction"] = """"
Given a document and keywords description, analyze the content to identify and extract relevant keys that align with the provided descriptions, return the extraction result in JSON format. Ensure the extraction process is thorough and accurate by considering the following:
1. **Contextual Relevance**: Determine how the keywords relate to the document’s content and extract keys that best capture the essence of the described concepts.
2. **Granularity**: Do not summarize or condense the content; it should only provide direct quotations from the original text with the same language. If content for the key is empty, leave it empty.
3. **Precision**: Use the keyword descriptions as a guide to ensure the extracted keys are precise and meaningful, avoiding irrelevant or ambiguous terms.

{extra_desc}
Keywords Description:
```
{kws_desc}
```

Document Content:
```
{doc_content}
```
"""
