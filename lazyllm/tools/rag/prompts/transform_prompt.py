# flake8: noqa E501

from pydantic import BaseModel, Field
from typing import List, Dict, Any

from lazyllm.prompt_templates import PromptTemplate, FewShotPromptTemplate, BasePromptTemplate

# Below is the default QA transform prompt
DEFAULT_QA_EGS_TEMPLATE = PromptTemplate.from_template('''
example {index}:
input:
{input}
output:
{output}
''')

DEFAULT_QA_EXAMPLES = [
    {
        'index': 1,
        'input': 'Hello, I am an AI robot developed by SenseTime, named LazyLLM.\nMy mission is to assist you in building the most powerful large-scale model applications with minimal cost.',
        'output': '''Q: What is the name of the AI robot developed by SenseTime?
A: LazyLLM.
Q: Which company developed the AI robot named LazyLLM?
A: SenseTime.
Q: What is the mission of the AI robot named LazyLLM?
A: To assist in building the most powerful large-scale model applications with minimal cost.''',
    },
    {
        'index': 2,
        'input': '你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。',
        'output': '''Q: 商汤科技开发的AI机器人叫什么名字？
A: LazyLLM。
Q: 名为LazyLLM的AI机器人是由哪家公司开发的？
A: 商汤科技。
Q: 名为LazyLLM的AI机器人的使命是什么？
A: 以最小成本协助构建最强大的大规模模型应用程序。''',}]

DEFAULT_QA_PREFIX = '''## Role: Q&A Generation Assistant.
You are a Q&A generation assistant. Your task is to generate {num_pairs_min}-{num_pairs_max} question-answer pairs based on the context.
## Constraints:
- Each question must be answerable directly and specifically from the text.
- The answers should be concise and accurate.
- Do not include information not supported by the context.
- Prefer factual, detail-rich questions over vague ones.
- Questions must be self-contained: avoid ambiguous references such as 'this text', 'this passage', 'here', or 'the above'. Each question should stand alone without requiring external context.
## Text Format:
The input is a string contains the user"s raw input text
## Output Format
Format your output as:
Q: <question>
A: <answer>
Q: <question>
A: <answer>
## Example:'''

DEFAULT_QA_SUFFIX = '''You should not have any unnecessary output. Lets begin:'''

# Use DEFAULT_QA_PROMPT to extract QA pairs from text
DEFAULT_QA_PROMPT = FewShotPromptTemplate(
    prefix=DEFAULT_QA_PREFIX,
    suffix=DEFAULT_QA_SUFFIX,
    examples=DEFAULT_QA_EXAMPLES,
    egs_prompt_template=DEFAULT_QA_EGS_TEMPLATE,
    required_vars=[],
    partial_vars={'num_pairs_min': 3, 'num_pairs_max': 5}
)


# Below is the default Text Summarization transform prompt
DEFAULT_SUMMARIZATION_EGS_TEMPLATE = PromptTemplate.from_template('''
Example {index}:
#input:
{input}
#output:
{output}
''')

DEFAULT_SUMMARIZATION_EXAMPLES = [
    {
        'index': 1,
        'input': 'Hello, I am an AI robot developed by SenseTime, named LazyLLM.\nMy mission is to assist you in building the most powerful large-scale model applications with minimal cost.',
        'output': 'Introduction of AI robot LazyLLM'
    },
    {
        'index': 2,
        'input': '你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。',
        'output': '人工智能机器人LazyLLM的简介'
    }
]

DEFAULT_SUMMARIZATION_PREFIX = '''## Role: Text Summarizer
You are a text summarization engine responsible for analyzing user input text and providing a concise summary based on the requested task.
## Constraints:
- Respond only with the requested output: a brief summary.
- Do not add any extra fields, explanations, or translations.
- Use the same language as the user"s input language.
## Text Format:
The input is a string contains the user"s raw input text
## Examples:'''

DEFAULT_SUMMARIZATION_SUFFIX = '''You should not have any unnecessary output. Lets begin:'''

# Use DEFAULT_SUMMARIZATION_PROMPT to extract text summary from text
DEFAULT_SUMMARIZATION_PROMPT = FewShotPromptTemplate(
    prefix=DEFAULT_SUMMARIZATION_PREFIX,
    suffix=DEFAULT_SUMMARIZATION_SUFFIX,
    examples=DEFAULT_SUMMARIZATION_EXAMPLES,
    egs_prompt_template=DEFAULT_SUMMARIZATION_EGS_TEMPLATE,
    required_vars=[],
    partial_vars={}
)


# Below is the default Keyword Extraction transform prompt
DEFAULT_KEYWORD_EXTRACTION_EGS_TEMPLATE = PromptTemplate.from_template('''
Example {index}:
#input:
{input}
#output:
{output}
''')

DEFAULT_KEYWORD_EXTRACTION_EXAMPLES = [
    {
        'index': 1,
        'input': 'Hello, I am an AI robot developed by SenseTime, named LazyLLM.\nMy mission is to assist you in building the most powerful large-scale model applications with minimal cost.',
        'output': 'LazyLLM, SenseTime, AI robot, large-scale model applications'
    },
    {
        'index': 2,
        'input': '你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。',
        'output': 'LazyLLM, 商汤, 人工智能机器人, 大模型应用'
    }
]

DEFAULT_KEYWORD_EXTRACTION_PREFIX = '''## Role: Keyword Extractor
You are a text keyword extraction engine responsible for analyzing user input text and providing a extracting relevant keywords based on the requested task.
## Constraints:
- Respond only with a list of keywords.
- Do not add any extra fields, explanations, or translations.
- Use the same language as the user"s input language.
## Text Format:
The input is a string contains the user"s raw input text
## Examples:'''

DEFAULT_KEYWORD_EXTRACTION_SUFFIX = '''You should not have any unnecessary output. Lets begin:'''

# Use DEFAULT_KEYWORD_EXTRACTION_PROMPT to extract keywords from text
DEFAULT_KEYWORD_EXTRACTION_PROMPT = FewShotPromptTemplate(
    prefix=DEFAULT_KEYWORD_EXTRACTION_PREFIX,
    suffix=DEFAULT_KEYWORD_EXTRACTION_SUFFIX,
    examples=DEFAULT_KEYWORD_EXTRACTION_EXAMPLES,
    egs_prompt_template=DEFAULT_KEYWORD_EXTRACTION_EGS_TEMPLATE,
    required_vars=[],
    partial_vars={}
)


# Below is the default Q&A Pair Extraction transform prompt
DEFAULT_VLM_QA_EGS_TEMPLATE = PromptTemplate.from_template('''
Example {index}:
Input is an image of {image_description}.
#output:
{output}
''')

DEFAULT_VLM_QA_EXAMPLES = [
    {
        'index': 1,
        'image_description': 'a pig',
        'output': '''Q: What color is the pig?
A: The pig is pink.
Q: What is the pig doing?
A: The pig is running on the lawn.'''
    }
]

DEFAULT_VLM_QA_PREFIX = '''## Role: Q&A Pair Extraction Engine
You are a Q&A pair extraction engine, responsible for analyzing and extracting Q&A pairs from images.
## Constraints:
- Only reply with the requested output content: extracted Q&A pairs.
- Do not add extra fields, explanations, or translations.
- You must answer the question in {language}
## Example:'''

DEFAULT_VLM_QA_SUFFIX = '''You should not output any extra characters. Let"s start now.'''

# Use DEFAULT_VLM_QA_PROMPT to extract Q&A pairs from images
DEFAULT_VLM_QA_PROMPT = FewShotPromptTemplate(
    prefix=DEFAULT_VLM_QA_PREFIX,
    suffix=DEFAULT_VLM_QA_SUFFIX,
    examples=DEFAULT_VLM_QA_EXAMPLES,
    egs_prompt_template=DEFAULT_VLM_QA_EGS_TEMPLATE,
    required_vars=['language'],
    partial_vars={}
)


class LLMTransformParserPrompts(BaseModel):
    # 'summary', 'keywords', 'qa', 'qa_img'
    qa: BasePromptTemplate = DEFAULT_QA_PROMPT
    summary: BasePromptTemplate = DEFAULT_SUMMARIZATION_PROMPT
    keywords: BasePromptTemplate = DEFAULT_KEYWORD_EXTRACTION_PROMPT
    qa_img: BasePromptTemplate = DEFAULT_VLM_QA_PROMPT
