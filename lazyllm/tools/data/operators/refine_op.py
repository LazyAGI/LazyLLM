from ..base_data import data_register
import re


Refine = data_register.new_group('refine')

EMOJIS = re.compile(
    '['
    u'\U0001F600-\U0001F64F'
    u'\U0001F300-\U0001F5FF'
    u'\U0001F680-\U0001F6FF'
    u'\U0001F900-\U0001F9FF'
    u'\U0001FA00-\U0001FAFF'
    u'\U0001F1E0-\U0001F1FF'
    u'\U0001F3FB-\U0001F3FF'
    u'\U0000FE00-\U0000FE0F'
    u'\U0001F000-\U0001F02F'
    u'\U0001F0A0-\U0001F0FF'
    u'\U0001F200-\U0001F2FF'
    u'\U00002600-\U000026FF'
    u'\U00002700-\U000027BF'
    '\\u200D'
    '\\uFE0F'
    ']+'
    '|'
    '[\\U0001F1E6-\\U0001F1FF]{2}'
    '|'
    '[\\u200D]',
    flags=re.UNICODE
)
URL_PATTERN = re.compile(r'https?://\S+', flags=re.MULTILINE)
HTML_PATTERN = re.compile(r'<.*?>')
HTML_ENTITY_LIST = [
    'nbsp', 'lt', 'gt', 'amp', 'quot', 'apos', 'hellip', 'ndash', 'mdash',
    'lsquo', 'rsquo', 'ldquo', 'rdquo'
]
ENTITY_PATTERNS = [
    fr'&{entity};'
    for entity in HTML_ENTITY_LIST
] + [
    fr'＆{entity};'
    for entity in HTML_ENTITY_LIST
] + [
    fr'&{entity}；'
    for entity in HTML_ENTITY_LIST
] + [
    fr'＆{entity}；'
    for entity in HTML_ENTITY_LIST
]
HTML_ENTITY_PATTERN = re.compile('|'.join(ENTITY_PATTERNS))


@data_register('data.refine', rewrite_func='forward', _concurrency_mode='process')
def remove_extra_spaces(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        data[input_key] = ' '.join(text.split())
    return data


@data_register('data.refine', rewrite_func='forward', _concurrency_mode='process')
def remove_emoji(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        data[input_key] = EMOJIS.sub('', text)
    return data


@data_register('data.refine', rewrite_func='forward', _concurrency_mode='process')
def remove_html_url(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        text = URL_PATTERN.sub('', text)
        text = HTML_PATTERN.sub('', text)
        data[input_key] = text
    return data


@data_register('data.refine', rewrite_func='forward', _concurrency_mode='process')
def remove_html_entity(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        data[input_key] = HTML_ENTITY_PATTERN.sub('', text)
    return data
