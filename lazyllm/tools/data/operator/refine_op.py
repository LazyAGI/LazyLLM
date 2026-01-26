from ..base_data import DataOperatorRegistry
import re


# Emoji pattern
EMOJI_PATTERN = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # Emoticons (faces)
    u'\U0001F300-\U0001F5FF'  # Symbols & pictographs (animals, weather, etc)
    u'\U0001F680-\U0001F6FF'  # Transport & map symbols
    u'\U0001F900-\U0001F9FF'  # Supplemental symbols
    u'\U0001FA00-\U0001FAFF'  # Extended pictographs
    u'\U0001F1E0-\U0001F1FF'  # Flags
    u'\U0001F3FB-\U0001F3FF'  # Skin tone modifiers
    u'\U0000FE00-\U0000FE0F'  # Variation selectors
    u'\U0001F000-\U0001F02F'  # Mahjong/Domino tiles
    u'\U0001F0A0-\U0001F0FF'  # Playing cards
    u'\U0001F200-\U0001F2FF'  # Enclosed ideographic supplement
    u'\U00002600-\U000026FF'  # Miscellaneous symbols (weather, zodiac, etc)
    u'\U00002700-\U000027BF'  # Dingbats
    '\\u200D'                 # Zero-width joiner
    '\\uFE0F'                 # Variation selector-16
    ']+'
    '|'
    '[\\U0001F1E6-\\U0001F1FF]{2}'  # Regional indicator pairs (flags)
    '|'
    '[\\u200D]',
    flags=re.UNICODE
)

# HTML url and tags pattern
URL_PATTERN = re.compile(r'https?://\S+', flags=re.MULTILINE)
HTML_PATTERN = re.compile(r'<.*?>')

# HTML entity pattern 
HTML_ENTITIES = [
    'nbsp', 'lt', 'gt', 'amp', 'quot', 'apos', 'hellip', 'ndash', 'mdash',
    'lsquo', 'rsquo', 'ldquo', 'rdquo'
]
entity_patterns = []
for entity in HTML_ENTITIES:
    entity_patterns.append(fr'&{entity};')      # &nbsp;
    entity_patterns.append(fr'＆{entity};')     # ＆nbsp; (full-width &)
    entity_patterns.append(fr'&{entity}；')     # &nbsp； (Chinese semicolon)
    entity_patterns.append(fr'＆{entity}；')    # ＆nbsp； (both full-width)
HTML_ENTITY_PATTERN = re.compile('|'.join(entity_patterns))


@DataOperatorRegistry.register
def remove_extra_spaces(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        data[input_key] = ' '.join(text.split())
    return data


@DataOperatorRegistry.register
def remove_emoji(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        data[input_key] = EMOJI_PATTERN.sub('', text)
    return data


@DataOperatorRegistry.register
def remove_html_url(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        text = URL_PATTERN.sub('', text)
        text = HTML_PATTERN.sub('', text)
        data[input_key] = text
    return data


@DataOperatorRegistry.register
def remove_html_entity(data, input_key='content'):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if text:
        data[input_key] = HTML_ENTITY_PATTERN.sub('', text)
    return data
