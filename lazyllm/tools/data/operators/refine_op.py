from ..base_data import data_register
import re


Refine = data_register.new_group('refine')


class RemoveExtraSpaces(Refine):
    def __init__(self, input_key='content', _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if text:
            data[self.input_key] = ' '.join(text.split())
        return data


class RemoveEmoji(Refine):
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

    def __init__(self, input_key='content', _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if text:
            data[self.input_key] = self.EMOJI_PATTERN.sub('', text)
        return data


class RemoveHtmlUrl(Refine):
    URL_PATTERN = re.compile(r'https?://\S+', flags=re.MULTILINE)
    HTML_PATTERN = re.compile(r'<.*?>')

    def __init__(self, input_key='content', _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if text:
            text = self.URL_PATTERN.sub('', text)
            text = self.HTML_PATTERN.sub('', text)
            data[self.input_key] = text
        return data


class RemoveHtmlEntity(Refine):
    HTML_ENTITIES = [
        'nbsp', 'lt', 'gt', 'amp', 'quot', 'apos', 'hellip', 'ndash', 'mdash',
        'lsquo', 'rsquo', 'ldquo', 'rdquo'
    ]
    ENTITY_PATTERNS = [
        fr'&{entity};'
        for entity in HTML_ENTITIES
    ] + [
        fr'＆{entity};'
        for entity in HTML_ENTITIES
    ] + [
        fr'&{entity}；'
        for entity in HTML_ENTITIES
    ] + [
        fr'＆{entity}；'
        for entity in HTML_ENTITIES
    ]
    HTML_ENTITY_PATTERN = re.compile('|'.join(ENTITY_PATTERNS))

    def __init__(self, input_key='content', _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        text = data.get(self.input_key, '')
        if text:
            data[self.input_key] = self.HTML_ENTITY_PATTERN.sub('', text)
        return data
